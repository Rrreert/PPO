"""
车间调度环境（事件驱动，优化版）
工序顺序：A 并 (B→C)，汇合后 D→E→F→G
"""
import numpy as np
from data_loader import load_all_data, get_setup_seconds

STAGES    = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
STAGE_IDX = {s: i for i, s in enumerate(STAGES)}

PREREQ = {
    'A': [], 'B': [], 'C': ['B'],
    'D': ['A', 'C'], 'E': ['D'], 'F': ['E'], 'G': ['F'],
}

NOT_STARTED = 0
WAITING     = 1
IN_PROCESS  = 2
DONE        = 3


class ShopFloorEnv:
    def __init__(self, data):
        self.data         = data
        self.n_orders     = len(data['orders'])
        self.stages       = STAGES
        self.stage_devices= data['stage_devices']
        self.all_devices  = list(data['device_compatibility'].keys())
        self.n_devices    = len(self.all_devices)
        self.device_index = {d: i for i, d in enumerate(self.all_devices)}
        self._n_types     = len(data['product_types'])

    def reset(self):
        self.current_time       = 0.0
        self.order_stage_status = np.zeros((self.n_orders, 7), dtype=np.int32)
        self.device_busy_until  = {d: 0.0 for d in self.all_devices}
        self.device_last_type   = {d: None for d in self.all_devices}
        self.device_total_proc  = {d: 0.0 for d in self.all_devices}
        self.order_finish_time  = [None] * self.n_orders
        self.stage_finish_time  = [{} for _ in range(self.n_orders)]
        self.schedule_log       = []

        # active events heap: (finish_time, order_idx, stage)
        self._active_events = []

        for i in range(self.n_orders):
            self.order_stage_status[i][STAGE_IDX['A']] = WAITING
            self.order_stage_status[i][STAGE_IDX['B']] = WAITING

        # cache for _get_state
        self._state_dirty = True
        self._cached_state = None
        return self._get_state()

    # ── 状态查询 ──────────────────────────────────────────────────
    def _update_waiting_status(self):
        for i in range(self.n_orders):
            for stage in STAGES:
                si = STAGE_IDX[stage]
                if self.order_stage_status[i][si] in (IN_PROCESS, DONE):
                    continue
                if self.order_stage_status[i][si] == NOT_STARTED:
                    if all(self.order_stage_status[i][STAGE_IDX[p]] == DONE
                           for p in PREREQ[stage]):
                        self.order_stage_status[i][si] = WAITING
        self._state_dirty = True

    def get_schedulable(self):
        schedulable = []
        for i in range(self.n_orders):
            for stage in STAGES:
                si = STAGE_IDX[stage]
                if self.order_stage_status[i][si] != WAITING:
                    continue
                ptype = self.data['orders'][i]['product_type']
                if self._get_idle_compatible(stage, ptype):
                    schedulable.append((i, stage))
        return schedulable

    def _get_idle_compatible(self, stage, ptype):
        result = []
        for d in self.stage_devices.get(stage, []):
            if (ptype in self.data['device_compatibility'][d] and
                    self.device_busy_until[d] <= self.current_time):
                result.append(d)
        return result

    def get_compatible_idle_devices(self, order_idx, stage):
        ptype = self.data['orders'][order_idx]['product_type']
        return self._get_idle_compatible(stage, ptype)

    # ── 调度执行 ──────────────────────────────────────────────────
    def dispatch(self, order_idx, stage, device):
        order    = self.data['orders'][order_idx]
        ptype    = order['product_type']
        qty      = order['quantity']
        proc_sec = self.data['processing_time'][(device, ptype)] * qty

        idle_t    = max(0.0, self.current_time - self.device_busy_until[device])
        setup_sec = get_setup_seconds(
            self.data, stage, self.device_last_type[device], ptype, idle_t)

        start_proc = self.current_time + setup_sec
        end_time   = start_proc + proc_sec

        if setup_sec > 0:
            self.schedule_log.append({
                'device': device, 'order_id': order['order_id'],
                'stage': stage, 'start': self.current_time,
                'end': start_proc, 'is_setup': True
            })
        self.schedule_log.append({
            'device': device, 'order_id': order['order_id'],
            'stage': stage, 'start': start_proc,
            'end': end_time, 'is_setup': False
        })

        self.device_busy_until[device]  = end_time
        self.device_last_type[device]   = ptype
        self.device_total_proc[device] += proc_sec

        si = STAGE_IDX[stage]
        self.order_stage_status[order_idx][si] = IN_PROCESS
        self.stage_finish_time[order_idx][stage] = end_time
        self._active_events.append((end_time, order_idx, stage))
        self._state_dirty = True
        return end_time

    # ── 时间推进 ──────────────────────────────────────────────────
    def advance_time(self):
        active = [(ft, oi, st)
                  for (ft, oi, st) in self._active_events
                  if self.order_stage_status[oi][STAGE_IDX[st]] == IN_PROCESS
                  and ft > self.current_time]
        if not active:
            return None

        next_time = min(ft for ft, _, _ in active)
        self.current_time = next_time

        for ft, oi, st in active:
            if ft <= self.current_time:
                self.order_stage_status[oi][STAGE_IDX[st]] = DONE

        self._update_waiting_status()

        for i in range(self.n_orders):
            if (self.order_finish_time[i] is None and
                    self.order_stage_status[i][STAGE_IDX['G']] == DONE):
                self.order_finish_time[i] = self.current_time

        self._state_dirty = True
        return self.current_time

    def is_done(self):
        return all(self.order_stage_status[i][STAGE_IDX['G']] == DONE
                   for i in range(self.n_orders))

    # ── 奖励 ─────────────────────────────────────────────────────
    def compute_reward(self):
        makespan  = max((t for t in self.order_finish_time if t is not None), default=0.0)
        mto_delay = mts_delay = 0.0
        for i, order in enumerate(self.data['orders']):
            ft = self.order_finish_time[i]
            if ft is None:
                continue
            delay = max(0.0, ft - order['due_time'])
            if order['order_type'] == 'MTO':
                mto_delay += delay
            else:
                mts_delay += delay
        reward = -(0.7 * mto_delay + 0.3 * mts_delay) - makespan
        return reward, makespan, mto_delay, mts_delay

    # ── 状态向量（带缓存）────────────────────────────────────────
    def _get_state(self):
        if not self._state_dirty and self._cached_state is not None:
            return self._cached_state

        n_types = self._n_types

        # 订单特征
        order_feat = np.zeros((self.n_orders, 25), dtype=np.float32)
        for i, order in enumerate(self.data['orders']):
            ti = self.data['product_types'].index(order['product_type'])
            order_feat[i, ti] = 1.0
            order_feat[i, n_types]     = order['quantity'] / 2000.0
            order_feat[i, n_types + 1] = 1.0 if order['order_type'] == 'MTO' else 0.0
            order_feat[i, n_types + 2] = order['due_time'] / (4320.0 * 60)
            order_feat[i, n_types + 3: n_types + 10] = self.order_stage_status[i] / 3.0

        # 设备特征
        device_feat = np.zeros((self.n_devices, 25), dtype=np.float32)
        total_t     = max(self.current_time, 1.0)
        for j, d in enumerate(self.all_devices):
            stage = self.data['device_to_stage'][d]
            device_feat[j, STAGE_IDX[stage]] = 1.0
            device_feat[j, 7]  = len(self.data['device_compatibility'][d]) / n_types
            device_feat[j, 8]  = min(1.0, self.device_total_proc[d] / total_t)
            lt = self.device_last_type[d]
            if lt is not None:
                device_feat[j, 9 + self.data['product_types'].index(lt)] = 1.0
            else:
                device_feat[j, 9 + n_types] = 1.0   # none-flag

        # 全局特征
        total_proc = sum(self.device_total_proc.values())
        total_delay = sum(
            max(0.0, (self.order_finish_time[i] or 0.0) - self.data['orders'][i]['due_time'])
            for i in range(self.n_orders)
        )
        global_feat = np.array([
            self.current_time / (4320.0 * 60),
            total_proc / max(1.0, total_t * self.n_devices),
            total_delay / max(1.0, total_t * self.n_orders),
        ], dtype=np.float32)

        state = {
            'order_feat': order_feat,
            'device_feat': device_feat,
            'global_feat': global_feat,
            'order_stage_status': self.order_stage_status.copy(),
        }
        self._cached_state = state
        self._state_dirty  = False
        return state

    def get_min_remaining(self, order_idx, stage):
        oid = self.data['orders'][order_idx]['order_id']
        return self.data['min_remaining'][oid].get(stage, 0)
