"""
车间调度环境：事件驱动仿真
状态、动作、奖励均按 PDF 规格实现
"""
import numpy as np
from copy import deepcopy
from data_loader import load_all, OPS, PREREQS, PRODUCT_TYPES

# 加工状态常量
ST_NOT_STARTED  = 0
ST_WAITING      = 1   # 前序已完成，等待调度
ST_IN_PROGRESS  = 2
ST_DONE         = 3

data = load_all()
COMPATIBILITY = data['compatibility']
MACHINE_OP    = data['machine_op']
OPS_MACHINES  = data['ops_machines']
PROC_TIME     = data['proc_time']
SETUP_TIME    = data['setup_time']
ORDERS        = data['orders']
MIN_REMAINING = data['min_remaining']
N_ORDERS      = len(ORDERS)
ALL_MACHINES  = [m for op in OPS for m in OPS_MACHINES[op]]
N_MACHINES    = len(ALL_MACHINES)
MACHINE_IDX   = {m: i for i, m in enumerate(ALL_MACHINES)}


def _need_setup(op, prev_type, new_type):
    """判断是否需要换料（返回换料时间秒数，0表示不需要）"""
    if op not in SETUP_TIME:
        return 0
    if prev_type is None or prev_type == new_type:
        return 0
    info = SETUP_TIME[op]
    if info['groups'] is None:
        # 所有型号之间都需要换料
        return info['time_min'] * 60
    # 分组换料（工序E）
    def get_group(pt):
        for g in info['groups']:
            if pt in g:
                return g
        return None
    if get_group(prev_type) == get_group(new_type):
        return 0
    return info['time_min'] * 60


class Machine:
    def __init__(self, machine_id):
        self.id = machine_id
        self.op = MACHINE_OP[machine_id]
        self.can_process = COMPATIBILITY[machine_id]
        self.status = 'idle'          # 'idle' | 'busy'
        self.finish_time = 0          # 绝对秒数
        self.total_busy = 0           # 累计已加工时间（秒）
        self.last_product_type = None # 上一次加工的产品型号
        self.history = []             # [(order_id, start, end, is_setup)]

    def can_take(self, product_type):
        return product_type in self.can_process and self.status == 'idle'

    def assign(self, order_id, product_type, quantity, current_time):
        """分配订单到此设备，返回实际开始时间和结束时间"""
        proc_sec = PROC_TIME[(self.id, product_type)] * quantity

        # 计算换料时间
        setup_sec = _need_setup(self.op, self.last_product_type, product_type)
        # 若空闲时间 >= 换料时间，则换料时间忽略
        idle_since = self.finish_time  # 上次结束时间（设备空闲起点）
        idle_duration = current_time - idle_since
        if idle_duration >= setup_sec:
            setup_sec = 0

        start_time = current_time
        if setup_sec > 0:
            # 换料段
            setup_end = start_time + setup_sec
            self.history.append((order_id, start_time, setup_end, True))
            start_time = setup_end

        end_time = start_time + proc_sec
        self.history.append((order_id, start_time, end_time, False))

        self.status = 'busy'
        self.finish_time = end_time
        self.total_busy += proc_sec
        self.last_product_type = product_type
        return start_time, end_time


class OrderState:
    def __init__(self, order_dict):
        self.id           = order_dict['order_id']
        self.product_type = order_dict['product_type']
        self.quantity     = order_dict['quantity']
        self.mode         = order_dict['mode']
        self.due_time     = order_dict['due_time'] * 60  # 转换为秒
        # 每道工序的状态
        self.op_status = {op: ST_NOT_STARTED for op in OPS}
        self.op_start  = {op: None for op in OPS}
        self.op_end    = {op: None for op in OPS}
        self.op_machine= {op: None for op in OPS}
        # 初始化：A 和 B 可以立即开始（无前序）
        self.op_status['A'] = ST_WAITING
        self.op_status['B'] = ST_WAITING

    def update_waiting(self):
        """根据已完成工序，更新哪些工序进入等待状态"""
        for op, prereqs in PREREQS.items():
            if self.op_status[op] == ST_NOT_STARTED:
                if all(self.op_status[p] == ST_DONE for p in prereqs):
                    self.op_status[op] = ST_WAITING

    def finish_op(self, op, end_time, machine_id):
        self.op_status[op] = ST_DONE
        self.op_end[op]    = end_time
        self.op_machine[op]= machine_id
        self.update_waiting()

    def start_op(self, op, start_time, machine_id):
        self.op_status[op]  = ST_IN_PROGRESS
        self.op_start[op]   = start_time
        self.op_machine[op] = machine_id

    def completion_time(self):
        """返回最后工序G的完成时间，未完成则返回None"""
        return self.op_end.get('G')

    def tardiness(self, current_time=None):
        ct = self.completion_time()
        if ct is None:
            # 未完成，用当前时间估算
            ct = current_time if current_time else 0
        return max(0, ct - self.due_time)


class WorkshopEnv:
    """
    车间调度环境
    - step() 输入: (order_idx, machine_idx) —— 已解码的索引
    - 返回: (obs_dict, reward, done, info)
    - 采用事件驱动：每次 step 后自动推进到下一个决策点
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.machines = {m: Machine(m) for m in ALL_MACHINES}
        self.order_states = [OrderState(o) for o in ORDERS]
        self.current_time = 0.0   # 秒
        self.done = False
        self.event_queue = []     # 按 finish_time 排序的 (time, machine_id)
        self._advance_time_if_needed()
        return self._get_obs()

    # ------------------------------------------------------------------ #
    # 核心调度逻辑
    # ------------------------------------------------------------------ #

    def _machines_free_for(self, op, product_type):
        """返回工序 op 中空闲且能加工 product_type 的设备列表"""
        return [
            m for m in OPS_MACHINES[op]
            if self.machines[m].can_take(product_type)
        ]

    def _schedulable_orders(self):
        """
        返回可调度订单列表：
        已完成前序工序 & 当前工序处于 WAITING & 当前工序有空闲可用设备
        """
        result = []
        for os in self.order_states:
            for op in OPS:
                if os.op_status[op] == ST_WAITING:
                    free = self._machines_free_for(op, os.product_type)
                    if free:
                        result.append((os, op))
                        break   # 一个订单只暴露一个待调度工序（最靠前的 WAITING）
        return result

    def step(self, order_os, op, machine_id):
        """
        执行一个调度决策
        order_os: OrderState 对象
        op: 工序字符串
        machine_id: 设备id字符串
        """
        qty = order_os.quantity
        pt  = order_os.product_type
        m   = self.machines[machine_id]

        start, end = m.assign(order_os.id, pt, qty, self.current_time)
        order_os.start_op(op, start, machine_id)

        # 注册完成事件
        self.event_queue.append((end, machine_id, order_os.id, op))
        self.event_queue.sort(key=lambda x: x[0])

        # 返回观察（调用方自己决定是否需要推进时间）
        obs = self._get_obs()
        reward = 0.0
        done = self._check_done()
        if done:
            reward = self._terminal_reward()
            self.done = True
        return obs, reward, done

    def advance_to_next_event(self):
        """推进时间到下一个设备释放事件，处理完成回调"""
        if not self.event_queue:
            return False
        end_time, machine_id, order_id, op = self.event_queue.pop(0)
        self.current_time = end_time
        m = self.machines[machine_id]
        m.status = 'idle'
        # 找到对应 order_state
        os = next(o for o in self.order_states if o.id == order_id)
        os.finish_op(op, end_time, machine_id)
        return True

    def _advance_time_if_needed(self):
        """若当前没有可调度订单且还有未完成的，推进时间"""
        while not self._schedulable_orders() and not self._all_done():
            if not self.advance_to_next_event():
                break
        if self._all_done() and not self.done:
            self.done = True

    def _all_done(self):
        return all(os.op_status['G'] == ST_DONE for os in self.order_states)

    def _check_done(self):
        return self._all_done()

    def _terminal_reward(self):
        mto_tard = sum(
            os.tardiness() for os in self.order_states
            if next(o for o in ORDERS if o['order_id'] == os.id)['mode'] == 'MTO'
        )
        mts_tard = sum(
            os.tardiness() for os in self.order_states
            if next(o for o in ORDERS if o['order_id'] == os.id)['mode'] == 'MTS'
        )
        makespan = max(os.completion_time() or 0 for os in self.order_states)
        # 转换为分钟以使数值合理
        mto_tard /= 60
        mts_tard /= 60
        makespan /= 60
        reward = -(0.7 * mto_tard + 0.3 * mts_tard) - makespan
        return reward

    # ------------------------------------------------------------------ #
    # 观察空间构建
    # ------------------------------------------------------------------ #

    def _get_obs(self):
        """
        返回 dict，包含：
        - order_features: [N_ORDERS, order_feat_dim]
        - machine_features: [N_MACHINES, machine_feat_dim]
        - global_features: [2]
        - schedulable: list of (order_state, op) tuples
        - free_machines_for: callable
        """
        # 订单特征
        order_feats = []
        for os in self.order_states:
            pt_idx = PRODUCT_TYPES.index(os.product_type) / 14.0  # 归一化
            qty_norm = os.quantity / 1280.0
            mode_val = 1.0 if os.mode == 'MTO' else 0.0
            due_norm = os.due_time / (4320 * 60)
            # 工序状态（7道工序，4个状态）one-hot 展平 7*4=28
            op_st = []
            for op in OPS:
                st = os.op_status[op]
                oh = [0.0] * 4
                oh[st] = 1.0
                op_st.extend(oh)
            # 当前时刻距交货时间（归一化）
            time_to_due = max(0, os.due_time - self.current_time / 60) / (4320)
            order_feats.append([pt_idx, qty_norm, mode_val, due_norm, time_to_due] + op_st)

        # 机器特征
        machine_feats = []
        for m_id in ALL_MACHINES:
            m = self.machines[m_id]
            op_idx = OPS.index(m.op) / 6.0
            status_val = 0.0 if m.status == 'idle' else 1.0
            busy_ratio = m.total_busy / max(self.current_time, 1.0)
            # 可加工型号数（归一化）
            n_types = len(m.can_process) / 15.0
            # 到空闲的剩余时间（归一化）
            remain = max(0, m.finish_time - self.current_time) / (4320 * 60)
            machine_feats.append([op_idx, status_val, busy_ratio, n_types, remain])

        # 全局特征
        total_busy = sum(m.total_busy for m in self.machines.values())
        total_tard = sum(os.tardiness(self.current_time) for os in self.order_states) / 60
        global_feats = [
            total_busy / max(self.current_time * N_MACHINES, 1.0),
            total_tard / (N_ORDERS * 4320),
        ]

        schedulable = self._schedulable_orders()

        return {
            'order_features': np.array(order_feats, dtype=np.float32),
            'machine_features': np.array(machine_feats, dtype=np.float32),
            'global_features': np.array(global_feats, dtype=np.float32),
            'schedulable': schedulable,
            'current_time': self.current_time,
        }

    # ------------------------------------------------------------------ #
    # 启发式规则得分（供 PPO 网络使用）
    # ------------------------------------------------------------------ #

    def heuristic_order_scores(self, schedulable):
        """
        为每个可调度订单计算启发式得分向量 [type_score, delay_score]
        返回 np.array [n_schedulable, 2]
        """
        scores = []
        for os, op in schedulable:
            # 规则一：订单类别优先级
            type_score = 0.5 if os.mode == 'MTO' else 0.3
            # 规则二：交货紧迫度
            min_rem = MIN_REMAINING[os.id][op]  # 秒
            delta_t = os.due_time - (self.current_time / 60) - (min_rem / 60)  # 分钟
            if delta_t <= 0:
                delay_score = 1.0
            elif delta_t <= 1440:
                delay_score = 0.7
            else:
                delay_score = 0.3
            scores.append([type_score, delay_score])
        return np.array(scores, dtype=np.float32)

    def heuristic_machine_scores(self, os, op, free_machines):
        """
        为每台空闲可用设备计算启发式得分向量 [specif, util, setup, proc]
        返回 np.array [n_free, 4]
        """
        if not free_machines:
            return np.zeros((0, 4), dtype=np.float32)

        scores = []
        proc_times = []
        for m_id in free_machines:
            m = self.machines[m_id]
            # 规则一：专用性得分
            specif = 1.0 - len(m.can_process) / 15.0
            # 规则二：空闲率得分
            util = 1.0 - m.total_busy / max(self.current_time, 1.0)
            # 规则三：换料成本得分
            setup_sec = _need_setup(op, m.last_product_type, os.product_type)
            idle_dur  = self.current_time - m.finish_time
            if setup_sec == 0 or idle_dur >= setup_sec:
                setup_score = 0.5
            else:
                setup_score = 0.3
            # 规则四：加工时间（先收集）
            pt_val = PROC_TIME.get((m_id, os.product_type), 9999)
            proc_times.append(pt_val)
            scores.append([specif, util, setup_score, 0.0])  # proc暂填0

        # 归一化加工时间得分（平均分布到[0,1]）
        sorted_times = sorted(set(proc_times))
        rank_map = {}
        n = len(proc_times)
        # 对重复值使用相同得分
        seen = {}
        rank_counter = [0]
        for t in sorted(proc_times):
            if t not in seen:
                rank_counter[0] += 1
                seen[t] = rank_counter[0]
        max_rank = max(seen.values())
        for i, t in enumerate(proc_times):
            scores[i][3] = seen[t] / max_rank  # 加工时间越短得分越低越好（从小到大）

        return np.array(scores, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 甘特图数据导出
    # ------------------------------------------------------------------ #

    def get_gantt_data(self):
        gantt = []
        for m_id, m in self.machines.items():
            for entry in m.history:
                order_id, start, end, is_setup = entry
                gantt.append({
                    'machine': m_id,
                    'op': m.op,
                    'order_id': order_id,
                    'start': start,
                    'end': end,
                    'is_setup': is_setup,
                })
        return gantt

    def get_metrics(self):
        makespan = max((os.completion_time() or 0) for os in self.order_states) / 60
        tardiness = {
            os.id: os.tardiness() / 60 for os in self.order_states
        }
        mto_tard = sum(
            v for os, v in zip(self.order_states, tardiness.values())
            if os.mode == 'MTO'
        )
        mts_tard = sum(
            v for os, v in zip(self.order_states, tardiness.values())
            if os.mode == 'MTS'
        )
        return {
            'makespan': makespan,
            'tardiness': tardiness,
            'mto_tardiness': mto_tard,
            'mts_tardiness': mts_tard,
        }


if __name__ == '__main__':
    env = WorkshopEnv()
    obs = env.reset()
    print("order_features shape:", obs['order_features'].shape)
    print("machine_features shape:", obs['machine_features'].shape)
    print("schedulable count:", len(obs['schedulable']))
    print("global:", obs['global_features'])
