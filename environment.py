"""
车间调度环境模块
实现事件驱动的调度仿真环境，支持：
- 并行工序 A || (B->C)，同步后串行 D->E->F->G
- 设备通用性约束
- 换料时间约束
- 完整状态空间表示
"""
import numpy as np
from copy import deepcopy
from data_loader import load_all_data, needs_setup


# 订单工序状态
STATUS_NOT_STARTED = 0
STATUS_WAITING = 1      # 前序工序已完成，等待被调度
STATUS_IN_PROGRESS = 2
STATUS_DONE = 3

STAGES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
# A 和 (B->C) 并行，同步后 D->E->F->G
# 对每个订单，逻辑上有 7 个工序槽位
STAGE_INDEX = {s: i for i, s in enumerate(STAGES)}


class MachineState:
    def __init__(self, machine_id, stage):
        self.machine_id = machine_id
        self.stage = stage
        self.is_busy = False
        self.finish_time = 0.0        # 当前任务结束时刻
        self.total_proc_time = 0.0    # 累计加工时间（用于空闲率）
        self.last_product_type = None # 上次加工型号（用于换料判断）
        self.current_order = None     # 当前加工的订单ID


class OrderState:
    def __init__(self, order_dict, min_remaining):
        self.order_id = order_dict['order_id']
        self.product_type = order_dict['product_type']
        self.quantity = order_dict['quantity']
        self.mode = order_dict['mode']
        self.deadline_sec = order_dict['deadline_sec']
        self.min_remaining = min_remaining  # {stage: sec}

        # 工序状态矩阵 [7] for A,B,C,D,E,F,G
        self.stage_status = np.zeros(7, dtype=np.int32)  # 初始全部未开始
        # 各工序实际完成时刻
        self.stage_finish_time = np.full(7, np.inf)
        # 初始化：A、B 可以开始
        self.stage_status[STAGE_INDEX['A']] = STATUS_WAITING
        self.stage_status[STAGE_INDEX['B']] = STATUS_WAITING

        self.completion_time = None  # 最终完工时刻（G完成）

    def get_current_stage(self):
        """返回订单当前可被调度的工序列表"""
        available = []
        for s in ['A', 'B']:
            idx = STAGE_INDEX[s]
            if self.stage_status[idx] == STATUS_WAITING:
                available.append(s)
        # D-G：串行
        for s in ['D', 'E', 'F', 'G']:
            idx = STAGE_INDEX[s]
            if self.stage_status[idx] == STATUS_WAITING:
                available.append(s)
                break  # 只返回第一个待加工的串行工序
        # C：B完成后可开始
        if self.stage_status[STAGE_INDEX['B']] == STATUS_DONE and \
           self.stage_status[STAGE_INDEX['C']] == STATUS_WAITING:
            available.append('C')
        return available

    def complete_stage(self, stage, finish_time):
        """标记某工序完成，并更新后续工序状态"""
        idx = STAGE_INDEX[stage]
        self.stage_status[idx] = STATUS_DONE
        self.stage_finish_time[idx] = finish_time

        if stage == 'B':
            # B完成 -> C可以开始
            self.stage_status[STAGE_INDEX['C']] = STATUS_WAITING

        elif stage == 'C':
            # C完成，检查A是否也完成
            self._try_unlock_D()

        elif stage == 'A':
            # A完成，检查C是否也完成
            self._try_unlock_D()

        elif stage == 'D':
            self.stage_status[STAGE_INDEX['E']] = STATUS_WAITING
        elif stage == 'E':
            self.stage_status[STAGE_INDEX['F']] = STATUS_WAITING
        elif stage == 'F':
            self.stage_status[STAGE_INDEX['G']] = STATUS_WAITING
        elif stage == 'G':
            self.completion_time = finish_time

    def _try_unlock_D(self):
        """A 和 C 都完成后才能开始 D"""
        a_done = self.stage_status[STAGE_INDEX['A']] == STATUS_DONE
        c_done = self.stage_status[STAGE_INDEX['C']] == STATUS_DONE
        if a_done and c_done:
            self.stage_status[STAGE_INDEX['D']] = STATUS_WAITING

    def is_complete(self):
        return self.stage_status[STAGE_INDEX['G']] == STATUS_DONE


class SchedulingEnv:
    def __init__(self, data):
        self.data = data
        self.n_orders = len(data['orders'])
        self.n_machines = sum(len(v) for v in data['stage_to_machines'].values())
        self.product_types = data['product_types']
        self.n_types = len(self.product_types)
        self.reset()

    def reset(self):
        """重置环境到初始状态"""
        self.current_time = 0.0
        # 初始化订单状态
        self.orders = [
            OrderState(o, self.data['min_remaining'][o['order_id']])
            for o in self.data['orders']
        ]
        self.order_id_to_idx = {o.order_id: i for i, o in enumerate(self.orders)}

        # 初始化设备状态
        self.machines = {}
        for stage, machine_list in self.data['stage_to_machines'].items():
            for m in machine_list:
                self.machines[m] = MachineState(m, stage)

        # 调度历史（用于甘特图）
        self.schedule_history = []  # list of (order_id, machine, stage, start, end, is_setup)
        self.done = False
        return self._get_state()

    def _get_available_orders(self):
        """
        获取可被调度的 (order_idx, stage) 对：
        - 对应工序状态为 WAITING
        - 当前工序有至少一台空闲设备可以加工该订单型号
        """
        available = []
        for i, order in enumerate(self.orders):
            if order.is_complete():
                continue
            stages = order.get_current_stage()
            for stage in stages:
                if order.stage_status[STAGE_INDEX[stage]] != STATUS_WAITING:
                    continue
                # 检查该工序是否有空闲设备可加工该型号
                for m_id in self.data['stage_to_machines'].get(stage, []):
                    m = self.machines[m_id]
                    if not m.is_busy and order.product_type in self.data['compatibility'][m_id]:
                        available.append((i, stage))
                        break
        return available

    def _get_available_machines(self, order_idx, stage):
        """获取可以加工指定订单当前工序的空闲设备列表"""
        order = self.orders[order_idx]
        machines = []
        for m_id in self.data['stage_to_machines'].get(stage, []):
            m = self.machines[m_id]
            if not m.is_busy and order.product_type in self.data['compatibility'][m_id]:
                machines.append(m_id)
        return machines

    def assign(self, order_idx, machine_id):
        """
        将订单分配到指定设备，返回 (done, info)
        """
        order = self.orders[order_idx]
        machine = self.machines[machine_id]
        stage = machine.stage

        # 计算换料时间
        setup_min = needs_setup(
            stage,
            machine.last_product_type,
            order.product_type,
            self.data['setup_rules']
        )
        setup_sec = setup_min * 60.0

        # 如果换料前空闲时间 >= 换料时间，则换料时间可忽略
        idle_time = self.current_time - machine.finish_time
        if idle_time >= setup_sec:
            setup_sec = 0.0

        # 加工时间
        proc_sec = self.data['proc_time'][machine_id][order.product_type] * order.quantity
        start_time = self.current_time
        finish_time = start_time + setup_sec + proc_sec

        # 更新设备状态
        machine.is_busy = True
        machine.finish_time = finish_time
        machine.total_proc_time += proc_sec
        machine.last_product_type = order.product_type
        machine.current_order = order.order_id

        # 更新订单工序状态
        order.stage_status[STAGE_INDEX[stage]] = STATUS_IN_PROGRESS

        # 记录历史（甘特图用）
        if setup_sec > 0:
            self.schedule_history.append({
                'order_id': order.order_id,
                'machine': machine_id,
                'stage': stage,
                'start': start_time,
                'end': start_time + setup_sec,
                'type': 'setup',
                'product_type': order.product_type,
            })
        self.schedule_history.append({
            'order_id': order.order_id,
            'machine': machine_id,
            'stage': stage,
            'start': start_time + setup_sec,
            'end': finish_time,
            'type': 'process',
            'product_type': order.product_type,
        })

        return finish_time

    def step_to_next_event(self):
        """
        推进时间到下一个设备完成事件
        返回释放的设备列表
        """
        busy_machines = [(m.finish_time, m_id)
                         for m_id, m in self.machines.items() if m.is_busy]
        if not busy_machines:
            # 没有设备在工作，检查是否全部完成
            all_done = all(o.is_complete() for o in self.orders)
            self.done = all_done
            return []

        next_time = min(t for t, _ in busy_machines)
        self.current_time = next_time

        # 释放所有在 next_time 完成的设备
        released = []
        for m_id, m in self.machines.items():
            if m.is_busy and abs(m.finish_time - next_time) < 1e-6:
                m.is_busy = False
                # 完成对应订单的工序
                order_id = m.current_order
                order = self.orders[self.order_id_to_idx[order_id]]
                order.complete_stage(m.stage, next_time)
                m.current_order = None
                released.append(m_id)

        return released

    def _get_state(self):
        """
        构造状态向量，供神经网络使用
        返回:
          order_features: [n_orders, order_feat_dim]
          machine_features: [n_machines, machine_feat_dim]
          global_features: [global_feat_dim]
        """
        # --- 订单特征 ---
        # [type_onehot(15), qty_norm, mode(1), deadline_norm, stage_status(7), min_remaining_norm(7)]
        order_features = []
        for order in self.orders:
            type_onehot = np.zeros(self.n_types)
            type_onehot[self.product_types.index(order.product_type)] = 1.0
            qty_norm = order.quantity / 2000.0
            mode_val = 1.0 if order.mode == 'MTO' else 0.0
            deadline_norm = order.deadline_sec / (4320 * 60)
            stage_status = order.stage_status.astype(np.float32) / 3.0
            # 最短剩余加工时间归一化（使用最大值）
            min_rem = np.array([order.min_remaining[s] for s in STAGES], dtype=np.float32)
            min_rem_norm = min_rem / (max(min_rem.max(), 1))
            feat = np.concatenate([
                type_onehot, [qty_norm, mode_val, deadline_norm],
                stage_status, min_rem_norm
            ])
            order_features.append(feat)
        order_features = np.stack(order_features).astype(np.float32)

        # --- 设备特征 ---
        # [stage_onehot(7), n_compatible_types_norm, idle_ratio, is_busy, last_type_onehot(15)]
        machine_list = []
        for stage in STAGES:
            for m_id in self.data['stage_to_machines'].get(stage, []):
                machine_list.append(m_id)
        self._machine_list = machine_list

        machine_features = []
        for m_id in machine_list:
            m = self.machines[m_id]
            stage_onehot = np.zeros(7)
            stage_onehot[STAGE_INDEX[m.stage]] = 1.0
            n_compatible = len(self.data['compatibility'][m_id]) / self.n_types
            idle_ratio = 1.0 - (m.total_proc_time / max(self.current_time, 1))
            is_busy = float(m.is_busy)
            last_type_onehot = np.zeros(self.n_types)
            if m.last_product_type is not None and m.last_product_type in self.product_types:
                last_type_onehot[self.product_types.index(m.last_product_type)] = 1.0
            feat = np.concatenate([
                stage_onehot, [n_compatible, idle_ratio, is_busy],
                last_type_onehot
            ])
            machine_features.append(feat)
        machine_features = np.stack(machine_features).astype(np.float32)

        # --- 全局特征 ---
        total_proc = sum(m.total_proc_time for m in self.machines.values())
        total_proc_norm = total_proc / max(self.current_time * self.n_machines, 1)
        total_delay = self._compute_total_delay()
        global_features = np.array([
            total_proc_norm,
            total_delay / 1e6,
            self.current_time / (4320 * 60),
        ], dtype=np.float32)

        return {
            'order_features': order_features,
            'machine_features': machine_features,
            'global_features': global_features,
        }

    def _compute_total_delay(self):
        """计算当前拖期总和"""
        total = 0.0
        for order in self.orders:
            if order.is_complete():
                delay = max(0, order.completion_time - order.deadline_sec)
                total += delay
        return total

    def compute_reward(self):
        """计算终端奖励"""
        mto_delay = 0.0
        mts_delay = 0.0
        makespan = 0.0
        for order in self.orders:
            if order.completion_time is not None:
                makespan = max(makespan, order.completion_time)
                delay = max(0, order.completion_time - order.deadline_sec)
                if order.mode == 'MTO':
                    mto_delay += delay
                else:
                    mts_delay += delay
        # 终端奖励 = -(0.7*MTO拖期 + 0.3*MTS拖期) - 最大完工时间
        reward = -(0.7 * mto_delay + 0.3 * mts_delay) - makespan
        return reward / 1e6  # 归一化量级

    def get_heuristic_scores_order(self, order_idx):
        """计算订单选择智能体的启发式得分 [type_score, delay_score]"""
        order = self.orders[order_idx]
        # 规则一：订单类别优先级
        type_score = 0.5 if order.mode == 'MTO' else 0.3

        # 规则二：交货紧迫度
        # 取当前工序的最短剩余加工时间
        current_stages = order.get_current_stage()
        if current_stages:
            min_rem = order.min_remaining.get(current_stages[0], 0)
        else:
            min_rem = 0
        delta_t_sec = order.deadline_sec - self.current_time - min_rem
        delta_t_min = delta_t_sec / 60.0
        if delta_t_min <= 0:
            delay_score = 1.0
        elif delta_t_min <= 1440:
            delay_score = 0.7
        else:
            delay_score = 0.3

        return np.array([type_score, delay_score], dtype=np.float32)

    def get_heuristic_scores_machine(self, order_idx, machine_ids):
        """
        计算设备分配智能体的启发式得分
        返回 [n_machines, 4] (专用性, 空闲率, 换料成本, 加工时间)
        """
        order = self.orders[order_idx]
        scores = []
        proc_times = []
        for m_id in machine_ids:
            m = self.machines[m_id]
            # 规则一：专用性得分
            specif = 1.0 - len(self.data['compatibility'][m_id]) / self.n_types

            # 规则二：空闲率得分
            util = 1.0 - m.total_proc_time / max(self.current_time, 1)

            # 规则三：换料成本
            setup_min = needs_setup(
                m.stage, m.last_product_type, order.product_type, self.data['setup_rules']
            )
            idle_time = self.current_time - m.finish_time
            if setup_min == 0 or idle_time >= setup_min * 60:
                setup_score = 0.5
            else:
                setup_score = 0.3

            # 规则四：加工时间（先收集，后归一化）
            pt = self.data['proc_time'][m_id][order.product_type] * order.quantity
            proc_times.append(pt)

            scores.append([specif, util, setup_score, 0.0])  # 先占位

        # 规则四归一化
        if len(proc_times) > 1:
            sorted_unique = sorted(set(proc_times))
            rank_map = {v: (i + 1) / len(sorted_unique) for i, v in enumerate(sorted_unique)}
            for i, pt in enumerate(proc_times):
                scores[i][3] = rank_map[pt]
        elif len(proc_times) == 1:
            scores[0][3] = 1.0

        return np.array(scores, dtype=np.float32)

    def is_done(self):
        return all(o.is_complete() for o in self.orders)

    def get_metrics(self):
        """计算最终调度指标"""
        makespan = 0.0
        delays = []
        for order in self.orders:
            if order.completion_time is not None:
                makespan = max(makespan, order.completion_time)
                delay = max(0, order.completion_time - order.deadline_sec)
                delays.append({
                    'order_id': order.order_id,
                    'mode': order.mode,
                    'deadline_sec': order.deadline_sec,
                    'completion_time': order.completion_time,
                    'delay_sec': delay,
                    'delay_min': delay / 60.0,
                })
        mto_delay = sum(d['delay_sec'] for d in delays if d['mode'] == 'MTO')
        mts_delay = sum(d['delay_sec'] for d in delays if d['mode'] == 'MTS')
        return {
            'makespan': makespan,
            'makespan_min': makespan / 60.0,
            'total_delay': mto_delay + mts_delay,
            'mto_delay': mto_delay,
            'mts_delay': mts_delay,
            'order_delays': delays,
        }


if __name__ == '__main__':
    from data_loader import load_all_data
    data = load_all_data('/mnt/user-data/uploads/data.xlsx')
    env = SchedulingEnv(data)
    state = env.reset()
    print("订单特征维度:", state['order_features'].shape)
    print("设备特征维度:", state['machine_features'].shape)
    print("全局特征维度:", state['global_features'].shape)
    print("可调度订单数:", len(env._get_available_orders()))
    print("环境初始化成功 ✓")
    
