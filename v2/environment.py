"""车间仿真环境"""
import numpy as np
import copy

# 工序依赖关系
# 每个订单需完成: A, B, C, D, E, F, G
# 约束: D 需要 A 完成 且 C 完成（C 依赖 B）
# 即: A 和 (B->C) 并行, 之后 D->E->F->G 串行
NEXT_OP = {'A': 'D', 'B': 'C', 'C': 'D', 'D': 'E', 'E': 'F', 'F': 'G', 'G': None}
PREREQ = {
    'A': [],
    'B': [],
    'C': ['B'],
    'D': ['A', 'C'],
    'E': ['D'],
    'F': ['E'],
    'G': ['F'],
}

class OrderState:
    def __init__(self, order_data):
        self.id = order_data['id']
        self.product_type = order_data['type']
        self.qty = order_data['qty']
        self.mode = order_data['mode']   # MTO / MTS
        self.due = order_data['due']     # 秒
        self.arrive = order_data['arrive']
        self.priority = 1.0 if self.mode == 'MTO' else 0.5

        # 各工序状态: 'pending' / 'ready' / 'processing' / 'done'
        self.op_status = {op: 'pending' for op in ['A','B','C','D','E','F','G']}
        self.op_status['A'] = 'ready'
        self.op_status['B'] = 'ready'

        self.op_start = {}   # op -> 开始时间
        self.op_end = {}     # op -> 结束时间
        self.op_device = {}  # op -> 设备id (可能有多次,取最后)
        self.op_qty_done = {}  # op -> 已完成数量(故障时用)

        self.completed = False
        self.finish_time = None

    def get_ready_ops(self):
        return [op for op, s in self.op_status.items() if s == 'ready']

    def is_op_ready(self, op):
        return self.op_status.get(op) == 'ready'

    def mark_op_start(self, op, t, device):
        self.op_status[op] = 'processing'
        self.op_start[op] = t
        self.op_device[op] = device

    def mark_op_done(self, op, t):
        self.op_status[op] = 'done'
        self.op_end[op] = t
        # 检查后续工序是否可以开始
        for next_op, prereqs in PREREQ.items():
            if self.op_status.get(next_op) == 'pending':
                if all(self.op_status.get(p) == 'done' for p in prereqs):
                    self.op_status[next_op] = 'ready'
        if all(self.op_status[op] == 'done' for op in ['A','B','C','D','E','F','G']):
            self.completed = True
            self.finish_time = t

    def tardiness(self, current_time):
        ft = self.finish_time if self.finish_time else current_time
        return max(0, ft - self.due)

    def urgency(self, current_time, min_remaining, device_remaining=0):
        """交货紧迫度（越大越紧迫）"""
        if self.completed:
            return 0
        ready_ops = self.get_ready_ops()
        if not ready_ops:
            # 正在加工中
            processing_ops = [op for op, s in self.op_status.items() if s == 'processing']
            if processing_ops:
                op = processing_ops[0]
                remaining_key = op
                mr = min_remaining.get(self.id, {}).get(remaining_key, 0)
                return current_time + device_remaining + mr - self.due
        # 等待调度
        # 找第一个ready的工序的最短剩余时间
        mr = min(min_remaining.get(self.id, {}).get(op, 0) for op in ready_ops) if ready_ops else 0
        return current_time + mr - self.due


class DeviceState:
    def __init__(self, dev_id, op, compatible_types):
        self.id = dev_id
        self.op = op
        self.compatible_types = compatible_types
        self.status = 'idle'   # idle / busy / broken
        self.current_order = None
        self.current_op = None
        self.busy_until = 0.0
        self.last_type = None   # 上次加工的型号（换料用）
        self.total_busy_time = 0.0
        self.changeover_end = 0.0  # 换料结束时间
        # 历史记录
        self.history = []  # (order_id, op, start, end, changeover_start)

    def can_process(self, product_type):
        return product_type in self.compatible_types

    def idle_rate(self, current_time):
        if current_time <= 0:
            return 1.0
        return 1.0 - self.total_busy_time / current_time

    def specialization(self, total_types=15):
        return 1.0 - len(self.compatible_types) / total_types


class ShopFloorEnv:
    def __init__(self, data):
        self.data = data
        self.reset()

    def reset(self):
        d = self.data
        # 初始化订单
        self.orders = {o['id']: OrderState(o) for o in d['orders']}
        self.new_orders_queue = sorted(d['new_orders'], key=lambda x: x['arrive'])
        # 初始化设备
        self.devices = {}
        for dev_id, info in d['device_info'].items():
            self.devices[dev_id] = DeviceState(dev_id, info['op'], info['compatible_types'])
        self.current_time = 0.0
        self.event_log = []   # 事件日志
        self.done = False
        # 故障队列
        self.maintenance_queue = sorted(d['maintenance'], key=lambda x: x['fail'])
        self.active_failures = {}  # dev -> recover_time
        return self._get_state()

    def _check_events(self):
        """处理到达当前时间的事件（新订单到达、设备故障/恢复）"""
        events = []
        # 新订单
        arrived = [o for o in self.new_orders_queue if o['arrive'] <= self.current_time]
        for o in arrived:
            self.new_orders_queue.remove(o)
            ns = OrderState(o)
            self.orders[o['id']] = ns
            events.append(('new_order', o['id'], self.current_time))
            self.event_log.append({'type':'new_order','order':o['id'],'time':self.current_time})

        # 设备故障
        failed = [m for m in self.maintenance_queue if m['fail'] <= self.current_time]
        for m in failed:
            self.maintenance_queue.remove(m)
            dev = self.devices[m['dev']]
            # 如果正在加工，拆单
            if dev.status == 'busy' and dev.current_order:
                self._handle_breakdown_split(dev, m['fail'])
            old_status = dev.status
            dev.status = 'broken'
            self.active_failures[m['dev']] = m['recover']
            events.append(('breakdown', m['dev'], m['fail']))
            self.event_log.append({'type':'breakdown','dev':m['dev'],
                                   'fail_time':m['fail'],'recover_time':m['recover']})

        # 设备恢复
        recovered = [dev_id for dev_id, rt in list(self.active_failures.items())
                     if rt <= self.current_time]
        for dev_id in recovered:
            del self.active_failures[dev_id]
            dev = self.devices[dev_id]
            dev.status = 'idle'
            dev.current_order = None
            dev.current_op = None
            events.append(('recover', dev_id, self.current_time))
            self.event_log.append({'type':'recover','dev':dev_id,'time':self.current_time})

        return events

    def _handle_breakdown_split(self, dev, fail_time):
        """设备故障时，将正在加工的订单拆分"""
        order = self.orders.get(dev.current_order)
        if not order:
            return
        op = dev.current_op
        start = order.op_start.get(op, dev.busy_until - 1)
        # 计算已完成数量
        elapsed = fail_time - start
        unit_time = self.data['proc_time'].get((dev.id, order.product_type), 1)
        done_qty = int(elapsed / unit_time)
        remaining_qty = order.qty - done_qty

        if done_qty > 0 and remaining_qty > 0:
            # 拆分：已完成部分继续下一工序，未完成部分重新等待该工序
            # 修改原订单为已完成部分
            order.qty = done_qty
            order.mark_op_done(op, fail_time)
            # 创建新订单（未完成部分）
            new_id = order.id + '_split'
            new_order_data = {
                'id': new_id, 'type': order.product_type,
                'qty': remaining_qty, 'mode': order.mode,
                'due': order.due, 'arrive': self.current_time
            }
            new_state = OrderState(new_order_data)
            # 未完成部分需要重做该工序
            for prev_op in ['A','B','C','D','E','F','G']:
                if prev_op == op:
                    break
                new_state.op_status[prev_op] = 'done'
                new_state.op_end[prev_op] = order.op_end.get(prev_op, self.current_time)
            new_state.op_status[op] = 'ready'
            # 重置 D 之前 ready 状态（保证依赖关系）
            if op in ['D','E','F','G']:
                pass  # 前置工序已done
            self.orders[new_id] = new_state
            self.event_log.append({'type':'split','original':order.id,
                                   'new':new_id,'time':fail_time})
        elif done_qty == 0:
            # 没有完成任何，直接重置工序状态
            order.op_status[op] = 'ready'
            if op in order.op_start:
                del order.op_start[op]
        else:
            # 全部完成
            order.mark_op_done(op, fail_time)

        dev.current_order = None
        dev.current_op = None
        dev.busy_until = fail_time
        self.event_log.append({'type':'breakdown_split','dev':dev.id,'time':fail_time})

    def get_schedulable_pairs(self):
        """获取当前可调度的(订单, 设备)组合"""
        pairs = []
        # 可用设备：空闲且不在故障中
        available_devices = [dev for dev in self.devices.values()
                             if dev.status == 'idle']
        # 可调度订单：有ready工序且已到达
        for order in self.orders.values():
            if order.completed:
                continue
            if order.arrive > self.current_time:
                continue
            ready_ops = order.get_ready_ops()
            for op in ready_ops:
                for dev in available_devices:
                    if dev.op == op and dev.can_process(order.product_type):
                        pairs.append((order.id, dev.id, op))
        return pairs

    def assign(self, order_id, device_id, op):
        """执行调度决策"""
        order = self.orders[order_id]
        dev = self.devices[device_id]
        assert dev.status == 'idle'
        assert dev.can_process(order.product_type)

        # 换料时间
        changeover_fn = self.data['changeover'].get(op)
        changeover_secs = 0
        changeover_start = None
        if changeover_fn and dev.last_type is not None:
            co_mins = changeover_fn(dev.last_type, order.product_type)
            if co_mins > 0:
                # 如果设备空闲时间已超过换料时间，则换料时间忽略
                idle_since = dev.busy_until
                idle_time = self.current_time - idle_since
                if idle_time < co_mins * 60:
                    changeover_secs = co_mins * 60 - idle_time
                    changeover_start = self.current_time

        proc_sec = self.data['proc_time'].get((device_id, order.product_type), 0)
        total_proc = proc_sec * order.qty
        start_time = self.current_time + changeover_secs
        end_time = start_time + total_proc

        dev.status = 'busy'
        dev.current_order = order_id
        dev.current_op = op
        dev.busy_until = end_time
        dev.last_type = order.product_type
        dev.total_busy_time += changeover_secs + total_proc
        dev.history.append({
            'order_id': order_id,
            'op': op,
            'changeover_start': changeover_start,
            'changeover_end': self.current_time + changeover_secs if changeover_start else None,
            'start': start_time,
            'end': end_time,
            'product_type': order.product_type
        })

        order.mark_op_start(op, start_time, device_id)

        self.event_log.append({
            'type': 'assign', 'order': order_id, 'dev': device_id,
            'op': op, 'start': start_time, 'end': end_time,
            'changeover': changeover_secs
        })
        return end_time

    def advance_to_next_event(self):
        """推进时间到下一个事件（设备完成/故障/恢复/新订单到达）"""
        candidate_times = []
        # 设备完成时间
        for dev in self.devices.values():
            if dev.status == 'busy':
                candidate_times.append(dev.busy_until)
        # 故障时间
        for m in self.maintenance_queue:
            candidate_times.append(m['fail'])
        # 恢复时间
        for rt in self.active_failures.values():
            candidate_times.append(rt)
        # 新订单到达时间
        for o in self.new_orders_queue:
            candidate_times.append(o['arrive'])

        if not candidate_times:
            return False

        next_time = min(t for t in candidate_times if t >= self.current_time)
        self.current_time = next_time

        # 处理设备完成
        for dev in self.devices.values():
            if dev.status == 'busy' and abs(dev.busy_until - self.current_time) < 0.01:
                order = self.orders.get(dev.current_order)
                if order:
                    order.mark_op_done(dev.current_op, self.current_time)
                dev.status = 'idle'
                dev.current_order = None
                dev.current_op = None

        self._check_events()
        return True

    def is_terminal(self):
        all_done = all(o.completed for o in self.orders.values())
        no_pending = len(self.new_orders_queue) == 0
        no_busy = all(d.status != 'busy' for d in self.devices.values())
        # 如果没有新订单且所有订单完成
        if all_done and no_pending:
            return True
        # 防止死锁：无事件可推进
        if no_busy and not no_pending:
            return False
        return False

    def makespan(self):
        finish_times = [o.finish_time for o in self.orders.values()
                        if o.finish_time is not None]
        return max(finish_times) if finish_times else self.current_time

    def total_tardiness(self, mode=None):
        result = 0
        for o in self.orders.values():
            if mode and o.mode != mode:
                continue
            result += o.tardiness(self.current_time)
        return result

    def reward_value(self):
        mto = self.total_tardiness('MTO')
        mts = self.total_tardiness('MTS')
        ms = self.makespan()
        return 0.7 * mto + 0.3 * mts + ms

    def _get_state(self):
        return {
            'time': self.current_time,
            'orders': self.orders,
            'devices': self.devices,
        }
