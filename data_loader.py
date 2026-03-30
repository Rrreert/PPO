"""
数据加载模块 - 从Excel读取车间调度所需的所有数据
"""
import pandas as pd
import numpy as np
from openpyxl import load_workbook


def load_all_data(filepath):
    """加载所有Excel数据，返回结构化字典"""
    wb = load_workbook(filepath, read_only=True)

    # ===== 1. 设备通用性表 =====
    # compatibility[设备名][型号] = 1/None
    ws = wb['设备通用性']
    rows = [row for row in ws.iter_rows(values_only=True) if any(v is not None for v in row)]
    product_types = [str(v) for v in rows[1][2:]]  # ['01','02',...,'15']

    compatibility = {}   # {machine_id: set of product_types}
    machine_to_stage = {}  # {machine_id: stage}
    stage_to_machines = {}  # {stage: [machine_ids]}
    for row in rows[2:]:
        stage = row[0]
        machine = row[1]
        compatible_products = set()
        for idx, v in enumerate(row[2:]):
            if v == 1:
                compatible_products.add(product_types[idx])
        compatibility[machine] = compatible_products
        machine_to_stage[machine] = stage
        stage_to_machines.setdefault(stage, []).append(machine)

    # ===== 2. 生产加工时间表 =====
    # proc_time[设备名][型号] = 加工时间(秒)
    ws = wb['生产加工时间']
    rows = [row for row in ws.iter_rows(values_only=True) if any(v is not None for v in row)]
    proc_time = {}
    for row in rows[2:]:
        machine = row[1]
        proc_time[machine] = {}
        for idx, v in enumerate(row[2:]):
            if v is not None:
                proc_time[machine][product_types[idx]] = v

    # ===== 3. 换料时间表 =====
    # setup_time[stage] = {'time': minutes, 'groups': groups_list or None}
    ws = wb['换料时间']
    rows = [row for row in ws.iter_rows(values_only=True) if any(v is not None for v in row)]
    setup_rules = {}
    for row in rows[1:]:
        stage, rule_desc, minutes = row[0], row[1], row[2]
        if '所有型号之间切换' in str(rule_desc):
            # 所有型号互相切换都需要换料
            setup_rules[stage] = {'time_min': minutes, 'groups': None}
        elif '01-05' in str(rule_desc):
            # E工序：组内不换料，组间换料
            groups = [
                {'01', '02', '03', '04', '05'},
                {'06', '07', '08'},
                {'09', '10', '11', '12', '13', '14', '15'}
            ]
            setup_rules[stage] = {'time_min': minutes, 'groups': groups}

    # ===== 4. 订单表 =====
    ws = wb['订单']
    rows = [row for row in ws.iter_rows(values_only=True) if any(v is not None for v in row)]
    orders = []
    for row in rows[1:]:
        order_id, prod_type, qty, mode, deadline_min = row
        orders.append({
            'order_id': order_id,
            'product_type': str(prod_type).zfill(2),
            'quantity': int(qty),
            'mode': mode,           # 'MTO' or 'MTS'
            'deadline_sec': int(deadline_min) * 60,  # 转换为秒
        })

    # ===== 5. 最短剩余加工时间表 =====
    ws = wb['最短剩余加工时间']
    rows = [row for row in ws.iter_rows(values_only=True) if any(v is not None for v in row)]
    min_remaining = {}
    stages = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    for row in rows[1:]:
        order_id = row[0]
        min_remaining[order_id] = {stages[i]: int(row[i + 1]) for i in range(7)}

    wb.close()

    return {
        'product_types': product_types,           # ['01'..'15']
        'compatibility': compatibility,            # {machine: set(types)}
        'machine_to_stage': machine_to_stage,      # {machine: stage}
        'stage_to_machines': stage_to_machines,    # {stage: [machines]}
        'proc_time': proc_time,                    # {machine: {type: sec}}
        'setup_rules': setup_rules,                # {stage: {...}}
        'orders': orders,                          # list of order dicts
        'min_remaining': min_remaining,            # {order_id: {stage: sec}}
        'stages': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    }


def needs_setup(stage, prev_type, curr_type, setup_rules):
    """判断是否需要换料，返回换料时间（分钟），不需要则返回0"""
    if stage not in setup_rules:
        return 0
    if prev_type is None or prev_type == curr_type:
        return 0
    rule = setup_rules[stage]
    groups = rule['groups']
    if groups is None:
        # 所有型号之间都需要换料
        return rule['time_min']
    # 检查是否在同一组
    for group in groups:
        if prev_type in group and curr_type in group:
            return 0
    return rule['time_min']


if __name__ == '__main__':
    data = load_all_data('/mnt/user-data/uploads/data.xlsx')
    print("产品型号数:", len(data['product_types']))
    print("订单数:", len(data['orders']))
    print("工序->设备:", {k: v for k, v in data['stage_to_machines'].items()})
    print("换料规则:", {k: v['time_min'] for k, v in data['setup_rules'].items()})
    print("数据加载成功 ✓")
