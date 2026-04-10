"""数据加载模块"""
import pandas as pd
import numpy as np

DATA_PATH = "/kaggle/working/PPO/v2/data_dy.xlsx"

def load_all_data():
    xl = pd.read_excel(DATA_PATH, sheet_name=None, header=None)

    # ---- 设备通用性 ----
    raw = xl['设备通用性']
    # row0: 标题行(产品型号合并单元格), row1: 工序/设备/1..15, row2+: 数据
    ops = raw.iloc[2:, 0].values      # 工序列
    devs = raw.iloc[2:, 1].values     # 设备列
    types = list(range(1, 16))        # 型号 1-15
    compat_matrix = raw.iloc[2:, 2:].values.astype(float)  # NaN = 不可加工

    device_info = {}   # dev_id -> {op, compatible_types}
    op_devices = {}    # op -> [dev_id]
    for i, (op, dev) in enumerate(zip(ops, devs)):
        compatible = [types[j] for j in range(len(types))
                      if not np.isnan(compat_matrix[i, j]) and compat_matrix[i, j] == 1.0]
        device_info[dev] = {'op': op, 'compatible_types': compatible}
        op_devices.setdefault(op, []).append(dev)

    # ---- 生产加工时间 ----
    raw2 = xl['生产加工时间']
    proc_matrix = raw2.iloc[2:, 2:].values.astype(float)
    proc_time = {}  # (dev, type) -> seconds/unit
    for i, dev in enumerate(devs):
        for j, t in enumerate(types):
            v = proc_matrix[i, j]
            if not np.isnan(v):
                proc_time[(dev, int(t))] = v

    # ---- 换料时间 ----
    raw3 = xl['换料时间']
    raw3.columns = ['工序', '切换型号', '换料时间']
    raw3 = raw3.iloc[1:].reset_index(drop=True)
    # B和D: 所有型号之间切换; E: 跨组切换
    changeover = {}  # op -> function(from_type, to_type) -> minutes
    E_groups = [set(range(1,6)), set(range(6,9)), set(range(9,16))]
    def make_changeover(op, minutes):
        if op == 'E':
            def fn(a, b):
                ga = next((i for i,g in enumerate(E_groups) if a in g), -1)
                gb = next((i for i,g in enumerate(E_groups) if b in g), -1)
                return minutes if ga != gb else 0
            return fn
        else:
            return lambda a, b: minutes if a != b else 0
    for _, row in raw3.iterrows():
        op = row['工序']
        mins = float(row['换料时间'])
        changeover[op] = make_changeover(op, mins)

    # ---- 订单 ----
    raw4 = xl['订单']
    raw4.columns = raw4.iloc[0]; raw4 = raw4.iloc[1:].reset_index(drop=True)
    orders = []
    for _, r in raw4.iterrows():
        orders.append({
            'id': r['订单编号'], 'type': int(r['产品型号']),
            'qty': int(r['产品数量']), 'mode': r['生产方式'],
            'due': float(r['交货时间/分钟']) * 60,  # 转秒
            'arrive': 0.0
        })

    # ---- 新订单 ----
    raw5 = xl['新订单']
    raw5.columns = raw5.iloc[0]; raw5 = raw5.iloc[1:].reset_index(drop=True)
    new_orders = []
    for _, r in raw5.iterrows():
        new_orders.append({
            'id': r['订单编号'], 'type': int(r['产品型号']),
            'qty': int(r['产品数量']), 'mode': r['生产方式'],
            'due': float(r['交货时间/分钟']) * 60,
            'arrive': float(r['到达时间/分钟']) * 60
        })

    # ---- 最短剩余加工时间 ----
    raw6 = xl['最短剩余加工时间']
    raw6.columns = raw6.iloc[0]; raw6 = raw6.iloc[1:].reset_index(drop=True)
    min_remaining = {}
    op_cols = ['工序A_最短剩余加工时间/秒','工序B_最短剩余加工时间/秒',
               '工序C_最短剩余加工时间/秒','工序D_最短剩余加工时间/秒',
               '工序E_最短剩余加工时间/秒','工序F_最短剩余加工时间/秒',
               '工序G_最短剩余加工时间/秒']
    op_keys = ['A','B','C','D','E','F','G']
    for _, r in raw6.iterrows():
        oid = r['订单编号']
        min_remaining[oid] = {op_keys[i]: float(r[op_cols[i]]) for i in range(7)}

    # ---- 设备维修 ----
    raw7 = xl['设备维修']
    raw7.columns = raw7.iloc[0]; raw7 = raw7.iloc[1:].reset_index(drop=True)
    maintenance = []
    for _, r in raw7.iterrows():
        maintenance.append({
            'dev': r['故障设备'],
            'fail': float(r['故障时间/分钟']) * 60,
            'recover': float(r['恢复时间/分钟']) * 60
        })

    # 工序顺序（A和B/C并行，D-G串行）
    # 每个订单的工序路径: A -> BC(并行) -> D -> E -> F -> G
    # 并行: A 和 [B->C] 同时进行，均完成后才能进入D

    all_devices = list(device_info.keys())

    return {
        'device_info': device_info,
        'op_devices': op_devices,
        'proc_time': proc_time,
        'changeover': changeover,
        'orders': orders,
        'new_orders': new_orders,
        'min_remaining': min_remaining,
        'maintenance': maintenance,
        'all_devices': all_devices,
        'product_types': [int(t) for t in types],
        'ops_order': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    }
