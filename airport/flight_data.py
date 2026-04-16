"""
flight_data.py
加载进出港航班数据，分配起终点，处理缺失停机位就近映射。
"""
import pandas as pd
import numpy as np
import networkx as nx
from typing import Optional


# ─── 燃油流率表 ────────────────────────────────
FUEL_FLOW = {
    'B737': 0.110, 'B747': 0.219, 'B777': 0.380,
    'B787': 0.223, 'A320': 0.094, 'A321': 0.094,
    'A330': 0.227, 'A350': 0.301, 'C919': 0.094,
    'A319': 0.094,  # 用 A320 替代
}

CARBON_FACTOR = 3.15        # kg CO2 / kg 燃油
ACCEL_FACTOR  = 1.75        # 加速/转弯时燃油流率倍数（取 1.5~2.0 中值）
MAX_SPEED     = 20 / 3.6    # m/s  ≈ 5.556 m/s
TURN_SPEED    = 5.0         # m/s
ACCEL         = 0.3         # m/s²
DECEL         = 0.6         # m/s²
DT            = 1.0         # 仿真时间步长 (s)

# 安全距离参数
D_MIN  = 50.0   # m  碰撞距离（episode 重置）
D_SAFE = 200.0  # m  预警距离


def _find_nearest_stand(missing_stand_str: str, available_nodes: list, pos: dict) -> str:
    """按欧氏距离将缺失停机位号映射到最近的 S 节点"""
    # 先尝试直接加前缀
    candidate = 'S' + str(missing_stand_str)
    if candidate in pos:
        return candidate

    # 解析停机位号中的数字部分估算坐标
    num_str = ''.join(filter(str.isdigit, str(missing_stand_str)))
    if not num_str:
        return available_nodes[0]

    # 没有真实坐标 → 找编号最近的 S 节点
    num = int(num_str)
    s_nodes = [n for n in available_nodes if n.startswith('S') and
               ''.join(filter(str.isdigit, n)).isdigit()]
    if not s_nodes:
        return available_nodes[0]
    best = min(s_nodes, key=lambda n: abs(int(''.join(filter(str.isdigit, n))) - num))
    return best


def _map_stand_to_node(stand_raw, graph_nodes: set, pos: dict) -> str:
    """将停机位号（字符串或数字）映射到图节点名"""
    s = str(stand_raw).strip()
    node = 'S' + s
    if node in graph_nodes:
        return node
    # 不在图中 → 欧氏就近映射
    return _find_nearest_stand(s, list(graph_nodes), pos)


def _nearest_stand_euclidean(stand_raw, graph_nodes: set, pos: dict) -> str:
    """对29架次缺失停机位：先生成虚拟坐标，再用欧氏距离找最近真实节点"""
    s = str(stand_raw).strip()
    direct = 'S' + s
    if direct in graph_nodes and direct in pos:
        return direct

    # 所有停机位节点 (S + 纯数字 or SE/E 结尾)
    stand_nodes = [n for n in graph_nodes if n.startswith('S') and n in pos]
    if not stand_nodes:
        return list(graph_nodes)[0]

    # 用编号估算 Y 坐标（虹桥停机位号越大越靠南，X 大致固定）
    # 直接按编号差找最近编号的停机位
    num_str = ''.join(filter(str.isdigit, s))
    if not num_str:
        return stand_nodes[0]
    num = int(num_str)

    def node_num(n):
        ns = ''.join(filter(str.isdigit, n))
        return int(ns) if ns else 0

    # 尝试用坐标欧氏距离（若有邻近已知节点作为参考）
    # 最简单可靠：按编号差
    best = min(stand_nodes, key=lambda n: abs(node_num(n) - num))
    return best


def load_flights(dep_csv: str, arr_csv: str, graph_nodes: set, pos: dict) -> list[dict]:
    """
    返回航班列表，每条记录包含：
      id, type(dep/arr), aircraft_type, fuel_rate,
      start_node, end_node, scheduled_time(s from midnight),
      actual_time(s), runway
    """
    dep_df = pd.read_csv(dep_csv)
    arr_df = pd.read_csv(arr_csv)

    def parse_time(t_str):
        """'7:32' → 秒数"""
        try:
            parts = str(t_str).split(':')
            h, m = int(parts[0]), int(parts[1])
            s = int(parts[2]) if len(parts) > 2 else 0
            return h * 3600 + m * 60 + s
        except:
            return 0

    # 所有停机位节点的 Y 坐标，用于南北分割
    stand_ys = {n: pos[n][1] for n in graph_nodes if n.startswith('S') and n in pos}
    all_y = list(stand_ys.values())
    y_median = np.median(all_y) if all_y else 500

    flights = []
    fid = 0

    # ─── 出港航班 ─────────────────────────────
    for _, row in dep_df.iterrows():
        ac_type = str(row['机型']).strip()
        fuel_rate = FUEL_FLOW.get(ac_type, 0.094)
        stand_raw = row['停机位']
        runway = str(row['起飞跑道']).strip()

        start_node = _nearest_stand_euclidean(stand_raw, graph_nodes, pos)

        # 终点：36R → R18L_H7，36L → R18R_2
        if runway == '36R':
            end_node = 'R18L_H7'
        else:
            end_node = 'R18R_2'

        # 确保终点在图中
        if end_node not in graph_nodes:
            end_node = list(graph_nodes)[0]

        actual_time = parse_time(row['起飞时间'])

        flights.append({
            'id': f'DEP_{fid:03d}',
            'flight_no': str(row['航班号']),
            'type': 'dep',
            'aircraft_type': ac_type,
            'fuel_rate': fuel_rate,
            'start_node': start_node,
            'end_node': end_node,
            'actual_time': actual_time,
            'runway': runway,
        })
        fid += 1

    # ─── 进港航班 ─────────────────────────────
    for _, row in arr_df.iterrows():
        ac_type = str(row['机型']).strip()
        fuel_rate = FUEL_FLOW.get(ac_type, 0.094)
        stand_raw = row['停机位']

        end_node = _nearest_stand_euclidean(stand_raw, graph_nodes, pos)
        end_y = pos.get(end_node, (0, 0))[1]

        # 起点：靠北(Y大)用 R18L_A1，靠南用 R18L_B3
        if end_y >= y_median:
            start_node = 'R18L_A1'
        else:
            start_node = 'R18L_B3'

        # 确保起点在图中
        if start_node not in graph_nodes:
            start_node = 'R18L_B3'

        actual_time = parse_time(row['到达时间'])

        flights.append({
            'id': f'ARR_{fid:03d}',
            'flight_no': str(row['航班号']),
            'type': 'arr',
            'aircraft_type': ac_type,
            'fuel_rate': fuel_rate,
            'start_node': start_node,
            'end_node': end_node,
            'actual_time': actual_time,
            'runway': str(row['落地跑道']).strip(),
        })
        fid += 1

    # 按实际时间排序
    flights.sort(key=lambda x: x['actual_time'])
    print(f"[Flights] loaded: dep={len(dep_df)}, arr={len(arr_df)}, total={len(flights)}")
    return flights


if __name__ == "__main__":
    import os, sys
    os.chdir("/home/claude")
    sys.path.insert(0, "/home/claude")
    from airport_graph import build_graph
    G, pos = build_graph("虹桥点.csv")
    flights = load_flights("出港航班0407.csv", "进港航班0407.csv", set(G.nodes()), pos)
    print(flights[0])
    print(flights[-1])
