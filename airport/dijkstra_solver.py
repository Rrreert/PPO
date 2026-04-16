"""
dijkstra_solver.py
用 Dijkstra 最短路径（按距离）为每架航班规划滑行路径，
计算 4 项指标：滑行时间、滑行距离、CO2 排放、冲突次数。
"""
import numpy as np
import networkx as nx
from collections import defaultdict
from flight_data import (CARBON_FACTOR, ACCEL_FACTOR, MAX_SPEED,
                          TURN_SPEED, ACCEL, DECEL, DT, D_MIN, D_SAFE, FUEL_FLOW)
from airport_graph import FORBIDDEN_NODES, get_restricted_graph


def _path_distance(path: list, pos: dict) -> float:
    """计算路径总欧氏距离（米）"""
    total = 0.0
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        total += np.linalg.norm(np.array(pos[a]) - np.array(pos[b]))
    return total


def _angle_at_node(path: list, idx: int, pos: dict) -> float:
    """计算 path[idx] 处的转弯角（度）"""
    if idx <= 0 or idx >= len(path) - 1:
        return 0.0
    a, b, c = path[idx - 1], path[idx], path[idx + 1]
    v1 = np.array(pos[b]) - np.array(pos[a])
    v2 = np.array(pos[c]) - np.array(pos[b])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_a))
    return angle  # 0=直行, 180=U形


def compute_flight_metrics(path: list, fuel_rate: float, pos: dict) -> dict:
    """
    给定路径，计算：
      - total_time (s)
      - total_distance (m)
      - co2 (kg)
    航空器从起点出发，匀速 MAX_SPEED，遇转弯减到 TURN_SPEED 再加速。
    转弯处额外 CO2 按 PDF 公式计算。
    """
    if len(path) < 2:
        return {'time': 0.0, 'distance': 0.0, 'co2': 0.0}

    total_dist = _path_distance(path, pos)
    total_co2  = 0.0
    total_time = 0.0

    THRESHOLD_ANGLE = 20.0  # 超过此角度认为是"转弯"

    # 按节点段逐段模拟
    speed = 0.0  # 从停机位静止出发（或从跑道入口以低速进入）
    for seg_idx in range(len(path) - 1):
        a, b = path[seg_idx], path[seg_idx + 1]
        seg_len = np.linalg.norm(np.array(pos[b]) - np.array(pos[a]))

        # 判断下一段是否转弯
        turn_at_b = False
        angle = _angle_at_node(path, seg_idx + 1, pos)
        if angle > THRESHOLD_ANGLE:
            turn_at_b = True

        # 目标速度：转弯节点前减速到 TURN_SPEED，否则 MAX_SPEED
        v_target = TURN_SPEED if turn_at_b else MAX_SPEED

        # 简化动力学：用平均速度估算时间
        # Phase 1: 当前速度 → v_target 加速/减速
        if abs(speed - v_target) > 0.1:
            a_eff = ACCEL if v_target > speed else DECEL
            dt_accel = abs(v_target - speed) / a_eff
            dist_accel = abs(v_target ** 2 - speed ** 2) / (2 * a_eff)
        else:
            dt_accel, dist_accel = 0.0, 0.0

        dist_accel = min(dist_accel, seg_len)
        dist_cruise = max(0.0, seg_len - dist_accel)

        # 时间
        t_accel = dt_accel if dist_accel > 0 else 0.0
        t_cruise = dist_cruise / v_target if v_target > 0 else 0.0
        seg_time = t_accel + t_cruise

        # CO2 —— 加速段用 ACCEL_FACTOR * m_idle
        co2_accel = fuel_rate * ACCEL_FACTOR * CARBON_FACTOR * t_accel
        co2_cruise = fuel_rate * CARBON_FACTOR * t_cruise

        # 转弯补偿（PDF 公式：Taccel = π*R/180 + (Vcruise-Vturn)/a）
        if turn_at_b and seg_idx + 1 < len(path) - 1:
            R = 30.0  # 假设转弯半径 30 m
            T_accel_turn = (np.pi * R * angle / 180.0 / TURN_SPEED +
                            (MAX_SPEED - TURN_SPEED) / ACCEL)
            co2_turn = fuel_rate * ACCEL_FACTOR * CARBON_FACTOR * T_accel_turn
            total_co2  += co2_turn
            total_time += T_accel_turn

        total_co2  += co2_accel + co2_cruise
        total_time += seg_time
        speed = v_target

    return {
        'time': total_time,
        'distance': total_dist,
        'co2': total_co2,
    }


def check_conflicts(flight_trajectories: list, pos: dict) -> int:
    """
    粗粒度冲突检测：将每架飞机的轨迹离散化为时间戳→位置，
    检查同一时刻两架飞机距离 < D_MIN 的次数（每对飞机只算一次）。
    flight_trajectories: list of {'id', 'path', 'start_time', 'metrics'}
    """
    # 构建 {time: [(aircraft_id, node_idx, x, y)]}
    time_pos = defaultdict(list)

    for ft in flight_trajectories:
        path = ft['path']
        if len(path) < 1:
            continue
        t0 = ft['start_time']
        metrics = ft['metrics']
        total_t = metrics['time']
        if total_t <= 0 or len(path) < 2:
            continue

        # 均匀插值：每秒记录一个位置
        # 简化：把总时间平均分配到每段
        coords = [np.array(pos[n]) for n in path if n in pos]
        if len(coords) < 2:
            continue

        # 计算累计距离
        dists = [0.0]
        for i in range(1, len(coords)):
            dists.append(dists[-1] + np.linalg.norm(coords[i] - coords[i-1]))
        total_d = dists[-1]
        if total_d <= 0:
            continue

        n_steps = max(int(total_t), 1)
        for step in range(n_steps + 1):
            t = t0 + step
            frac = min(step / n_steps, 1.0)
            target_d = frac * total_d

            # 找当前所在段
            seg = 0
            for k in range(len(dists) - 1):
                if dists[k] <= target_d <= dists[k + 1]:
                    seg = k
                    break
            seg_len = dists[seg + 1] - dists[seg]
            if seg_len > 0:
                alpha = (target_d - dists[seg]) / seg_len
                xy = coords[seg] * (1 - alpha) + coords[seg + 1] * alpha
            else:
                xy = coords[seg]

            time_pos[t].append((ft['id'], xy))

    # 检测冲突
    conflict_pairs = set()
    for t, aircraft_list in time_pos.items():
        for i in range(len(aircraft_list)):
            for j in range(i + 1, len(aircraft_list)):
                id_i, xy_i = aircraft_list[i]
                id_j, xy_j = aircraft_list[j]
                d = np.linalg.norm(xy_i - xy_j)
                if d < D_MIN:
                    pair = tuple(sorted([id_i, id_j]))
                    conflict_pairs.add(pair)

    return len(conflict_pairs)


def run_dijkstra(G: nx.Graph, flights: list, pos: dict) -> dict:
    """
    为所有航班运行 Dijkstra，返回结果字典。
    """
    Gr = get_restricted_graph(G)

    results = []
    no_path_count = 0

    for fl in flights:
        src, dst = fl['start_node'], fl['end_node']

        # 处理起点/终点不在受限图中的情况
        src_use = src if src in Gr else (
            min(Gr.nodes(), key=lambda n: np.linalg.norm(
                np.array(pos.get(n, (0,0))) - np.array(pos.get(src, (0,0))))))
        dst_use = dst if dst in Gr else (
            min(Gr.nodes(), key=lambda n: np.linalg.norm(
                np.array(pos.get(n, (0,0))) - np.array(pos.get(dst, (0,0))))))

        try:
            path = nx.dijkstra_path(Gr, src_use, dst_use, weight='weight')
        except nx.NetworkXNoPath:
            # 降级：用完整图
            try:
                path = nx.dijkstra_path(G, src, dst, weight='weight')
            except:
                path = [src, dst]
            no_path_count += 1

        metrics = compute_flight_metrics(path, fl['fuel_rate'], pos)

        results.append({
            'id': fl['id'],
            'flight_no': fl['flight_no'],
            'type': fl['type'],
            'aircraft_type': fl['aircraft_type'],
            'path': path,
            'start_time': fl['actual_time'],
            'metrics': metrics,
        })

    if no_path_count:
        print(f"[Dijkstra] {no_path_count} flights fell back to unrestricted graph")

    # 全局冲突检测
    n_conflicts = check_conflicts(results, pos)

    # 汇总
    total_time     = sum(r['metrics']['time'] for r in results)
    total_distance = sum(r['metrics']['distance'] for r in results)
    total_co2      = sum(r['metrics']['co2'] for r in results)

    summary = {
        'algorithm': 'Dijkstra',
        'total_time_s': total_time,
        'total_distance_m': total_distance,
        'total_co2_kg': total_co2,
        'conflicts': n_conflicts,
        'avg_time_s': total_time / len(results),
        'avg_distance_m': total_distance / len(results),
        'avg_co2_kg': total_co2 / len(results),
        'results': results,
    }

    print(f"[Dijkstra] Done. flights={len(results)}, "
          f"total_time={total_time/3600:.2f}h, "
          f"total_dist={total_distance/1000:.2f}km, "
          f"total_co2={total_co2:.1f}kg, "
          f"conflicts={n_conflicts}")
    return summary


if __name__ == "__main__":
    import os, sys
    os.chdir("/home/claude")
    sys.path.insert(0, "/home/claude")
    from airport_graph import build_graph
    from flight_data import load_flights
    G, pos = build_graph("虹桥点.csv")
    flights = load_flights("出港航班0407.csv", "进港航班0407.csv", set(G.nodes()), pos)
    summary = run_dijkstra(G, flights, pos)
    print("avg_time:", summary['avg_time_s'], "s")
    print("avg_dist:", summary['avg_distance_m'], "m")
    print("avg_co2:", summary['avg_co2_kg'], "kg")
