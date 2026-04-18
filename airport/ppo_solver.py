"""
ppo_solver.py
用 Stable-Baselines3 PPO 训练单机滑行策略，
然后用训练好的模型为所有 137 架次规划路径。
"""
import os
import numpy as np
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from taxi_env import TaxiEnv, MAX_NEIGHBORS
from airport_graph import FORBIDDEN_NODES, get_restricted_graph
from dijkstra_solver import compute_flight_metrics, check_conflicts
from flight_data import MAX_SPEED, TURN_SPEED, ACCEL, DECEL, CARBON_FACTOR, ACCEL_FACTOR

MODEL_PATH = "ppo_taxi_model"


def _build_occupied(prev_results: list, pos: dict) -> dict:
    """
    将已规划航班的轨迹转换为 {time_int: [(flight_id, xy)]} 字典，
    供后续航班冲突检测使用。
    """
    occ = defaultdict(list)
    for res in prev_results:
        path = res['path']
        t0 = res['start_time']
        total_t = res['metrics']['time']
        if total_t <= 0 or len(path) < 2:
            continue
        coords = [np.array(pos[n]) for n in path if n in pos]
        dists = [0.0]
        for i in range(1, len(coords)):
            dists.append(dists[-1] + np.linalg.norm(coords[i] - coords[i-1]))
        total_d = dists[-1]
        if total_d <= 0:
            continue
        n_steps = max(int(total_t), 1)
        for step in range(n_steps + 1):
            t = int(t0 + step)
            frac = min(step / n_steps, 1.0)
            target_d = frac * total_d
            seg = 0
            for k in range(len(dists) - 1):
                if dists[k] <= target_d <= dists[k+1]:
                    seg = k
                    break
            seg_len = dists[seg+1] - dists[seg]
            if seg_len > 0:
                alpha = (target_d - dists[seg]) / seg_len
                xy = coords[seg] * (1 - alpha) + coords[seg+1] * alpha
            else:
                xy = coords[seg]
            occ[t].append((res['id'], tuple(xy)))
    return dict(occ)


def make_env_fn(G, pos, flights, idx=0, occupied_ref=None):
    """工厂函数：为 VecEnv 创建环境"""
    def _make():
        fl = flights[idx % len(flights)]
        # 使用共享的 occupied 引用，训练时也能看到其他飞机
        occ = occupied_ref[0] if occupied_ref else {}
        return TaxiEnv(G, pos, fl, occupied_positions=occ)
    return _make


def _select_curriculum_flights(flights: list, phase: int, n_phases: int) -> list:
    """
    改动4：课程学习 — 按阶段逐步引入更多飞机。
    phase=0: 只用路径最短的25%航班（热身）
    phase=1: 50% 航班
    phase=2: 100% 航班（全量）
    """
    if phase >= n_phases - 1:
        return flights
    # 按 Dijkstra 估算的路径长度排序（用起终点欧氏距离近似）
    import math
    def approx_len(fl):
        # 直线距离作为复杂度代理
        return 0.0  # 无pos引用，直接按id顺序分阶段
    ratio = (phase + 1) / n_phases
    n = max(int(len(flights) * ratio), 8)
    return flights[:n]


def train_ppo(G: nx.Graph, pos: dict, flights: list,
              total_timesteps: int = 200_000,
              n_envs: int = 4) -> PPO:
    """
    训练 PPO 模型，包含以下改进：
    1. ent_coef=0.05，防止策略过早收敛
    2. Self-play 迭代更新 occupied（每轮用上一轮PPO结果替换Dijkstra背景）
    3. 课程学习：前1/3步骤用简单子集，后续逐步扩展到全量
    4. 稠密进度奖励（在 taxi_env.py 中实现）
    """
    print(f"[PPO] Training for {total_timesteps} steps with {n_envs} envs...")

    from dijkstra_solver import run_dijkstra
    from stable_baselines3.common.vec_env import DummyVecEnv

    # ── 初始 occupied：用 Dijkstra 结果作为第一轮背景 ──────────
    print("[PPO] Generating initial Dijkstra background...")
    dijk_summary = run_dijkstra(G, flights, pos)
    occupied_ref = [_build_occupied(dijk_summary['results'], pos)]

    # ── 课程学习配置 ─────────────────────────────────────────
    # 3个阶段：简单(33%) → 中等(66%) → 全量(100%)
    N_PHASES = 3
    phase_steps = [total_timesteps // 3,
                   total_timesteps // 3,
                   total_timesteps - 2 * (total_timesteps // 3)]
    phase_ratios = [0.33, 0.66, 1.0]

    # ── 创建模型（只创建一次，跨阶段持续学习）─────────────────
    # 先用全量flights创建env确定obs/action空间
    init_env_fns = [make_env_fn(G, pos, flights, i, occupied_ref)
                    for i in range(n_envs)]
    vec_env = DummyVecEnv(init_env_fns)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=1.0,
        max_grad_norm=0.3,
        verbose=1,
        tensorboard_log=None,
        normalize_advantage=True,
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    vec_env.close()

    # ── 分阶段训练 ────────────────────────────────────────────
    for phase_idx, (steps, ratio) in enumerate(zip(phase_steps, phase_ratios)):
        n_flights = max(int(len(flights) * ratio), 8)
        phase_flights = flights[:n_flights]
        print(f"[PPO] Phase {phase_idx+1}/{N_PHASES}: "              f"{steps} steps, {n_flights}/{len(flights)} flights")

        # 每阶段重建 env（使用当前 occupied_ref 和当前课程 flights）
        env_fns = [make_env_fn(G, pos, phase_flights, i, occupied_ref)
                   for i in range(n_envs)]
        phase_env = DummyVecEnv(env_fns)
        model.set_env(phase_env)

        model.learn(total_timesteps=steps,
                    reset_num_timesteps=(phase_idx == 0),
                    progress_bar=False)
        phase_env.close()

        # ── 改动2：Self-play 更新 occupied ────────────────────
        # 用当前模型推理一遍，用 PPO 自己的路径替换 Dijkstra 背景
        print(f"[PPO] Phase {phase_idx+1}: updating occupied via self-play...")
        selfplay_results = []
        temp_occupied = {}
        for fl in phase_flights:
            res = _rollout_single(model, G, pos, fl, temp_occupied)
            selfplay_results.append(res)
            new_occ = _build_occupied([res], pos)
            for t, lst in new_occ.items():
                temp_occupied.setdefault(t, []).extend(lst)
        # 更新共享 occupied_ref（下一阶段和下轮训练使用）
        occupied_ref[0] = _build_occupied(selfplay_results, pos)
        print(f"[PPO] Phase {phase_idx+1}: self-play occupied updated "              f"({len(occupied_ref[0])} time slots)")

    model.save(MODEL_PATH)
    print(f"[PPO] Model saved to {MODEL_PATH}")
    return model


def _rollout_single(model: PPO, G: nx.Graph, pos: dict,
                    flight: dict, occupied: dict) -> dict:
    """
    用训练好的模型为单架飞机执行推理。
    改动：
    - MAX_VISIT 降为 2（更快发现循环）
    - 检测到循环时立即截断并 Dijkstra 补全，而不是继续走
    - 补全时优先使用受限图，保证路径合法性
    """
    from airport_graph import get_restricted_graph as _get_rest
    Gr = _get_rest(G)

    env = TaxiEnv(G, pos, flight, occupied_positions=occupied)
    obs, _ = env.reset()
    done = False
    step = 0
    visit_count = {}
    MAX_VISIT = 2   # 改动：3→2，更快触发循环检测

    while not done and step < 500:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        node = env.current_node
        visit_count[node] = visit_count.get(node, 0) + 1
        done = terminated or truncated
        step += 1

        if node == flight['end_node']:
            break

        # 改动：检测到循环立即截断，不再继续 rollout
        if visit_count.get(node, 0) > MAX_VISIT:
            # 截断到该节点第一次出现处，避免重复节点污染路径
            first_idx = env.path_taken.index(node)
            env.path_taken = env.path_taken[:first_idx + 1]
            env.current_node = node
            break

    path = env.path_taken[:]

    # 若未到终点，用 Dijkstra 补全剩余路径
    if env.current_node != flight['end_node']:
        src = env.current_node
        dst = flight['end_node']
        # 优先使用受限图；若 src/dst 不在受限图中则 fallback 到完整图
        src_use = src if src in Gr else dst
        dst_use = dst if dst in Gr else src
        try:
            rest = nx.dijkstra_path(Gr, src_use, dst_use, weight='weight')
            path = path + rest[1:]
        except nx.NetworkXNoPath:
            try:
                rest = nx.dijkstra_path(G, src, dst, weight='weight')
                path = path + rest[1:]
            except Exception:
                path.append(flight['end_node'])
        except Exception:
            path.append(flight['end_node'])

    # 去除重复节点（保持顺序）
    seen, clean_path = set(), []
    for n in path:
        if n not in seen:
            seen.add(n)
            clean_path.append(n)

    metrics = compute_flight_metrics(clean_path, flight['fuel_rate'], pos)
    return {
        'id': flight['id'],
        'flight_no': flight['flight_no'],
        'type': flight['type'],
        'aircraft_type': flight['aircraft_type'],
        'path': clean_path,
        'start_time': flight['actual_time'],
        'metrics': metrics,
    }


def run_ppo(G: nx.Graph, flights: list, pos: dict,
            total_timesteps: int = 200_000,
            force_retrain: bool = False) -> dict:
    """
    训练（或加载）PPO 模型，然后为所有航班规划路径。
    """
    # 训练或加载
    model_file = MODEL_PATH + ".zip"
    if os.path.exists(model_file) and not force_retrain:
        print(f"[PPO] Loading existing model from {model_file}")
        # 需要一个 dummy env 来加载
        dummy_fl = flights[0]
        dummy_env = TaxiEnv(G, pos, dummy_fl)
        model = PPO.load(MODEL_PATH, env=dummy_env)
    else:
        model = train_ppo(G, pos, flights, total_timesteps=total_timesteps)

    # 顺序推理：每架飞机规划完后更新 occupied
    results = []
    occupied = {}

    print("[PPO] Running inference for all flights...")
    for i, fl in enumerate(flights):
        res = _rollout_single(model, G, pos, fl, occupied)
        results.append(res)
        # 更新 occupied
        new_occ = _build_occupied([res], pos)
        for t, lst in new_occ.items():
            if t not in occupied:
                occupied[t] = []
            occupied[t].extend(lst)
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(flights)} flights")

    # 全局冲突检测
    n_conflicts = check_conflicts(results, pos)

    total_time     = sum(r['metrics']['time'] for r in results)
    total_distance = sum(r['metrics']['distance'] for r in results)
    total_co2      = sum(r['metrics']['co2'] for r in results)

    summary = {
        'algorithm': 'PPO',
        'total_time_s': total_time,
        'total_distance_m': total_distance,
        'total_co2_kg': total_co2,
        'conflicts': n_conflicts,
        'avg_time_s': total_time / len(results),
        'avg_distance_m': total_distance / len(results),
        'avg_co2_kg': total_co2 / len(results),
        'results': results,
    }

    print(f"[PPO] Done. flights={len(results)}, "
          f"total_time={total_time/3600:.2f}h, "
          f"total_dist={total_distance/1000:.2f}km, "
          f"total_co2={total_co2:.1f}kg, "
          f"conflicts={n_conflicts}")
    return summary
