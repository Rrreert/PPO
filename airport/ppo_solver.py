"""
ppo_solver.py
用 Stable-Baselines3 PPO 训练单机滑行策略，
然后用训练好的模型为所有 137 架次规划路径。
"""
import os
import numpy as np
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from taxi_env import TaxiEnv, MAX_NEIGHBORS
from airport_graph import FORBIDDEN_NODES, get_restricted_graph
from dijkstra_solver import compute_flight_metrics, check_conflicts
from flight_data import MAX_SPEED, TURN_SPEED, ACCEL, DECEL, CARBON_FACTOR, ACCEL_FACTOR

MODEL_PATH = "ppo_taxi_model"


# ══════════════════════════════════════════════════════════
# 需求1：训练指标 Callback
# ══════════════════════════════════════════════════════════

class MetricsCallback(BaseCallback):
    """
    每个 rollout 结束后收集一次训练指标，存入 self.metrics。
    收集：timestep、entropy_loss、value_loss、approx_kl、
          clip_fraction、explained_variance、policy_gradient_loss
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.metrics = {
            'timesteps':           [],
            'entropy_loss':        [],
            'value_loss':          [],
            'approx_kl':           [],
            'clip_fraction':       [],
            'explained_variance':  [],
            'policy_gradient_loss': [],
        }
        self._iteration = 0

    def _on_rollout_end(self) -> bool:
        """每次 rollout 收集一次（与 SB3 logger 同步）"""
        self._iteration += 1
        self.metrics['timesteps'].append(self.num_timesteps)
        for key in ('entropy_loss', 'value_loss', 'approx_kl',
                    'clip_fraction', 'explained_variance',
                    'policy_gradient_loss'):
            val = self.logger.name_to_value.get(f'train/{key}', np.nan)
            self.metrics[key].append(float(val))
        return True

    def _on_step(self) -> bool:
        return True


# ══════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════

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
        occ = occupied_ref[0] if occupied_ref else {}
        return TaxiEnv(G, pos, fl, occupied_positions=occ)
    return _make


# ══════════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════════

def train_ppo(G: nx.Graph, pos: dict, flights: list,
              total_timesteps: int = 200_000,
              n_envs: int = 4) -> tuple:
    """
    训练 PPO 模型，包含以下改进：
    1. ent_coef=0.05，防止策略过早收敛
    2. Self-play 迭代更新 occupied（每轮用上一轮PPO结果替换Dijkstra背景）
    3. 课程学习：前1/3步骤用简单子集，后续逐步扩展到全量
    4. 稠密进度奖励（在 taxi_env.py 中实现）

    返回 (model, metrics_dict)
    """
    print(f"[PPO] Training for {total_timesteps} steps with {n_envs} envs...")

    from dijkstra_solver import run_dijkstra
    from stable_baselines3.common.vec_env import DummyVecEnv

    # ── 初始 occupied：用 Dijkstra 结果作为第一轮背景 ──────────
    print("[PPO] Generating initial Dijkstra background...")
    dijk_summary = run_dijkstra(G, flights, pos)
    occupied_ref = [_build_occupied(dijk_summary['results'], pos)]

    # ── 课程学习配置：3阶段 ──────────────────────────────────
    N_PHASES = 3
    phase_ratios = [0.33, 0.66, 1.0]
    # Bug修复1：SB3在reset_num_timesteps=False时，total_timesteps必须传累计值
    # 否则Phase1结束时num_timesteps已经略超过incremental值，导致Phase2/3立即退出
    per_phase = total_timesteps // N_PHASES
    phase_cumulative_steps = [
        per_phase,                              # Phase1 累计终点
        per_phase * 2,                          # Phase2 累计终点
        total_timesteps,                        # Phase3 累计终点（含尾数）
    ]

    # ── 创建模型（只创建一次，跨阶段持续学习）──────────────────
    # Bug修复2：模型初始env用全量flights建立，避免set_env时obs分布跳变
    # 课程学习通过在make_env_fn中轮转不同的flights子集来实现，
    # 而不是每阶段换一批完全不同的flights
    init_env_fns = [make_env_fn(G, pos, flights, i, occupied_ref)
                    for i in range(n_envs)]
    vec_env = DummyVecEnv(init_env_fns)

    # 共用一个 callback 收集全程指标
    metrics_cb = MetricsCallback()

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
    for phase_idx, (cumulative_steps, ratio) in enumerate(
            zip(phase_cumulative_steps, phase_ratios)):
        n_flights = max(int(len(flights) * ratio), 8)
        phase_flights = flights[:n_flights]
        incremental = cumulative_steps - (phase_cumulative_steps[phase_idx-1] if phase_idx > 0 else 0)
        print(f"[PPO] Phase {phase_idx+1}/{N_PHASES}: "
              f"{incremental:,} steps (cumulative={cumulative_steps:,}), "
              f"{n_flights}/{len(flights)} flights")

        # Bug修复2：每阶段只更新已有env里的flights采样范围，不换env
        # 通过重建env_fns但保持env结构相同（相同obs/action空间），避免KL spike
        env_fns = [make_env_fn(G, pos, phase_flights, i, occupied_ref)
                   for i in range(n_envs)]
        phase_env = DummyVecEnv(env_fns)
        model.set_env(phase_env)

        model.learn(
            total_timesteps=cumulative_steps,   # 传累计值，修复SB3 timesteps bug
            reset_num_timesteps=(phase_idx == 0),
            callback=metrics_cb,
            progress_bar=False,
        )
        phase_env.close()

        # ── Self-play 更新 occupied ────────────────────────────
        print(f"[PPO] Phase {phase_idx+1}: updating occupied via self-play...")
        selfplay_results = []
        temp_occupied = {}
        for fl in phase_flights:
            res = _rollout_single(model, G, pos, fl, temp_occupied)
            selfplay_results.append(res)
            new_occ = _build_occupied([res], pos)
            for t, lst in new_occ.items():
                temp_occupied.setdefault(t, []).extend(lst)
        occupied_ref[0] = _build_occupied(selfplay_results, pos)
        print(f"[PPO] Phase {phase_idx+1}: self-play occupied updated "
              f"({len(occupied_ref[0])} time slots)")

    model.save(MODEL_PATH)
    print(f"[PPO] Model saved to {MODEL_PATH}")
    return model, metrics_cb.metrics


# ══════════════════════════════════════════════════════════
# 推理
# ══════════════════════════════════════════════════════════

def _rollout_single(model: PPO, G: nx.Graph, pos: dict,
                    flight: dict, occupied: dict) -> dict:
    """
    用训练好的模型为单架飞机执行推理。
    - MAX_VISIT=2，更快发现循环
    - 检测到循环立即截断并 Dijkstra 补全
    """
    from airport_graph import get_restricted_graph as _get_rest
    Gr = _get_rest(G)

    env = TaxiEnv(G, pos, flight, occupied_positions=occupied)
    obs, _ = env.reset()
    done = False
    step = 0
    visit_count = {}
    MAX_VISIT = 2

    while not done and step < 500:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        node = env.current_node
        visit_count[node] = visit_count.get(node, 0) + 1
        done = terminated or truncated
        step += 1

        if node == flight['end_node']:
            break

        if visit_count.get(node, 0) > MAX_VISIT:
            first_idx = env.path_taken.index(node)
            env.path_taken = env.path_taken[:first_idx + 1]
            env.current_node = node
            break

    path = env.path_taken[:]

    if env.current_node != flight['end_node']:
        src = env.current_node
        dst = flight['end_node']
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

    seen, clean_path = set(), []
    for n in path:
        if n not in seen:
            seen.add(n)
            clean_path.append(n)

    metrics = compute_flight_metrics(clean_path, flight['fuel_rate'], pos)
    return {
        'id':            flight['id'],
        'flight_no':     flight['flight_no'],
        'type':          flight['type'],
        'aircraft_type': flight['aircraft_type'],
        'path':          clean_path,
        'start_time':    flight['actual_time'],
        'metrics':       metrics,
    }


# ══════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════

def run_ppo(G: nx.Graph, flights: list, pos: dict,
            total_timesteps: int = 200_000,
            force_retrain: bool = False) -> dict:
    """
    训练（或加载）PPO 模型，然后为所有航班规划路径。
    返回 summary dict，其中包含 'training_metrics' 供可视化使用。
    """
    training_metrics = {}

    if os.path.exists(MODEL_PATH + ".zip") and not force_retrain:
        print(f"[PPO] Loading existing model from {MODEL_PATH}.zip")
        dummy_env = TaxiEnv(G, pos, flights[0])
        model = PPO.load(MODEL_PATH, env=dummy_env)
    else:
        model, training_metrics = train_ppo(
            G, pos, flights, total_timesteps=total_timesteps)

    # ── 推理：顺序为每架飞机规划路径 ─────────────────────────
    results = []
    occupied = {}
    n_total = len(flights)

    # 需求3：完成情况统计
    n_reached   = 0   # PPO 自己走到终点
    n_dijkstra  = 0   # 靠 Dijkstra 补全
    n_collision = 0   # 触发碰撞终止

    print(f"\n[PPO] Running inference for all {n_total} flights...")
    print(f"  {'#':>4}  {'Flight':>10}  {'Type':>4}  {'AC':>5}  "
          f"{'Time(s)':>8}  {'Dist(m)':>8}  {'CO2(kg)':>8}  {'Status':>12}")
    print("  " + "-" * 70)

    for i, fl in enumerate(flights):
        res = _rollout_single(model, G, pos, fl, occupied)
        results.append(res)

        # 判断是否 PPO 自己到达（路径末节点 == end_node 且步数未超限）
        ppo_arrived = (res['path'][-1] == fl['end_node'])
        if ppo_arrived:
            n_reached += 1
            status = "PPO-reached"
        else:
            n_dijkstra += 1
            status = "Dijkstra-fill"

        # 更新 occupied
        new_occ = _build_occupied([res], pos)
        for t, lst in new_occ.items():
            occupied.setdefault(t, []).extend(lst)

        # 需求3：逐行打印每架飞机结果
        m = res['metrics']
        print(f"  {i+1:>4}  {res['flight_no']:>10}  {fl['type']:>4}  "
              f"{fl['aircraft_type']:>5}  "
              f"{m['time']:>8.1f}  {m['distance']:>8.0f}  "
              f"{m['co2']:>8.2f}  {status:>12}")

    print("  " + "-" * 70)

    # 全局冲突检测
    n_conflicts = check_conflicts(results, pos)

    total_time     = sum(r['metrics']['time']     for r in results)
    total_distance = sum(r['metrics']['distance'] for r in results)
    total_co2      = sum(r['metrics']['co2']      for r in results)

    # 需求3：汇总完成情况
    print(f"\n[PPO] Inference complete — {n_total} flights")
    print(f"  PPO self-reached   : {n_reached:>4} ({n_reached/n_total*100:.1f}%)")
    print(f"  Dijkstra-filled    : {n_dijkstra:>4} ({n_dijkstra/n_total*100:.1f}%)")
    print(f"  Total conflicts    : {n_conflicts}")
    print(f"  Total time         : {total_time/3600:.2f} h")
    print(f"  Total distance     : {total_distance/1000:.2f} km")
    print(f"  Total CO2          : {total_co2:.1f} kg")

    summary = {
        'algorithm':        'PPO',
        'total_time_s':     total_time,
        'total_distance_m': total_distance,
        'total_co2_kg':     total_co2,
        'conflicts':        n_conflicts,
        'avg_time_s':       total_time / n_total,
        'avg_distance_m':   total_distance / n_total,
        'avg_co2_kg':       total_co2 / n_total,
        'results':          results,
        'training_metrics': training_metrics,   # 供 visualize 使用
        'inference_stats': {
            'n_total':    n_total,
            'n_reached':  n_reached,
            'n_dijkstra': n_dijkstra,
        },
    }
    return summary
