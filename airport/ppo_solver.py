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


def train_ppo(G: nx.Graph, pos: dict, flights: list,
              total_timesteps: int = 200_000,
              n_envs: int = 4) -> PPO:
    """
    训练 PPO 模型。使用所有航班轮转训练。

    关键改动：
    1. 用 Dijkstra 结果预填充 occupied，让训练期间环境就能看到多机冲突
    2. 增加 EntropyScheduleCallback，防止 entropy 过早坍缩
    3. 扩大网络容量至 [256,256,128]，增强对复杂场景的表达力
    4. 降低 ent_coef 初始值，配合回调动态调整
    """
    print(f"[PPO] Training for {total_timesteps} steps with {n_envs} envs...")

    # ── 预填充 occupied（训练/推理分布对齐）────────────────────────
    # 原问题：训练时 occupied 为空，策略从未在密集冲突场景下被训练，
    # 导致推理时面对 137 架并发时冲突激增。
    # 修复：用 Dijkstra 基线路径预先构建 occupied 作为训练背景。
    from dijkstra_solver import run_dijkstra
    dijk_summary = run_dijkstra(G, flights, pos)
    static_occupied = _build_occupied(dijk_summary['results'], pos)
    print(f"[PPO] Pre-filled occupied table: {len(static_occupied)} time slots")

    env_fns = [make_env_fn(G, pos, flights, i, [static_occupied])
               for i in range(n_envs)]
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec_env = DummyVecEnv(env_fns)

    # ── Entropy 调度回调 ──────────────────────────────────────────
    # 原问题：entropy 在约前 1/4 训练后就停在 -1.5 附近不再下降，
    # 剩余 200 万步在强化一个次优策略。
    # 修复：监控 entropy，当它停止下降超过 patience 步时，
    # 短暂提升 ent_coef 重新注入探索性，避免陷入局部最优。
    from stable_baselines3.common.callbacks import BaseCallback

    class EntropyScheduleCallback(BaseCallback):
        """
        动态 entropy 调度：
        - 记录近期 entropy 均值
        - 若连续 patience 次迭代 entropy 提升不足 threshold，
          将 ent_coef 提升至 boost_value 持续 boost_steps 步后恢复
        """
        def __init__(self, patience=30, threshold=0.01,
                     boost_value=0.05, boost_steps=20, verbose=0):
            super().__init__(verbose)
            self.patience = patience
            self.threshold = threshold
            self.boost_value = boost_value
            self.boost_steps = boost_steps
            self._entropy_history = []
            self._stagnant_count = 0
            self._boosting = 0          # 剩余 boost 迭代数
            self._base_ent_coef = None  # 恢复用的原始值

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            # 从 logger 读取当前 entropy（SB3 内部 key）
            ent = self.logger.name_to_value.get('train/entropy_loss', None)
            if ent is None:
                return

            self._entropy_history.append(ent)

            if self._boosting > 0:
                self._boosting -= 1
                if self._boosting == 0:
                    # 恢复原始 ent_coef
                    self.model.ent_coef = self._base_ent_coef
                    if self.verbose:
                        print(f"[EntropySchedule] Restored ent_coef={self._base_ent_coef:.4f}")
                return

            if len(self._entropy_history) < self.patience:
                return

            recent = self._entropy_history[-self.patience:]
            # entropy 是负值，越接近 0 表示越均匀（探索越充分）
            # 若近期均值几乎不变（变化 < threshold），说明已停滞
            improvement = abs(recent[-1] - recent[0])
            if improvement < self.threshold:
                self._stagnant_count += 1
                if self._stagnant_count >= 2:
                    self._base_ent_coef = self.model.ent_coef
                    self.model.ent_coef = self.boost_value
                    self._boosting = self.boost_steps
                    self._stagnant_count = 0
                    if self.verbose:
                        print(f"[EntropySchedule] Entropy stagnant, "
                              f"boosting ent_coef to {self.boost_value} "
                              f"for {self.boost_steps} iters")
            else:
                self._stagnant_count = 0

    entropy_cb = EntropyScheduleCallback(
        patience=40,       # 观察窗口（iterations）
        threshold=0.008,   # 低于此改善量视为停滞
        boost_value=0.06,  # boost 期间的 ent_coef
        boost_steps=25,    # boost 持续迭代数
        verbose=1,
    )

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
        ent_coef=0.02,
        vf_coef=1.0,
        max_grad_norm=0.3,
        verbose=1,
        tensorboard_log=None,
        normalize_advantage=True,
        policy_kwargs=dict(net_arch=[256, 256, 128]),
    )

    model.learn(total_timesteps=total_timesteps,
                callback=entropy_cb,
                progress_bar=False)

    model.save(MODEL_PATH)
    print(f"[PPO] Model saved to {MODEL_PATH}")
    vec_env.close()
    return model


def dijkstra_with_conflict_penalty(G: nx.Graph, src: str, dst: str,
                                     occupied: dict, t_current: float,
                                     pos: dict) -> list:
    """
    带冲突惩罚的 Dijkstra 补全：被占用的边权重×10，绕开已规划飞机的路径。
    """
    G_penalized = G.copy()
    t_key = int(t_current)
    # 只在有占用数据时才做惩罚计算，避免无谓遍历
    if t_key in occupied:
        occ_positions = [np.array(xy) for _, xy in occupied[t_key]]
        for u, v, data in G.edges(data=True):
            if u not in pos or v not in pos:
                continue
            base_w = data.get('weight', 1.0)
            mid_xy = (np.array(pos[u]) + np.array(pos[v])) / 2.0
            for occ_xy in occ_positions:
                if np.linalg.norm(mid_xy - occ_xy) < 200.0:
                    G_penalized[u][v]['weight'] = base_w * 10.0
                    break  # 一个冲突就够了，不需要继续检查
    return nx.dijkstra_path(G_penalized, src, dst, weight='weight')


def _rollout_single(model: PPO, G: nx.Graph, pos: dict,
                    flight: dict, occupied: dict) -> dict:
    """
    用训练好的模型为单架飞机执行推理，返回路径和指标。
    快速模式：最多走 1500 步，超过或检测到死循环则直接 Dijkstra 补全。

    修复：原代码在循环检测分支对 path 赋值后，被后续
    `path = env.path_taken[:]` 无条件覆盖，截断逻辑完全失效。
    现将截断状态用 early_stop_path 单独保存。
    """
    from airport_graph import get_restricted_graph as _get_rest
    Gr = _get_rest(G)

    env = TaxiEnv(G, pos, flight, occupied_positions=occupied)
    obs, _ = env.reset()
    done = False
    step = 0
    visit_count = {}
    MAX_VISIT = 3   # 同一节点最多经过3次，超过则认为陷入死循环
    early_stop_path = None  # 循环检测触发时保存截断后的路径

    while not done and step < 1500:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        node = env.current_node
        visit_count[node] = visit_count.get(node, 0) + 1
        done = terminated or truncated
        step += 1

        if node == flight['end_node']:
            break

        # 检测循环：重访次数超限时截断路径，交由 Dijkstra 补全
        if visit_count.get(node, 0) > MAX_VISIT:
            # 找到该节点第一次出现的位置，截断多余的绕圈部分
            path_so_far = env.path_taken[:]
            if node in path_so_far:
                first_idx = path_so_far.index(node)
                early_stop_path = path_so_far[:first_idx + 1]
            else:
                early_stop_path = path_so_far
            done = True  # 强制退出推理循环
            break

    # 确定当前已走路径
    if early_stop_path is not None:
        path = early_stop_path
        # 同步 env.current_node 以便后续 Dijkstra 从正确节点补全
        current_node_for_completion = path[-1]
    else:
        path = env.path_taken[:]
        current_node_for_completion = env.current_node

    # 如果没到终点，用 Dijkstra 补全剩余路径
    if current_node_for_completion != flight['end_node']:
        src = current_node_for_completion
        dst = flight['end_node']
        src_use = src if src in Gr else dst
        dst_use = dst if dst in Gr else src
        t_current = env.t_global
        try:
            rest = dijkstra_with_conflict_penalty(Gr, src_use, dst_use, occupied, t_current, pos)
            path = path + rest[1:]
        except Exception:
            try:
                rest = dijkstra_with_conflict_penalty(G, src, dst, occupied, t_current, pos)
                path = path + rest[1:]
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
