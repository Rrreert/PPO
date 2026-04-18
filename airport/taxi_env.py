"""
taxi_env.py
虹桥机场多机 PPO 滑行环境（Gymnasium 接口）。

设计：每个 episode 对单架飞机进行滑行仿真（sequential），
多机冲突通过记录全局占用时间表来惩罚。
状态空间、动作空间都以当前节点的邻居为基础（离散动作）。
"""
import numpy as np
import networkx as nx
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
from flight_data import (CARBON_FACTOR, ACCEL_FACTOR, MAX_SPEED,
                          TURN_SPEED, ACCEL, DECEL, DT, D_MIN, D_SAFE)
from airport_graph import FORBIDDEN_NODES, get_restricted_graph
from dijkstra_solver import compute_flight_metrics, _angle_at_node

# ─── PPO 奖励权重 ────────────────────────────
W_TIME     = -0.005   # 每秒时间惩罚
W_CARBON   = -0.020   # 每 kg CO2 惩罚
W_SAFE     = -2.000   # 安全距离惩罚系数
W_PROGRESS =  5.000   # 进度奖励系数（↑ 从2.0提高到5.0，加强稠密引导）
W_STEP     = -0.050   # 每步固定惩罚（↑ 新增，防止绕路拖延）

MAX_STEPS  = 3000     # 单 episode 最大步数
MAX_NEIGHBORS = 8     # 最大邻居数（动作空间上界）


class TaxiEnv(gym.Env):
    """
    单架飞机滑行环境。
    其他飞机的轨迹通过 occupied_positions（时间→[(node,xy)]）注入，
    用于冲突惩罚计算。
    """
    metadata = {"render_modes": []}

    def __init__(self, G: nx.Graph, pos: dict, flight: dict,
                 occupied_positions: dict = None,
                 max_neighbors: int = MAX_NEIGHBORS):
        super().__init__()
        self.G_full = G
        self.G_restricted = get_restricted_graph(G)
        self.pos = pos
        self.flight = flight
        self.occupied = occupied_positions or {}
        self.max_nb = max_neighbors

        # ── 动作空间：离散，选择邻居索引 ─────────
        self.action_space = spaces.Discrete(max_neighbors)

        # ── 状态空间：
        # [当前节点归一化x, 当前节点归一化y,
        #  目标节点归一化x, 目标节点归一化y,
        #  到目标剩余距离（归一化）,
        #  当前速度（归一化）,
        #  最近冲突飞机距离（归一化）,
        #  邻居1..8 各自的 (dx, dy, dist) 共 max_neighbors*3 维]
        obs_dim = 7 + max_neighbors * 3
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32)

        # 坐标归一化
        xs = [v[0] for v in pos.values()]
        ys = [v[1] for v in pos.values()]
        self.x_min, self.x_max = min(xs), max(xs)
        self.y_min, self.y_max = min(ys), max(ys)
        self.xy_scale = max(self.x_max - self.x_min,
                            self.y_max - self.y_min, 1.0)
        # 最大可能距离（归一化用）
        self.max_dist = self.xy_scale * 2

        # ── 预计算 Dijkstra 最短路距离，用于稠密进度奖励 ──
        # 从终点出发做单源最短路，得到"每个节点→终点"的图距离
        self._dijk_dist_cache: dict = {}
        try:
            lengths = nx.single_source_dijkstra_path_length(
                self.G_restricted, self.flight['end_node'], weight='weight')
            self._dijk_dist_cache = dict(lengths)
        except Exception:
            pass  # 图不连通时退化为欧氏距离

        self._reset_state()

    def _norm_xy(self, node):
        if node not in self.pos:
            return 0.0, 0.0
        x, y = self.pos[node]
        return (x - self.x_min) / self.xy_scale, (y - self.y_min) / self.xy_scale

    def _dist_to_goal(self, node):
        # 优先使用预计算的图距离（更准确），fallback到欧氏距离
        if node in self._dijk_dist_cache:
            return self._dijk_dist_cache[node]
        if node not in self.pos or self.goal not in self.pos:
            return self.max_dist
        return np.linalg.norm(
            np.array(self.pos[node]) - np.array(self.pos[self.goal])
        )

    def _get_neighbors(self, node):
        """返回受限图中的邻居列表（最多 max_nb 个）"""
        G_use = self.G_restricted if node in self.G_restricted else self.G_full
        nbs = list(G_use.neighbors(node))
        # 排除禁用节点
        nbs = [n for n in nbs if n not in FORBIDDEN_NODES]
        return nbs[:self.max_nb]

    def _reset_state(self):
        self.current_node = self.flight['start_node']
        self.goal = self.flight['end_node']
        self.step_count = 0
        self.elapsed_time = 0.0
        self.speed = 0.0
        self.path_taken = [self.current_node]
        self.total_co2 = 0.0
        self.prev_dist = self._dist_to_goal(self.current_node)
        # self.init_dist = self.prev_dist
        # self.traveled_dist = 0.0
        self.done_flag = False
        self.t_global = self.flight['actual_time']

    def _get_obs(self):
        cx, cy = self._norm_xy(self.current_node)
        gx, gy = self._norm_xy(self.goal)
        dist_norm = min(self._dist_to_goal(self.current_node) / self.max_dist, 1.0)
        speed_norm = self.speed / MAX_SPEED

        # 最近冲突飞机距离
        cur_pos = np.array(self.pos.get(self.current_node, (0, 0)))
        min_conf_dist = 1.0  # 归一化，1 = 没有冲突
        t_key = int(self.t_global)
        if t_key in self.occupied:
            for _, xy in self.occupied[t_key]:
                d = np.linalg.norm(cur_pos - np.array(xy))
                nd = min(d / D_SAFE, 1.0)
                if nd < min_conf_dist:
                    min_conf_dist = nd

        # 邻居方向特征：每个邻居编码为 (dx, dy, dist)，无邻居处填0
        nbs = self._get_neighbors(self.current_node)
        nb_features = np.zeros(self.max_nb * 3, dtype=np.float32)
        cur_xy = np.array(self.pos.get(self.current_node, (0.0, 0.0)))
        for i, nb in enumerate(nbs[:self.max_nb]):
            if nb in self.pos:
                nb_xy = np.array(self.pos[nb])
                dx = (nb_xy[0] - cur_xy[0]) / self.xy_scale
                dy = (nb_xy[1] - cur_xy[1]) / self.xy_scale
                d  = np.linalg.norm([dx, dy])
                nb_features[i*3:i*3+3] = [dx, dy, d]

        obs = np.array([cx, cy, gx, gy, dist_norm, speed_norm,
                        min_conf_dist], dtype=np.float32)
        obs = np.concatenate([obs, nb_features])
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: int):
        if self.done_flag:
            return self._get_obs(), 0.0, True, False, {}

        nbs = self._get_neighbors(self.current_node)
        if len(nbs) == 0:
            # 死路：大惩罚
            self.done_flag = True
            return self._get_obs(), -50.0, True, False, {'reason': 'dead_end'}

        # 动作裁剪：clip 到有效邻居范围，避免取模导致的动作分布偏斜
        action = min(int(action), len(nbs) - 1)
        next_node = nbs[action]

        # ── 运动物理 ───────────────────────────
        # 判断是否转弯
        angle = 0.0
        if len(self.path_taken) >= 2:
            prev_n = self.path_taken[-2]
            if prev_n in self.pos and self.current_node in self.pos and next_node in self.pos:
                v1 = np.array(self.pos[self.current_node]) - np.array(self.pos[prev_n])
                v2 = np.array(self.pos[next_node]) - np.array(self.pos[self.current_node])
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 1e-9 and n2 > 1e-9:
                    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
                    angle = np.degrees(np.arccos(cos_a))

        is_turn = angle > 20.0
        v_target = TURN_SPEED if is_turn else MAX_SPEED

        # 段距离
        seg_len = np.linalg.norm(
            np.array(self.pos.get(next_node, (0,0))) -
            np.array(self.pos.get(self.current_node, (0,0))))
        # self.traveled_dist += seg_len

        # 加速/减速时间
        a_eff = ACCEL if v_target > self.speed else DECEL
        t_accel = abs(v_target - self.speed) / a_eff if abs(v_target - self.speed) > 0.01 else 0.0
        d_accel = abs(v_target**2 - self.speed**2) / (2 * a_eff) if t_accel > 0 else 0.0
        d_accel = min(d_accel, seg_len)
        d_cruise = max(0.0, seg_len - d_accel)
        t_cruise = d_cruise / v_target if v_target > 0 else 0.0

        if is_turn:
            R = 30.0
            t_turn_extra = (np.pi * R * angle / 180.0 / TURN_SPEED +
                            (MAX_SPEED - TURN_SPEED) / ACCEL)
        else:
            t_turn_extra = 0.0

        dt_seg = t_accel + t_cruise + t_turn_extra

        # CO2
        co2_seg = (self.flight['fuel_rate'] * ACCEL_FACTOR * CARBON_FACTOR * t_accel +
                   self.flight['fuel_rate'] * CARBON_FACTOR * t_cruise +
                   self.flight['fuel_rate'] * ACCEL_FACTOR * CARBON_FACTOR * t_turn_extra)

        self.speed = v_target
        self.elapsed_time += dt_seg
        self.total_co2 += co2_seg
        self.t_global += dt_seg

        # ── 冲突检测（与已规划飞机） ───────────
        cur_xy = np.array(self.pos.get(next_node, (0, 0)))
        min_dist = float('inf')
        t_key = int(self.t_global)
        if t_key in self.occupied:
            for _, occ_xy in self.occupied[t_key]:
                d = np.linalg.norm(cur_xy - np.array(occ_xy))
                if d < min_dist:
                    min_dist = d

        # ── 奖励计算 ───────────────────────────
        # 1. 时间惩罚
        r_time = W_TIME * dt_seg

        # 2. 碳排放惩罚
        r_carbon = W_CARBON * co2_seg

        # 3. 每步固定惩罚（防止绕路拖延）
        r_step = W_STEP

        # 4. 安全惩罚
        if min_dist < D_MIN:
            r_safe = W_SAFE * (D_MIN / max(min_dist, 1e-3)) ** 2
        elif min_dist < D_SAFE:
            k = 2.0
            r_safe = W_SAFE * k * (1.0 - min_dist / D_SAFE) ** 2
        else:
            r_safe = 0.0

        # 5. 稠密进度奖励（基于图距离差，归一化到初始图距离）
        new_dist = self._dist_to_goal(next_node)
        init_dist = max(self._dijk_dist_cache.get(self.flight["start_node"], self.max_dist), 1.0)
        r_progress = W_PROGRESS * (self.prev_dist - new_dist) / init_dist
        self.prev_dist = new_dist

        reward = r_time + r_carbon + r_step + r_safe + r_progress

        # 更新状态
        self.current_node = next_node
        self.path_taken.append(next_node)
        self.step_count += 1

        # ── 终止条件 ───────────────────────────
        terminated = False
        truncated  = False
        info = {}

        if self.current_node == self.goal:
            # efficiency = self.init_dist / max(self.traveled_dist, 1.0)
            reward += 200.0  # 到达奖励
            terminated = True
            info['reason'] = 'reached_goal'
        elif min_dist < D_MIN:
            reward += W_SAFE * 100.0  # 碰撞大惩罚
            terminated = True
            info['reason'] = 'collision'
        elif self.step_count >= MAX_STEPS:
            truncated = True
            info['reason'] = 'timeout'

        self.done_flag = terminated or truncated
        reward = np.clip(reward, -50.0, 50.0)
        return self._get_obs(), float(reward), terminated, truncated, info
