"""
PPO 双智能体网络（优化版）
优化项：
  1. 向量化前向：候选集合整体做一次 Linear，不再逐条调用
  2. 混合精度训练（AMP）：autocast 包裹前向，GradScaler 管理反向
  3. 网络容量提升：HIDDEN 256，支持更大 batch
"""
import torch
import torch.nn as nn
import numpy as np

# ── 自动选择设备 ──────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 超参数 ────────────────────────────────────────────────────────
ORDER_FEAT_DIM  = 25
DEVICE_FEAT_DIM = 25
GLOBAL_FEAT_DIM = 3
HIDDEN          = 256   # 64 → 256，充分利用显存


class FeatureExtractor(nn.Module):
    def __init__(self, order_dim=ORDER_FEAT_DIM,
                 device_dim=DEVICE_FEAT_DIM,
                 global_dim=GLOBAL_FEAT_DIM,
                 hidden=HIDDEN):
        super().__init__()
        self.order_enc = nn.Sequential(
            nn.Linear(order_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        self.device_enc = nn.Sequential(
            nn.Linear(device_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
        )
        self.global_enc = nn.Sequential(
            nn.Linear(global_dim, 64), nn.ReLU(),
        )
        self.aggregate = nn.Sequential(
            nn.Linear(hidden * 2 + 64, hidden), nn.ReLU(),
        )

    def forward(self, order_feat, device_feat, global_feat):
        # 向量化：整批一起过 Linear，而非逐条
        o = self.order_enc(order_feat).mean(dim=1)    # (B, H)
        d = self.device_enc(device_feat).mean(dim=1)  # (B, H)
        g = self.global_enc(global_feat)              # (B, 64)
        return self.aggregate(torch.cat([o, d, g], dim=-1))  # (B, H)


class HeuristicFusion(nn.Module):
    def __init__(self, n_rules, hidden=HIDDEN):
        super().__init__()
        self.attn    = nn.Linear(hidden, n_rules)
        self.out_dim = hidden + n_rules

    def forward(self, repr_vec, h_mean):
        w = torch.softmax(self.attn(repr_vec), dim=-1)
        return torch.cat([repr_vec, w * h_mean], dim=-1)


class OrderSelectionNet(nn.Module):
    N_RULES = 2

    def __init__(self):
        super().__init__()
        self.feat_ext   = FeatureExtractor()
        self.hfuse      = HeuristicFusion(self.N_RULES)
        fused = HIDDEN + self.N_RULES
        # 向量化打分：一次 Linear 处理所有候选
        self.cand_enc   = nn.Sequential(nn.Linear(ORDER_FEAT_DIM, HIDDEN), nn.ReLU())
        self.score_head = nn.Linear(HIDDEN + fused, 1)
        self.value_head = nn.Sequential(
            nn.Linear(fused, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1)
        )

    def forward(self, order_feat, device_feat, global_feat,
                candidate_indices, heuristic_scores):
        repr_v  = self.feat_ext(order_feat, device_feat, global_feat)  # (1, H)
        h_mean  = heuristic_scores.mean(0, keepdim=True)               # (1, 2)
        fused   = self.hfuse(repr_v, h_mean)                           # (1, H+2)

        # 向量化：所有候选订单整体一次前向
        cand_f  = order_feat[0, candidate_indices, :]                  # (n, OF)
        cand_e  = self.cand_enc(cand_f)                                # (n, H)
        fused_e = fused.expand(len(candidate_indices), -1)             # (n, H+2)
        logits  = self.score_head(
            torch.cat([cand_e, fused_e], dim=-1)
        ).squeeze(-1)                                                   # (n,)

        value = self.value_head(fused).squeeze()
        return logits, value


class MachineAssignmentNet(nn.Module):
    N_RULES = 4

    def __init__(self):
        super().__init__()
        self.feat_ext   = FeatureExtractor()
        self.hfuse      = HeuristicFusion(self.N_RULES)
        fused = HIDDEN + self.N_RULES
        # 向量化打分：所有候选设备整体一次前向
        self.cand_enc   = nn.Sequential(nn.Linear(DEVICE_FEAT_DIM, HIDDEN), nn.ReLU())
        self.score_head = nn.Linear(HIDDEN + fused, 1)
        self.value_head = nn.Sequential(
            nn.Linear(fused, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1)
        )

    def forward(self, order_feat, device_feat, global_feat,
                candidate_device_indices, heuristic_scores):
        repr_v  = self.feat_ext(order_feat, device_feat, global_feat)
        h_mean  = heuristic_scores.mean(0, keepdim=True)
        fused   = self.hfuse(repr_v, h_mean)

        cand_f  = device_feat[0, candidate_device_indices, :]
        cand_e  = self.cand_enc(cand_f)
        fused_e = fused.expand(len(candidate_device_indices), -1)
        logits  = self.score_head(
            torch.cat([cand_e, fused_e], dim=-1)
        ).squeeze(-1)

        value = self.value_head(fused).squeeze()
        return logits, value


# ── 启发式得分计算 ──────────────────────────────────────────────────
def compute_order_heuristics(env, candidate_order_stages):
    scores = []
    for order_idx, stage in candidate_order_stages:
        order   = env.data['orders'][order_idx]
        s_type  = 0.5 if order['order_type'] == 'MTO' else 0.3
        min_rem = env.get_min_remaining(order_idx, stage)
        delta_t = order['due_time'] - env.current_time - min_rem
        s_delay = 1.0 if delta_t <= 0 else (0.7 if delta_t <= 86400 else 0.3)
        scores.append([s_type, s_delay])
    return torch.tensor(scores, dtype=torch.float32, device=DEVICE)


def compute_machine_heuristics(env, order_idx, stage, candidate_devices):
    from data_loader import get_setup_seconds
    n_types    = len(env.data['product_types'])
    ptype      = env.data['orders'][order_idx]['product_type']
    scores     = []
    proc_times = []

    for d in candidate_devices:
        s_specif = 1.0 - len(env.data['device_compatibility'][d]) / n_types
        t_now    = max(env.current_time, 1.0)
        s_util   = 1.0 - env.device_total_proc[d] / t_now
        idle_t   = max(0.0, env.current_time - env.device_busy_until[d])
        setup_s  = get_setup_seconds(
            env.data, stage, env.device_last_type[d], ptype, idle_t)
        s_setup  = 0.5 if setup_s == 0 else 0.3
        pt       = env.data['processing_time'].get((d, ptype), float('inf'))
        proc_times.append(pt)
        scores.append([s_specif, s_util, s_setup, 0.0])

    if proc_times:
        n        = len(proc_times)
        rank_map = {}
        for rank, pt in enumerate(sorted(proc_times), 1):
            if pt not in rank_map:
                rank_map[pt] = rank / n
        for i, pt in enumerate(proc_times):
            scores[i][3] = rank_map[pt]

    return torch.tensor(scores, dtype=torch.float32, device=DEVICE)


def state_to_tensors(state):
    of = torch.tensor(
        state['order_feat'],  dtype=torch.float32, device=DEVICE).unsqueeze(0)
    df = torch.tensor(
        state['device_feat'], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    gf = torch.tensor(
        state['global_feat'], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    return of, df, gf
