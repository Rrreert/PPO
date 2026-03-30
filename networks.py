"""
PPO 双智能体网络（优化版）
共享特征提取器一次前向，再对候选集合打分，大幅提速。
  - OrderSelectionAgent：启发式规则：类别优先级 + 交货紧迫度
  - MachineAssignmentAgent：启发式规则：专用性 + 空闲率 + 换料成本 + 加工时间
"""
import torch
import torch.nn as nn
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ORDER_FEAT_DIM  = 25
DEVICE_FEAT_DIM = 25
GLOBAL_FEAT_DIM = 3
HIDDEN          = 64


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.order_enc = nn.Sequential(
            nn.Linear(ORDER_FEAT_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),         nn.ReLU(),
        )
        self.device_enc = nn.Sequential(
            nn.Linear(DEVICE_FEAT_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),          nn.ReLU(),
        )
        self.global_enc = nn.Sequential(nn.Linear(GLOBAL_FEAT_DIM, 32), nn.ReLU())
        self.aggregate  = nn.Sequential(nn.Linear(HIDDEN * 2 + 32, HIDDEN), nn.ReLU())

    def forward(self, order_feat, device_feat, global_feat):
        o = self.order_enc(order_feat).mean(dim=1)
        d = self.device_enc(device_feat).mean(dim=1)
        g = self.global_enc(global_feat)
        return self.aggregate(torch.cat([o, d, g], dim=-1))


class HeuristicFusion(nn.Module):
    def __init__(self, n_rules):
        super().__init__()
        self.attn    = nn.Linear(HIDDEN, n_rules)
        self.out_dim = HIDDEN + n_rules

    def forward(self, repr_vec, h_mean):
        w = torch.softmax(self.attn(repr_vec), dim=-1)
        return torch.cat([repr_vec, w * h_mean], dim=-1)


class OrderSelectionNet(nn.Module):
    N_RULES = 2
    def __init__(self):
        super().__init__()
        self.feat_ext  = FeatureExtractor()
        self.hfuse     = HeuristicFusion(self.N_RULES)
        fused = HIDDEN + self.N_RULES
        self.cand_enc   = nn.Sequential(nn.Linear(ORDER_FEAT_DIM, HIDDEN), nn.ReLU())
        self.score_head = nn.Linear(HIDDEN + fused, 1)
        self.value_head = nn.Sequential(nn.Linear(fused, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))

    def forward(self, order_feat, device_feat, global_feat, candidate_indices, heuristic_scores):
        repr_v = self.feat_ext(order_feat, device_feat, global_feat)
        h_mean = heuristic_scores.mean(0, keepdim=True)
        fused  = self.hfuse(repr_v, h_mean)
        cand_f = order_feat[0, candidate_indices, :]
        cand_e = self.cand_enc(cand_f)
        fused_e = fused.expand(len(candidate_indices), -1)
        logits  = self.score_head(torch.cat([cand_e, fused_e], -1)).squeeze(-1)
        value   = self.value_head(fused).squeeze()
        return logits, value


class MachineAssignmentNet(nn.Module):
    N_RULES = 4
    def __init__(self):
        super().__init__()
        self.feat_ext  = FeatureExtractor()
        self.hfuse     = HeuristicFusion(self.N_RULES)
        fused = HIDDEN + self.N_RULES
        self.cand_enc   = nn.Sequential(nn.Linear(DEVICE_FEAT_DIM, HIDDEN), nn.ReLU())
        self.score_head = nn.Linear(HIDDEN + fused, 1)
        self.value_head = nn.Sequential(nn.Linear(fused, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))

    def forward(self, order_feat, device_feat, global_feat, candidate_device_indices, heuristic_scores):
        repr_v  = self.feat_ext(order_feat, device_feat, global_feat)
        h_mean  = heuristic_scores.mean(0, keepdim=True)
        fused   = self.hfuse(repr_v, h_mean)
        cand_f  = device_feat[0, candidate_device_indices, :]
        cand_e  = self.cand_enc(cand_f)
        fused_e = fused.expand(len(candidate_device_indices), -1)
        logits  = self.score_head(torch.cat([cand_e, fused_e], -1)).squeeze(-1)
        value   = self.value_head(fused).squeeze()
        return logits, value


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
        setup_s  = get_setup_seconds(env.data, stage, env.device_last_type[d], ptype, idle_t)
        s_setup  = 0.5 if setup_s == 0 else 0.3
        pt       = env.data['processing_time'].get((d, ptype), float('inf'))
        proc_times.append(pt)
        scores.append([s_specif, s_util, s_setup, 0.0])

    if proc_times:
        n = len(proc_times)
        sorted_pts = sorted(proc_times)
        rank_map   = {}
        for rank, pt in enumerate(sorted_pts, 1):
            if pt not in rank_map:
                rank_map[pt] = rank / n
        for i, pt in enumerate(proc_times):
            scores[i][3] = rank_map[pt]

    return torch.tensor(scores, dtype=torch.float32, device=DEVICE)


def state_to_tensors(state):
    of = torch.tensor(state['order_feat'],  dtype=torch.float32, device=DEVICE).unsqueeze(0)
    df = torch.tensor(state['device_feat'], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    gf = torch.tensor(state['global_feat'], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    return of, df, gf
