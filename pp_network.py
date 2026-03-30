"""
PPO 网络模块
实现双智能体 PPO 网络：
- 订单选择智能体（OrderSelectionAgent）
- 设备分配智能体（MachineAssignAgent）
两个智能体均包含：
  策略网络：状态 -> 特征提取 -> 启发式特征融合(注意力) -> 动作输出
  价值网络：状态 -> 特征提取 -> 价值输出
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─── 特征维度常量 ──────────────────────────────────────────────────────────────
# 订单特征：type_onehot(15) + qty(1) + mode(1) + deadline(1) + stage_status(7) + min_remaining(7)
ORDER_FEAT_DIM = 15 + 3 + 7 + 7   # = 32
# 设备特征：stage_onehot(7) + n_compatible(1) + idle_ratio(1) + is_busy(1) + last_type(15)
MACHINE_FEAT_DIM = 7 + 3 + 15      # = 25
# 全局特征
GLOBAL_FEAT_DIM = 3
# 启发式得分维度
ORDER_HEURISTIC_DIM = 2    # [type_score, delay_score]
MACHINE_HEURISTIC_DIM = 4  # [specif, util, setup, proc_time]


class HeuristicAttentionFusion(nn.Module):
    """
    注意力机制融合层：
    给启发式规则得分向量赋予动态权重，
    然后与特征提取层输出拼接。
    """
    def __init__(self, feat_dim, heuristic_dim, hidden_dim=32):
        super().__init__()
        # 注意力权重网络（输入为特征表征向量，输出为对各启发式规则的权重）
        self.attn_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, heuristic_dim),
        )

    def forward(self, feat_repr, heuristic_scores):
        """
        feat_repr:       [batch, feat_dim]
        heuristic_scores:[batch, heuristic_dim]
        返回:            [batch, feat_dim + heuristic_dim]
        """
        # 用 softmax 生成注意力权重
        attn_weights = torch.softmax(self.attn_net(feat_repr), dim=-1)  # [batch, H]
        # 加权启发式得分
        weighted_heuristic = attn_weights * heuristic_scores             # [batch, H]
        # 拼接融合
        fused = torch.cat([feat_repr, weighted_heuristic], dim=-1)       # [batch, F+H]
        return fused


class FeatureExtractor(nn.Module):
    """
    通用特征提取层：
    将订单特征、设备特征、全局特征编码为统一表征向量。
    """
    def __init__(self, order_feat_dim, machine_feat_dim, global_feat_dim,
                 hidden_dim=128, repr_dim=64):
        super().__init__()
        self.order_encoder = nn.Sequential(
            nn.Linear(order_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, repr_dim),
            nn.ReLU(),
        )
        self.machine_encoder = nn.Sequential(
            nn.Linear(machine_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, repr_dim),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feat_dim, 16),
            nn.ReLU(),
        )
        self.output_dim = repr_dim * 2 + 16

    def forward(self, order_feat, machine_feat, global_feat):
        """
        order_feat:   [batch, order_feat_dim]   — 单个订单的特征
        machine_feat: [batch, machine_feat_dim]  — 当前可用设备平均特征（或全局聚合）
        global_feat:  [batch, global_feat_dim]
        """
        o = self.order_encoder(order_feat)
        m = self.machine_encoder(machine_feat)
        g = self.global_encoder(global_feat)
        return torch.cat([o, m, g], dim=-1)   # [batch, repr_dim*2+16]


# ─── 订单选择智能体 ──────────────────────────────────────────────────────────────
class OrderSelectionPPO(nn.Module):
    """
    订单选择智能体的 PPO 网络。
    给定"当前可调度订单集合"，输出每个订单的选择概率及状态价值。
    """
    def __init__(self, hidden_dim=128, repr_dim=64):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            ORDER_FEAT_DIM, MACHINE_FEAT_DIM, GLOBAL_FEAT_DIM,
            hidden_dim=hidden_dim, repr_dim=repr_dim
        )
        feat_out_dim = self.feature_extractor.output_dim  # repr_dim*2+16

        # 注意力融合层（策略网络专用）
        self.heuristic_fusion = HeuristicAttentionFusion(feat_out_dim, ORDER_HEURISTIC_DIM)
        fused_dim = feat_out_dim + ORDER_HEURISTIC_DIM

        # 策略网络动作输出（评分，后接 softmax）
        self.policy_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),   # 每个订单输出一个标量分
        )

        # 价值网络（单独的特征提取路径）
        self.value_extractor = FeatureExtractor(
            ORDER_FEAT_DIM, MACHINE_FEAT_DIM, GLOBAL_FEAT_DIM,
            hidden_dim=hidden_dim, repr_dim=repr_dim
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.value_extractor.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def get_action_probs(self, order_feats, machine_mean_feat, global_feat, heuristic_scores):
        """
        order_feats:        [n_candidates, ORDER_FEAT_DIM]
        machine_mean_feat:  [ORDER_FEAT_DIM or MACHINE_FEAT_DIM] — 广播
        global_feat:        [GLOBAL_FEAT_DIM]
        heuristic_scores:   [n_candidates, ORDER_HEURISTIC_DIM]
        返回: action_probs  [n_candidates]
        """
        n = order_feats.shape[0]
        machine_mean_expanded = machine_mean_feat.unsqueeze(0).expand(n, -1)
        global_expanded = global_feat.unsqueeze(0).expand(n, -1)

        feat_repr = self.feature_extractor(order_feats, machine_mean_expanded, global_expanded)
        fused = self.heuristic_fusion(feat_repr, heuristic_scores)
        logits = self.policy_head(fused).squeeze(-1)        # [n_candidates]
        probs = torch.softmax(logits, dim=-1)
        return probs

    def get_value(self, order_feats, machine_mean_feat, global_feat):
        """返回全局状态价值（对所有候选订单特征取均值）"""
        machine_mean_expanded = machine_mean_feat.unsqueeze(0).expand(order_feats.shape[0], -1)
        global_expanded = global_feat.unsqueeze(0).expand(order_feats.shape[0], -1)
        feat_repr = self.value_extractor(order_feats, machine_mean_expanded, global_expanded)
        value = self.value_head(feat_repr).mean()           # 聚合为单个标量
        return value


# ─── 设备分配智能体 ──────────────────────────────────────────────────────────────
class MachineAssignPPO(nn.Module):
    """
    设备分配智能体的 PPO 网络。
    给定"已选订单"和"当前可用设备集合"，输出每台设备的选择概率及价值。
    """
    def __init__(self, hidden_dim=128, repr_dim=64):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            ORDER_FEAT_DIM, MACHINE_FEAT_DIM, GLOBAL_FEAT_DIM,
            hidden_dim=hidden_dim, repr_dim=repr_dim
        )
        feat_out_dim = self.feature_extractor.output_dim

        self.heuristic_fusion = HeuristicAttentionFusion(feat_out_dim, MACHINE_HEURISTIC_DIM)
        fused_dim = feat_out_dim + MACHINE_HEURISTIC_DIM

        self.policy_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.value_extractor = FeatureExtractor(
            ORDER_FEAT_DIM, MACHINE_FEAT_DIM, GLOBAL_FEAT_DIM,
            hidden_dim=hidden_dim, repr_dim=repr_dim
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.value_extractor.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def get_action_probs(self, order_feat, machine_feats, global_feat, heuristic_scores):
        """
        order_feat:       [ORDER_FEAT_DIM]          — 单个选定订单
        machine_feats:    [n_machines, MACHINE_FEAT_DIM]
        global_feat:      [GLOBAL_FEAT_DIM]
        heuristic_scores: [n_machines, MACHINE_HEURISTIC_DIM]
        返回: probs        [n_machines]
        """
        n = machine_feats.shape[0]
        order_expanded = order_feat.unsqueeze(0).expand(n, -1)
        global_expanded = global_feat.unsqueeze(0).expand(n, -1)

        feat_repr = self.feature_extractor(order_expanded, machine_feats, global_expanded)
        fused = self.heuristic_fusion(feat_repr, heuristic_scores)
        logits = self.policy_head(fused).squeeze(-1)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def get_value(self, order_feat, machine_feats, global_feat):
        n = machine_feats.shape[0]
        order_expanded = order_feat.unsqueeze(0).expand(n, -1)
        global_expanded = global_feat.unsqueeze(0).expand(n, -1)
        feat_repr = self.value_extractor(order_expanded, machine_feats, global_expanded)
        value = self.value_head(feat_repr).mean()
        return value


# ─── 工具函数 ──────────────────────────────────────────────────────────────────
def state_to_tensors(state, device):
    """将环境状态字典转为 GPU/CPU tensor"""
    return {
        'order_features': torch.FloatTensor(state['order_features']).to(device),
        'machine_features': torch.FloatTensor(state['machine_features']).to(device),
        'global_features': torch.FloatTensor(state['global_features']).to(device),
    }


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    order_agent = OrderSelectionPPO().to(device)
    machine_agent = MachineAssignPPO().to(device)

    # 测试前向传播
    n_cand_orders = 5
    n_cand_machines = 3
    order_feats = torch.randn(n_cand_orders, ORDER_FEAT_DIM).to(device)
    machine_feats = torch.randn(n_cand_machines, MACHINE_FEAT_DIM).to(device)
    machine_mean = machine_feats.mean(0)
    global_feat = torch.randn(GLOBAL_FEAT_DIM).to(device)
    h_order = torch.rand(n_cand_orders, ORDER_HEURISTIC_DIM).to(device)
    h_machine = torch.rand(n_cand_machines, MACHINE_HEURISTIC_DIM).to(device)

    probs_o = order_agent.get_action_probs(order_feats, machine_mean, global_feat, h_order)
    value_o = order_agent.get_value(order_feats, machine_mean, global_feat)
    print(f"订单选择概率: {probs_o.detach().cpu().numpy().round(3)}")
    print(f"订单价值: {value_o.item():.4f}")

    order_feat_single = order_feats[0]
    probs_m = machine_agent.get_action_probs(order_feat_single, machine_feats, global_feat, h_machine)
    value_m = machine_agent.get_value(order_feat_single, machine_feats, global_feat)
    print(f"设备分配概率: {probs_m.detach().cpu().numpy().round(3)}")
    print(f"设备价值: {value_m.item():.4f}")

    total_params = sum(p.numel() for p in order_agent.parameters()) + \
                   sum(p.numel() for p in machine_agent.parameters())
    print(f"网络参数总量: {total_params:,}")
    print("网络模块测试成功 ✓")
