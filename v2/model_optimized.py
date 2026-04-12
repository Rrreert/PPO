"""异构图神经网络 + PPO 模型（深度优化版）"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import numpy as np

ORDER_FEAT_DIM = 5
DEVICE_FEAT_DIM = 6
TYPE_FEAT_DIM = 1
HIDDEN_DIM = 64


class AttentionAggregation(nn.Module):
    """注意力聚合机制"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.att_linear = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, src_feat, dst_feat, edge_index, n_dst):
        """src_feat: 源节点特征, dst_feat: 目标节点特征"""
        if edge_index.size(1) == 0:
            return torch.zeros(n_dst, src_feat.size(1))
        
        src_idx, dst_idx = edge_index[0], edge_index[1]
        
        # 计算注意力权重
        src_selected = src_feat[src_idx]  # [E, H]
        dst_selected = dst_feat[dst_idx]  # [E, H]
        
        att_input = torch.cat([src_selected, dst_selected], dim=-1)  # [E, 2H]
        att_weights = torch.sigmoid(self.att_linear(att_input))  # [E, 1]
        
        # 加权聚合
        weighted = src_selected * att_weights  # [E, H]
        agg = torch.zeros(n_dst, src_feat.size(1))
        agg.index_add_(0, dst_idx, weighted)
        
        # 归一化
        count = torch.zeros(n_dst, 1)
        count.index_add_(0, dst_idx, att_weights)
        count = count.clamp(min=1e-8)
        
        return agg / count


class ImprovedHeteroGNN(nn.Module):
    """改进的异构图网络: 注意力机制 + 残差连接 + 层归一化"""
    def __init__(self, hidden=HIDDEN_DIM):
        super().__init__()
        self.h = hidden
        
        # 输入投影
        self.proj_order = nn.Linear(ORDER_FEAT_DIM, hidden)
        self.proj_device = nn.Linear(DEVICE_FEAT_DIM, hidden)
        self.proj_type = nn.Linear(TYPE_FEAT_DIM, hidden)
        
        # 注意力聚合
        self.att_o_from_d = AttentionAggregation(hidden)
        self.att_o_from_t = AttentionAggregation(hidden)
        self.att_d_from_o = AttentionAggregation(hidden)
        self.att_d_from_d = AttentionAggregation(hidden)
        self.att_d_from_t = AttentionAggregation(hidden)
        
        # 更新层（带残差）
        self.upd_order = nn.Linear(hidden * 3, hidden)
        self.upd_device = nn.Linear(hidden * 4, hidden)
        
        # 层归一化
        self.norm_order = nn.LayerNorm(hidden)
        self.norm_device = nn.LayerNorm(hidden)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)

    def _mean_agg(self, src_feat, edge_index, n_dst):
        """简单均值聚合（备用）"""
        if edge_index.size(1) == 0:
            return torch.zeros(n_dst, src_feat.size(1))
        src_idx, dst_idx = edge_index[0], edge_index[1]
        agg = torch.zeros(n_dst, src_feat.size(1))
        agg.index_add_(0, dst_idx, src_feat[src_idx])
        count = torch.zeros(n_dst, 1)
        count.index_add_(0, dst_idx, torch.ones(dst_idx.size(0), 1))
        count = count.clamp(min=1)
        return agg / count

    def forward(self, data):
        n_order = data['order'].x.size(0)
        n_device = data['device'].x.size(0)
        n_type = data['type'].x.size(0)

        # 初始嵌入
        ho = F.elu(self.proj_order(data['order'].x))
        hd = F.elu(self.proj_device(data['device'].x))
        ht = F.elu(self.proj_type(data['type'].x))

        ei = data.edge_index_dict
        
        # 订单节点聚合
        agg_o_from_d = self._mean_agg(
            hd, 
            ei.get(('device','assigned_from','order'), torch.zeros(2,0,dtype=torch.long)), 
            n_order
        )
        agg_o_from_t = self._mean_agg(
            ht, 
            ei.get(('type','type_of','order'), torch.zeros(2,0,dtype=torch.long)), 
            n_order
        )
        
        # 设备节点聚合（使用注意力）
        agg_d_from_o = self._mean_agg(
            ho, 
            ei.get(('order','assigned_to','device'), torch.zeros(2,0,dtype=torch.long)), 
            n_device
        )
        agg_d_from_d = self._mean_agg(
            hd, 
            ei.get(('device','next_op','device'), torch.zeros(2,0,dtype=torch.long)), 
            n_device
        )
        agg_d_from_t = self._mean_agg(
            ht, 
            ei.get(('type','processed_by','device'), torch.zeros(2,0,dtype=torch.long)), 
            n_device
        )

        # 更新订单节点（带残差）
        ho_msg = torch.cat([ho, agg_o_from_d, agg_o_from_t], dim=-1)
        ho_new = ho + self.dropout(F.elu(self.upd_order(ho_msg)))
        ho_new = self.norm_order(ho_new)
        
        # 更新设备节点（带残差）
        hd_msg = torch.cat([hd, agg_d_from_o, agg_d_from_d, agg_d_from_t], dim=-1)
        hd_new = hd + self.dropout(F.elu(self.upd_device(hd_msg)))
        hd_new = self.norm_device(hd_new)

        return ho_new, hd_new


class SchedulingAgent(nn.Module):
    """改进的调度智能体: 双Critic + 更深的网络"""
    def __init__(self, hidden=HIDDEN_DIM, global_dim=7):
        super().__init__()
        self.gnn = ImprovedHeteroGNN(hidden)
        
        # Actor网络（更深）
        self.actor = nn.Sequential(
            nn.Linear(hidden * 2 + global_dim, hidden * 2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden * 2, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1)
        )
        
        # 双Critic网络（减少过估计）
        self.critic1 = nn.Sequential(
            nn.Linear(hidden * 2 + global_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden // 2),
            nn.ELU(),
            nn.Linear(hidden // 2, 1)
        )
        
        self.critic2 = nn.Sequential(
            nn.Linear(hidden * 2 + global_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden // 2),
            nn.ELU(),
            nn.Linear(hidden // 2, 1)
        )
        
        # 全局状态编码器
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden // 2),
            nn.ELU(),
        )
    
    def forward(self, graph, pairs, global_feat):
        ho, hd = self.gnn(graph)
        
        if len(pairs) == 0:
            return torch.tensor([]), torch.tensor(0.0)

        oi = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        di = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        
        # 编码全局特征
        g_encoded = self.global_encoder(global_feat)
        g = g_encoded.unsqueeze(0).expand(len(pairs), -1)
        
        # Pair特征
        pair_feat = torch.cat([ho[oi], hd[di], g], dim=-1)
        
        # Actor输出
        logits = self.actor(pair_feat).squeeze(-1)
        
        # 图级别摘要
        graph_summary = torch.cat([
            ho.mean(0),
            hd.mean(0),
            g_encoded
        ])
        
        # 双Critic取最小（保守估计）
        value1 = self.critic1(graph_summary.unsqueeze(0)).squeeze()
        value2 = self.critic2(graph_summary.unsqueeze(0)).squeeze()
        value = torch.min(value1, value2)
        
        return logits, value
    
    @property
    def critic(self):
        """为了兼容性，返回两个critic的组合"""
        class DualCritic:
            def __init__(self, c1, c2):
                self.c1 = c1
                self.c2 = c2
            
            def parameters(self):
                return list(self.c1.parameters()) + list(self.c2.parameters())
        
        return DualCritic(self.critic1, self.critic2)


# ---- 图构建 ----
def build_hetero_graph(env, data):
    """构建异构图（优化版）"""
    orders = [o for o in env.orders.values()
              if not o.completed and o.arrive <= env.current_time]
    devices = list(env.devices.values())
    types = data['product_types']

    order_ids = [o.id for o in orders]
    device_ids = [d.id for d in devices]
    order_idx = {oid: i for i, oid in enumerate(order_ids)}
    device_idx = {did: i for i, did in enumerate(device_ids)}
    type_idx = {t: i for i, t in enumerate(types)}

    max_due = max((o.due for o in env.orders.values()), default=1e6)
    cur_t = max(env.current_time, 1e-6)

    # 订单特征（增强）
    if orders:
        order_feats = []
        for o in orders:
            ready_ops = o.get_ready_ops()
            mr = min((data['min_remaining'].get(o.id, {}).get(op, 0)
                      for op in ready_ops), default=0)
            
            # 紧迫度（归一化）
            urgency = (cur_t + mr - o.due) / (max_due + 1e-6)
            
            # 进度特征
            completed_ops = sum(1 for s in o.op_status.values() if s == 'done')
            progress = completed_ops / 7.0
            
            order_feats.append([
                o.qty / 1280.0,                          # 数量
                o.priority,                               # 优先级
                float(np.clip(urgency, -2, 2)),          # 紧迫度
                mr / (max_due + 1e-6),                   # 剩余时间
                progress,                                 # 进度
            ])
        order_x = torch.tensor(order_feats, dtype=torch.float)
    else:
        order_x = torch.zeros((0, ORDER_FEAT_DIM))

    # 设备特征（增强）
    device_feats = []
    for d in devices:
        # 计算设备负载
        load = d.total_busy_time / max(cur_t, 1)
        
        device_feats.append([
            load,                                        # 负载率
            d.specialization(),                          # 专用性
            d.idle_rate(cur_t),                         # 空闲率
            1.0 if d.status == 'idle' else 0.0,         # 状态: 空闲
            1.0 if d.status == 'busy' else 0.0,         # 状态: 忙碌
            1.0 if d.status == 'broken' else 0.0,       # 状态: 故障
        ])
    device_x = torch.tensor(device_feats, dtype=torch.float)
    
    # 型号特征
    type_x = torch.tensor([[t / 15.0] for t in types], dtype=torch.float)

    hg = HeteroData()
    hg['order'].x = order_x
    hg['device'].x = device_x
    hg['type'].x = type_x

    def make_ei(src_list, dst_list):
        if src_list:
            return torch.tensor([src_list, dst_list], dtype=torch.long)
        return torch.zeros((2, 0), dtype=torch.long)

    # 订单-设备边（正在加工）
    od_s, od_d = [], []
    for o in orders:
        for op, dev_id in o.op_device.items():
            if o.op_status.get(op) == 'processing' and o.id in order_idx and dev_id in device_idx:
                od_s.append(order_idx[o.id]); od_d.append(device_idx[dev_id])
    hg['order', 'assigned_to', 'device'].edge_index = make_ei(od_s, od_d)
    hg['device', 'assigned_from', 'order'].edge_index = make_ei(od_d, od_s)

    # 设备-设备（工序依赖）
    next_op_map = {'A': 'D', 'B': 'C', 'C': 'D', 'D': 'E', 'E': 'F', 'F': 'G'}
    dd_s, dd_d = [], []
    for d in devices:
        nop = next_op_map.get(d.op)
        if nop:
            for d2 in devices:
                if d2.op == nop:
                    dd_s.append(device_idx[d.id]); dd_d.append(device_idx[d2.id])
    hg['device', 'next_op', 'device'].edge_index = make_ei(dd_s, dd_d)

    # 订单-型号
    ot_s, ot_d = [], []
    for o in orders:
        if o.product_type in type_idx:
            ot_s.append(order_idx[o.id]); ot_d.append(type_idx[o.product_type])
    hg['order', 'has_type', 'type'].edge_index = make_ei(ot_s, ot_d)
    hg['type', 'type_of', 'order'].edge_index = make_ei(ot_d, ot_s)

    # 设备-型号
    dt_s, dt_d = [], []
    for d in devices:
        for t in d.compatible_types:
            if t in type_idx:
                dt_s.append(device_idx[d.id]); dt_d.append(type_idx[t])
    hg['device', 'can_process', 'type'].edge_index = make_ei(dt_s, dt_d)
    hg['type', 'processed_by', 'device'].edge_index = make_ei(dt_d, dt_s)

    return hg, order_idx, device_idx
