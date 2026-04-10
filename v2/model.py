"""异构图神经网络 + PPO 模型（优化版）"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import numpy as np

ORDER_FEAT_DIM = 5
DEVICE_FEAT_DIM = 6
TYPE_FEAT_DIM = 1
HIDDEN_DIM = 48


class LightHeteroGNN(nn.Module):
    """轻量级异构图网络: 单层消息传递，手写SAGE聚合"""
    def __init__(self, hidden=HIDDEN_DIM):
        super().__init__()
        self.h = hidden
        # 输入投影
        self.proj_order = nn.Linear(ORDER_FEAT_DIM, hidden)
        self.proj_device = nn.Linear(DEVICE_FEAT_DIM, hidden)
        self.proj_type = nn.Linear(TYPE_FEAT_DIM, hidden)
        # 消息传递更新 (concat self + agg_neighbor)
        self.upd_order = nn.Linear(hidden * 2, hidden)
        self.upd_device = nn.Linear(hidden * 3, hidden)  # 3 neighbor types
        self.norm_order = nn.LayerNorm(hidden)
        self.norm_device = nn.LayerNorm(hidden)

    def _sage_agg(self, src_feat, edge_index, n_dst):
        """简单均值聚合"""
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

        ho = F.elu(self.proj_order(data['order'].x))
        hd = F.elu(self.proj_device(data['device'].x))
        ht = F.elu(self.proj_type(data['type'].x))

        ei = data.edge_index_dict
        # order <- device (assigned_from)
        agg_o_from_d = self._sage_agg(hd, ei.get(('device','assigned_from','order'),
                                       torch.zeros(2,0,dtype=torch.long)), n_order)
        # order <- type (type_of)
        agg_o_from_t = self._sage_agg(ht, ei.get(('type','type_of','order'),
                                       torch.zeros(2,0,dtype=torch.long)), n_order)
        # device <- order (assigned_to)
        agg_d_from_o = self._sage_agg(ho, ei.get(('order','assigned_to','device'),
                                       torch.zeros(2,0,dtype=torch.long)), n_device)
        # device <- device (next_op)
        agg_d_from_d = self._sage_agg(hd, ei.get(('device','next_op','device'),
                                       torch.zeros(2,0,dtype=torch.long)), n_device)
        # device <- type (processed_by)
        agg_d_from_t = self._sage_agg(ht, ei.get(('type','processed_by','device'),
                                       torch.zeros(2,0,dtype=torch.long)), n_device)

        ho_new = self.norm_order(F.elu(self.upd_order(
            torch.cat([ho, agg_o_from_d + agg_o_from_t], dim=-1))))
        hd_new = self.norm_device(F.elu(self.upd_device(
            torch.cat([hd, agg_d_from_o, agg_d_from_d + agg_d_from_t], dim=-1))))

        return ho_new, hd_new


class SchedulingAgent(nn.Module):
    def __init__(self, hidden=HIDDEN_DIM):
        super().__init__()
        self.gnn = LightHeteroGNN(hidden)
        self.actor = nn.Sequential(
            nn.Linear(hidden * 2 + 2, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden + 2, hidden // 2),
            nn.ELU(),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, graph, pairs, global_feat):
        ho, hd = self.gnn(graph)
        if len(pairs) == 0:
            return torch.tensor([]), torch.tensor(0.0)

        oi = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        di = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        g = global_feat.unsqueeze(0).expand(len(pairs), -1)
        pair_feat = torch.cat([ho[oi], hd[di], g], dim=-1)
        logits = self.actor(pair_feat).squeeze(-1)

        graph_summary = torch.cat([(ho.mean(0) + hd.mean(0)) / 2, global_feat])
        value = self.critic(graph_summary).squeeze()
        return logits, value


# ---- 图构建 ----
def build_hetero_graph(env, data):
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

    # 订单特征
    if orders:
        order_feats = []
        for o in orders:
            ready_ops = o.get_ready_ops()
            mr = min((data['min_remaining'].get(o.id, {}).get(op, 0)
                      for op in ready_ops), default=0)
            urgency = (cur_t + mr - o.due) / (max_due + 1e-6)
            order_feats.append([
                o.qty / 1280.0,
                o.priority,
                float(np.clip(urgency, -1, 1)),
                mr / (max_due + 1e-6),
                o.arrive / (max_due + 1e-6),
            ])
        order_x = torch.tensor(order_feats, dtype=torch.float)
    else:
        order_x = torch.zeros((0, ORDER_FEAT_DIM))

    device_feats = []
    for d in devices:
        device_feats.append([
            0.0,
            d.specialization(),
            d.idle_rate(cur_t),
            1.0 if d.status == 'idle' else 0.0,
            1.0 if d.status == 'busy' else 0.0,
            1.0 if d.status == 'broken' else 0.0,
        ])
    device_x = torch.tensor(device_feats, dtype=torch.float)
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
