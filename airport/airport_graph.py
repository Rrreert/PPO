"""
airport_graph.py
构建虹桥机场滑行道拓扑图（基于 connect.py 逻辑）
并提供节点坐标、边权重（欧氏距离）等工具函数。
"""
import pandas as pd
import networkx as nx
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 禁止途经节点（不能穿越跑道）
# ─────────────────────────────────────────────
FORBIDDEN_NODES = {
    'R18R_H5', 'R18R_C3', 'R18R_H4', 'R18R_B7', 'R18R_H3',
    'R18L_B6', 'R18L_A4', 'R18L_A3', 'R18L_B4', 'R18L_H4',
    'R18L_A2', 'R18L_B1'
}

def build_graph(csv_path="虹桥点.csv") -> tuple[nx.Graph, dict]:
    """返回 (G, pos)，G 的边权重为欧氏距离（米）"""
    df = pd.read_csv(csv_path)
    nodes = df['name'].tolist()
    pos = {row['name']: (row['Column1'], row['Column2']) for _, row in df.iterrows()}

    G = nx.Graph()
    G.add_nodes_from(nodes)

    # ── 辅助函数 ──────────────────────────────
    def sort_by_distance(point_list, positions):
        pts = [(p, positions[p]) for p in point_list if p in positions]
        if len(pts) <= 1:
            return [p[0] for p in pts]
        sorted_pts, remaining = [], [(p[0], np.array(p[1])) for p in pts]
        current = remaining.pop(0)
        sorted_pts.append(current[0])
        while remaining:
            dists = [np.linalg.norm(current[1] - r[1]) for r in remaining]
            idx = int(np.argmin(dists))
            current = remaining.pop(idx)
            sorted_pts.append(current[0])
        return sorted_pts

    def find_nearest(src, candidates, positions):
        if src not in positions:
            return None
        p0 = np.array(positions[src])
        best, bd = None, float('inf')
        for c in candidates:
            if c in positions:
                d = np.linalg.norm(p0 - np.array(positions[c]))
                if d < bd:
                    bd, best = d, c
        return best

    def add_weighted_edge(G, u, v, pos):
        if u in pos and v in pos:
            w = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
            G.add_edge(u, v, weight=w)

    def connect_seq(pts, name=""):
        for i in range(len(pts) - 1):
            add_weighted_edge(G, pts[i], pts[i+1], pos)

    # ── 1. 各滑行道分组顺序连接 ──────────────
    l01 = sorted([f'S{i}' for i in range(501, 526) if f'S{i}' in G] +
                 [f'S{i}' for i in range(112, 128) if f'S{i}' in G] +
                 [f'S{i}' for i in range(313, 316) if f'S{i}' in G],
                 key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    l14 = sorted([f'S{i}' for i in range(286, 291) if f'S{i}' in G] +
                 [f'S{i}' for i in range(601, 609) if f'S{i}' in G],
                 key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    d_stand = sorted([f'S{i}' for i in range(401, 417) if f'S{i}' in G],
                     key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)

    for grp_name, grp_fn in [
        ("m1", lambda: sorted([f'S{i}' for i in range(212, 218) if f'S{i}' in G], key=lambda x: int(x[1:]))),
        ("m2", lambda: sorted([f'S{i}' for i in range(227, 232) if f'S{i}' in G], key=lambda x: int(x[1:]))),
        ("m5", lambda: sorted([f'S{i}' for i in range(266, 272) if f'S{i}' in G], key=lambda x: int(x[1:]))),
        ("m6", lambda: sorted([f'S{i}' for i in range(280, 286) if f'S{i}' in G], key=lambda x: int(x[1:]))),
    ]:
        connect_seq(grp_fn())

    m3 = sort_by_distance([p for p in ['S236','S237','S238E'] if p in G], pos)
    m4 = sort_by_distance([p for p in ['S259E','S260','S261'] if p in G], pos)
    for grp in [l01, l14, d_stand, m3, m4]:
        connect_seq(grp)

    for prefix in ['A', 'B', 'C', 'D', 'R18L', 'R18R']:
        pts = sort_by_distance([n for n in nodes if n.startswith(prefix)], pos)
        connect_seq(pts)

    # ── 2. 带 -数字 后缀的联络道内部连接 ──────
    dash_groups = {}
    for node in nodes:
        if '_' in node:
            parts = node.split('_')
            if len(parts) >= 2 and '-' in parts[1]:
                base = parts[1].split('-')[0]
                dash_groups.setdefault(base, []).append(node)
    for pts in dash_groups.values():
        if len(pts) > 1:
            for a, b in zip(sort_by_distance(pts, pos), sort_by_distance(pts, pos)[1:]):
                add_weighted_edge(G, a, b, pos)

    # ── 3. 点对点连接 ─────────────────────────
    explicit_edges = [
        ('C_D16-2','C_C4-2'),('C_C4-1','R18R_H5'),('C_C1-1','R18R_H3'),
        ('B_B4-1','B_H5'),('B_B5-1','B_B5-3'),('B_H6','B_B6-2'),
        ('R18L_H7','H7_1'),('R18L_T6','H7_1'),
        ('B_B5-2','R18L_A4'),('R18L_A4','A_K7'),('R18L_A3','A_K6'),
        ('A_A2-2','R18L_A2'),('S525','A_H7'),('S518','A_K7'),
        ('S508','A_K6'),('S316','L01_L10'),('S328','L01_L20'),
        ('S212','D_D6'),('Y1_S231','D_D7'),('S213','D_D8'),
        ('S236','D_D10'),('S261','D_D12'),('S246','S238E'),
        ('S250','S259E'),('S263','C_D13'),('S266','D_D14'),
        ('Y1_S285','D_D15'),('S285','D_H5'),('B_H1','R18L_H1'),
        ('R18R_H5','B_B5-3'),('B_B7','B_H3'),('B_B2-3','B_B2-1'),
        ('B_B2-3','R18L_A1'),('R18R_H4','B_H4-4'),
    ]
    for u, v in explicit_edges:
        if u in G and v in G:
            add_weighted_edge(G, u, v, pos)

    # 三角形
    for tri in [['C_H3','C_C2','R18R_B7'],['B_B8','B_H5','R18R_C3']]:
        for i in range(len(tri)):
            for j in range(i+1, len(tri)):
                if tri[i] in G and tri[j] in G:
                    add_weighted_edge(G, tri[i], tri[j], pos)

    # 链
    for chain in [['B_B2-2','B_H3','B_B3-1','B_H4-3']]:
        connect_seq(chain)

    # 顺次范围
    for s, e in [('S112','S102'),('S338','S342'),('S328','S332'),('S316','S321')]:
        sn, en = int(s[1:]), int(e[1:])
        rng = range(sn, en+1) if sn < en else range(sn, en-1, -1)
        pts = [f'S{n}' for n in rng if f'S{n}' in G]
        connect_seq(pts)

    # ── 4. 最近点连接（Y1/Y3/D滑行道） ────────
    y1 = [n for n in nodes if n.startswith('Y1_')]
    y3 = [n for n in nodes if n.startswith('Y3_')]
    d_tx = [n for n in nodes if n.startswith('D_')]

    for i in range(212, 218):
        p = f'S{i}'
        if p in G:
            nn = find_nearest(p, y1, pos)
            if nn: add_weighted_edge(G, p, nn, pos)
    for i in range(227, 232):
        p = f'S{i}'
        if p in G:
            nn = find_nearest(p, y1, pos)
            if nn: add_weighted_edge(G, p, nn, pos)
    for i in range(266, 286):
        p = f'S{i}'
        if p in G:
            nn = find_nearest(p, y3, pos)
            if nn: add_weighted_edge(G, p, nn, pos)
    for i in list(range(286, 291)) + list(range(601, 609)):
        p = f'S{i}'
        if p in G:
            nn = find_nearest(p, d_tx, pos)
            if nn: add_weighted_edge(G, p, nn, pos)

    # ── 5. 交叉口：前缀分组连接 ───────────────
    inter_groups = {}
    for node in nodes:
        if '_' in node:
            parts = node.split('_')
            if len(parts) >= 2:
                mt = parts[0]
                lb = parts[1].split('-')[0] if '-' in parts[1] else parts[1]
                inter_groups.setdefault(mt, []).append(node)
                inter_groups.setdefault(lb, []).append(node)

    suffix_1 = {n for n in nodes if n.endswith('_1')}
    for pts in inter_groups.values():
        if len(pts) > 1:
            sp = sort_by_distance(pts, pos)
            for a, b in zip(sp, sp[1:]):
                if not (a in suffix_1 and b in suffix_1):
                    add_weighted_edge(G, a, b, pos)

    # ── 6. 强制全连通 ─────────────────────────
    iters = 0
    while not nx.is_connected(G) and iters < 50:
        iters += 1
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        main = list(comps[0])
        for comp in comps[1:]:
            best, bd = (None, None), float('inf')
            for cn in comp:
                p1 = np.array(pos[cn])
                for mn in main:
                    d = np.linalg.norm(p1 - np.array(pos[mn]))
                    if d < bd:
                        bd, best = d, (cn, mn)
            if best[0]:
                add_weighted_edge(G, best[0], best[1], pos)
                main.append(best[0])
        if nx.is_connected(G):
            break

    # ── 7. 断开操作 ───────────────────────────
    disconnections = [
        ('R18L_H7','A_H7'),('B_B7','B_H4-3'),('R18L_A2','A_A2-3'),
        ('B_H1','A_H1'),('B_H3','B_B2-3'),('C_H3','R18R_B7'),
        ('C_C2','C_D6'),('C_C1-1','C_H3'),('R18R_H4','B_H4-3'),
        ('B_B7','B_H4-3'),('B_H3','B_B2-3'),('C_H3','R18R_B7'),
        ('C_C2','C_D6'),('C_C1-1','C_H3'),('R18L_H7','A_H7'),
        ('B_H1','A_H1'),
    ]
    for u, v in disconnections:
        if G.has_edge(u, v):
            G.remove_edge(u, v)

    print(f"[Graph] nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, "
          f"connected={nx.is_connected(G)}")
    return G, pos


def get_restricted_graph(G: nx.Graph) -> nx.Graph:
    """返回去掉禁止节点的图（用于路径规划）"""
    Gr = G.copy()
    for n in FORBIDDEN_NODES:
        if n in Gr:
            Gr.remove_node(n)
    return Gr


def euclidean(pos, u, v):
    return np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))


if __name__ == "__main__":
    import os
    os.chdir("/home/claude")
    G, pos = build_graph("虹桥点.csv")
    print("Sample edges:", list(G.edges(data=True))[:3])
