"""
visualize.py
绘制：
1. 机场拓扑图（底图）
2. 所有航班滑行路线图（PPO / Dijkstra）
3. PPO vs Dijkstra 4 项指标对比柱状图
"""
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

try:
    plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
except:
    pass
plt.rcParams['axes.unicode_minus'] = False


# ── 颜色方案 ──────────────────────────────────
CMAP_DEP = cm.get_cmap('Blues')
CMAP_ARR = cm.get_cmap('Oranges')
COLORS = {
    'PPO':      '#2196F3',
    'Dijkstra': '#FF5722',
    'node':     '#90CAF9',
    'edge':     '#B0BEC5',
    'stand':    '#A5D6A7',
    'forbidden':'#EF9A9A',
}


def draw_airport_base(G: nx.Graph, pos: dict, ax, title="Airport Topology",
                      highlight_forbidden=True):
    """绘制机场底图"""
    ax.set_facecolor('#F5F5F5')

    # 边
    edge_list = list(G.edges())
    xs, ys = [], []
    for u, v in edge_list:
        if u in pos and v in pos:
            xu, yu = pos[u]
            xv, yv = pos[v]
            ax.plot([xu, xv], [yu, yv], '-', color=COLORS['edge'],
                    linewidth=0.6, alpha=0.5, zorder=1)

    # 普通节点
    from airport_graph import FORBIDDEN_NODES
    normal_nodes = [n for n in G.nodes() if n not in FORBIDDEN_NODES]
    stand_nodes = [n for n in normal_nodes if n.startswith('S')]
    other_nodes = [n for n in normal_nodes if not n.startswith('S')]
    forbidden_nodes = [n for n in G.nodes() if n in FORBIDDEN_NODES and n in pos]

    for nlist, color, size, marker in [
        (stand_nodes, COLORS['stand'], 25, 'o'),
        (other_nodes, COLORS['node'], 15, 's'),
        (forbidden_nodes, COLORS['forbidden'], 30, 'x'),
    ]:
        if nlist:
            xs = [pos[n][0] for n in nlist if n in pos]
            ys = [pos[n][1] for n in nlist if n in pos]
            ax.scatter(xs, ys, c=color, s=size, marker=marker,
                       zorder=3, alpha=0.8)

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)
    ax.set_xlabel('X (m)', fontsize=8)
    ax.set_ylabel('Y (m)', fontsize=8)
    ax.tick_params(labelsize=7)


def draw_taxi_routes(results: list, G: nx.Graph, pos: dict,
                     ax, title: str, algorithm: str = 'PPO',
                     max_show: int = 60):
    """在底图上绘制滑行路线"""
    draw_airport_base(G, pos, ax, title=title)

    n = min(len(results), max_show)
    cmap_dep = cm.get_cmap('winter', n + 1)
    cmap_arr = cm.get_cmap('autumn', n + 1)

    dep_count, arr_count = 0, 0
    for i, res in enumerate(results[:max_show]):
        path = res['path']
        if len(path) < 2:
            continue
        xs = [pos[node][0] for node in path if node in pos]
        ys = [pos[node][1] for node in path if node in pos]
        if len(xs) < 2:
            continue

        is_dep = res['type'] == 'dep'
        if is_dep:
            color = cmap_dep(dep_count / max(n, 1))
            dep_count += 1
        else:
            color = cmap_arr(arr_count / max(n, 1))
            arr_count += 1

        ax.plot(xs, ys, '-', color=color, linewidth=1.0, alpha=0.6, zorder=5)
        # 起点标记
        ax.scatter([xs[0]], [ys[0]], c=[color], s=20, marker='^',
                   zorder=6, alpha=0.9)
        # 终点标记
        ax.scatter([xs[-1]], [ys[-1]], c=[color], s=20, marker='v',
                   zorder=6, alpha=0.9)

    legend_elements = [
        Line2D([0], [0], color='steelblue', lw=1.5, label='Departure'),
        Line2D([0], [0], color='darkorange', lw=1.5, label='Arrival'),
        Line2D([0], [0], marker='^', color='gray', lw=0,
               markersize=6, label='Start'),
        Line2D([0], [0], marker='v', color='gray', lw=0,
               markersize=6, label='End'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7,
              framealpha=0.7)


def draw_comparison(dijkstra_summary: dict, ppo_summary: dict,
                    ax_list, labels=None):
    """4 项指标对比柱状图"""
    metrics = [
        ('avg_time_s',        'Avg Taxiing Time (s)',     's'),
        ('avg_distance_m',    'Avg Taxiing Distance (m)', 'm'),
        ('avg_co2_kg',        'Avg CO₂ Emission (kg)',    'kg'),
        ('conflicts',         'Total Conflicts',          ''),
    ]

    bar_width = 0.35
    x = np.arange(2)

    for ax, (key, ylabel, unit) in zip(ax_list, metrics):
        dij_val = dijkstra_summary[key]
        ppo_val = ppo_summary[key]

        bars = ax.bar(['Dijkstra', 'PPO'],
                      [dij_val, ppo_val],
                      color=[COLORS['Dijkstra'], COLORS['PPO']],
                      width=0.5, edgecolor='white', linewidth=1.2,
                      alpha=0.85)

        # 数值标注
        for bar, val in zip(bars, [dij_val, ppo_val]):
            h = bar.get_height()
            fmt = f'{val:.1f}{unit}' if unit else f'{int(val)}'
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                    fmt, ha='center', va='bottom', fontsize=8,
                    fontweight='bold')

        # 改善百分比
        if dij_val > 0:
            pct = (dij_val - ppo_val) / dij_val * 100
            sign = '+' if pct < 0 else '-'
            color = 'green' if pct > 0 else 'red'
            ax.set_title(f'{ylabel}\nPPO vs Dijkstra: {sign}{abs(pct):.1f}%',
                         fontsize=8, fontweight='bold', color=color)
        else:
            ax.set_title(ylabel, fontsize=8, fontweight='bold')

        ax.set_ylabel(unit if unit else 'Count', fontsize=7)
        ax.tick_params(labelsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


def draw_learning_curve(rewards: list, ax):
    """PPO 学习曲线（episode 奖励）"""
    if not rewards:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        return
    ax.plot(rewards, color=COLORS['PPO'], linewidth=1.0, alpha=0.5)
    # 移动平均
    window = max(len(rewards) // 20, 10)
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), ma,
                color='navy', linewidth=2.0, label=f'MA-{window}')
    ax.set_title('PPO Learning Curve', fontsize=9, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=8)
    ax.set_ylabel('Episode Reward', fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=7)


def generate_full_report(dijkstra_summary: dict, ppo_summary: dict,
                         G: nx.Graph, pos: dict,
                         output_path: str = "taxi_comparison_report.png"):
    """
    生成完整报告图：
    Row 1: 机场拓扑  | Dijkstra 路线图
    Row 2: PPO 路线图 | 4个对比柱状图(2x2)
    """
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor('white')

    # 定义 GridSpec
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 4, figure=fig,
                  hspace=0.40, wspace=0.35,
                  left=0.05, right=0.97,
                  top=0.93, bottom=0.05)

    ax_topo  = fig.add_subplot(gs[0, :2])
    ax_dijk  = fig.add_subplot(gs[0, 2:])
    ax_ppo   = fig.add_subplot(gs[1, :2])
    ax_c1    = fig.add_subplot(gs[1, 2])
    ax_c2    = fig.add_subplot(gs[1, 3])
    ax_c3    = fig.add_subplot(gs[2, 2])
    ax_c4    = fig.add_subplot(gs[2, 3])
    ax_stats = fig.add_subplot(gs[2, :2])

    # ── 拓扑图 ──
    draw_airport_base(G, pos, ax_topo, "Hongqiao Airport Taxiway Topology")

    # ── Dijkstra 路线 ──
    draw_taxi_routes(dijkstra_summary['results'], G, pos, ax_dijk,
                     "Dijkstra Taxi Routes", 'Dijkstra')

    # ── PPO 路线 ──
    draw_taxi_routes(ppo_summary['results'], G, pos, ax_ppo,
                     "PPO Taxi Routes", 'PPO')

    # ── 对比柱状图 ──
    draw_comparison(dijkstra_summary, ppo_summary,
                    [ax_c1, ax_c2, ax_c3, ax_c4])

    # ── 统计文本表格 ──
    ax_stats.axis('off')
    table_data = [
        ['Metric', 'Dijkstra', 'PPO', 'Improvement'],
        ['Avg Time (s)',
         f"{dijkstra_summary['avg_time_s']:.1f}",
         f"{ppo_summary['avg_time_s']:.1f}",
         f"{(dijkstra_summary['avg_time_s']-ppo_summary['avg_time_s'])/dijkstra_summary['avg_time_s']*100:.1f}%"],
        ['Avg Distance (m)',
         f"{dijkstra_summary['avg_distance_m']:.1f}",
         f"{ppo_summary['avg_distance_m']:.1f}",
         f"{(dijkstra_summary['avg_distance_m']-ppo_summary['avg_distance_m'])/dijkstra_summary['avg_distance_m']*100:.1f}%"],
        ['Avg CO₂ (kg)',
         f"{dijkstra_summary['avg_co2_kg']:.2f}",
         f"{ppo_summary['avg_co2_kg']:.2f}",
         f"{(dijkstra_summary['avg_co2_kg']-ppo_summary['avg_co2_kg'])/dijkstra_summary['avg_co2_kg']*100:.1f}%"],
        ['Total Conflicts',
         f"{dijkstra_summary['conflicts']}",
         f"{ppo_summary['conflicts']}",
         f"{(dijkstra_summary['conflicts']-ppo_summary['conflicts'])/max(dijkstra_summary['conflicts'],1)*100:.1f}%"],
        ['Total Flights', '137', '137', '-'],
    ]

    tbl = ax_stats.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.5, 2.0)

    # 表格样式
    header_color = '#1565C0'
    row_colors = ['#E3F2FD', '#FFFFFF']
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor(row_colors[row % 2])
        cell.set_edgecolor('#BDBDBD')

    ax_stats.set_title('Summary Comparison Table', fontsize=10,
                        fontweight='bold', pad=8)

    # 总标题
    fig.suptitle('Hongqiao Airport Taxi Optimization: PPO vs Dijkstra\n'
                 'Date: 2024-04-07  |  Flights: 137 (Dep:88, Arr:49)',
                 fontsize=14, fontweight='bold', y=0.97)

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"[Viz] Report saved: {output_path}")
    return fig


def generate_route_detail(summary: dict, G: nx.Graph, pos: dict,
                           output_path: str = "routes_detail.png",
                           title_prefix: str = ""):
    """单独的高清路线图"""
    fig, ax = plt.subplots(figsize=(16, 14))
    draw_taxi_routes(summary['results'], G, pos, ax,
                     f"{title_prefix} Taxi Routes (All {len(summary['results'])} Flights)",
                     max_show=len(summary['results']))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white')
    print(f"[Viz] Route detail saved: {output_path}")
    plt.close()
