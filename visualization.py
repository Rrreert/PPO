"""
可视化模块：
(1) 甘特图（标注换料时间）
(2) 最大完工时间、拖期总和的箱线图
(3) 奖励/完工/拖期/Loss/Entropy 迭代图
(4) 过期订单拖期时间表
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from environment import ORDERS, OPS, N_ORDERS

plt.rcParams['font.family'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 颜色映射：订单
CMAP = plt.get_cmap('tab20')
ORDER_COLORS = {o['order_id']: CMAP(i % 20) for i, o in enumerate(ORDERS)}
ORDER_ID_TO_MODE = {o['order_id']: o['mode'] for o in ORDERS}


def plot_gantt(gantt_data, save_path='gantt.png'):
    """
    绘制甘特图
    gantt_data: list of {'machine','op','order_id','start','end','is_setup'}
    """
    if not gantt_data:
        print("No gantt data")
        return

    from environment import ALL_MACHINES
    machines = ALL_MACHINES
    m_idx    = {m: i for i, m in enumerate(machines)}
    n_m      = len(machines)

    fig, ax = plt.subplots(figsize=(20, max(8, n_m * 0.45)))

    max_end = max(d['end'] for d in gantt_data) / 60  # 转分钟

    for d in gantt_data:
        mi    = m_idx[d['machine']]
        start = d['start'] / 60
        dur   = (d['end'] - d['start']) / 60
        oid   = d['order_id']

        if d['is_setup']:
            color = '#d62728'
            ec    = 'darkred'
            alpha = 0.85
            label_text = '换料'
        else:
            color = ORDER_COLORS[oid]
            ec    = 'black'
            alpha = 0.8
            label_text = oid

        bar = ax.barh(mi, dur, left=start, height=0.6,
                      color=color, edgecolor=ec, linewidth=0.5, alpha=alpha)

        # 在 bar 中显示文字（若足够宽）
        if dur > max_end * 0.012:
            ax.text(start + dur / 2, mi, label_text,
                    ha='center', va='center', fontsize=5.5,
                    color='white' if not d['is_setup'] else 'white',
                    fontweight='bold')

    ax.set_yticks(range(n_m))
    ax.set_yticklabels(machines, fontsize=8)
    ax.set_xlabel('Time (min)', fontsize=11)
    ax.set_title('Workshop Scheduling Gantt Chart', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max_end * 1.02)

    # 图例
    handles = [
        mpatches.Patch(color='#d62728', label='Setup / 换料'),
    ]
    # 按交货期分组标注
    for mode, col in [('MTO', CMAP(2)), ('MTS', CMAP(0))]:
        handles.append(mpatches.Patch(color=col, label=mode, alpha=0.8))
    ax.legend(handles=handles, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gantt chart saved: {save_path}")


def plot_boxplots(makespan_list, mto_tard_list, mts_tard_list, save_path='boxplot.png'):
    """
    绘制最大完工时间与拖期总和的箱线图
    """
    tard_list = [m + t for m, t in zip(mto_tard_list, mts_tard_list)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bp1 = axes[0].boxplot(makespan_list, patch_artist=True,
                          boxprops=dict(facecolor='#4C72B0', alpha=0.7))
    axes[0].set_title('Makespan Distribution (min)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Minutes')
    axes[0].set_xticks([1])
    axes[0].set_xticklabels(['Makespan'])

    bp2 = axes[1].boxplot([mto_tard_list, mts_tard_list, tard_list],
                          patch_artist=True,
                          boxprops=dict(alpha=0.7))
    colors = ['#DD8452', '#55A868', '#C44E52']
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    axes[1].set_title('Tardiness Distribution (min)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Minutes')
    axes[1].set_xticks([1, 2, 3])
    axes[1].set_xticklabels(['MTO Tardiness', 'MTS Tardiness', 'Total Tardiness'])

    plt.suptitle('Performance Distribution (All Episodes)', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Boxplot saved: {save_path}")


def plot_training_curves(history, save_path='training_curves.png'):
    """
    绘制训练迭代曲线：reward, makespan, tardiness, loss, entropy
    """
    keys     = ['reward', 'makespan', 'mto_tard', 'mts_tard',
                 'loss_o', 'loss_m', 'entropy_o', 'entropy_m']
    titles   = ['Episode Reward', 'Makespan (min)',
                 'MTO Tardiness (min)', 'MTS Tardiness (min)',
                 'Order Agent Loss', 'Machine Agent Loss',
                 'Order Agent Entropy', 'Machine Agent Entropy']
    colors   = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                 '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    for i, (key, title, color) in enumerate(zip(keys, titles, colors)):
        data = history[key]
        x    = range(1, len(data) + 1)
        axes[i].plot(x, data, color=color, alpha=0.4, linewidth=0.8)
        # 平滑曲线
        window = max(1, len(data) // 20)
        smoothed = np.convolve(data, np.ones(window) / window, mode='valid')
        xs = range(window, len(data) + 1)
        axes[i].plot(xs, smoothed, color=color, linewidth=2)
        axes[i].set_title(title, fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Episode')
        axes[i].grid(True, alpha=0.3)

    plt.suptitle('PPO Training Curves', fontsize=14, fontweight='bold', y=1.005)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved: {save_path}")


def make_tardiness_table(env, save_path='tardiness_table.csv'):
    """
    生成过期订单的拖期时间表
    """
    order_id_to_info = {o['order_id']: o for o in ORDERS}
    rows = []
    for os in env.order_states:
        tard = os.tardiness() / 60  # 转分钟
        info = order_id_to_info[os.id]
        ct   = (os.completion_time() or 0) / 60
        rows.append({
            '订单编号': os.id,
            '产品型号': os.product_type,
            '产品数量': info['quantity'],
            '生产方式': info['mode'],
            '交货时间(分钟)': info['due_time'],
            '实际完工时间(分钟)': round(ct, 1),
            '拖期时间(分钟)': round(tard, 1),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('拖期时间(分钟)', ascending=False)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"Tardiness table saved: {save_path} ({len(df)} overdue orders)")
    return df


def run_multiple_episodes(n_eval, env, order_policy, order_value,
                          machine_policy, machine_value):
    """
    运行多个 episode 收集箱线图数据。
    使用随机采样（training=True）而非贪心，使每次结果有真实方差，
    """
    from trainer import run_episode, RolloutBuffer
    makespans, mto_tards, mts_tards = [], [], []
    dummy_buf = RolloutBuffer()
    for _ in range(n_eval):
        dummy_buf.clear()
        run_episode(env, order_policy, order_value,
                    machine_policy, machine_value, dummy_buf, training=True)  # 随机采样
        m = env.get_metrics()
        makespans.append(m['makespan'])
        mto_tards.append(m['mto_tardiness'])
        mts_tards.append(m['mts_tardiness'])
    return makespans, mto_tards, mts_tards
