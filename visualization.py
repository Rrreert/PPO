"""
可视化模块
生成四类输出图表：
1. 甘特图（含换料时间标注）
2. 最大完工时间、拖期总和的箱线图
3. 奖励、完工时间、拖期、Loss、Entropy 迭代曲线
4. 过期订单拖期时间表
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib import rcParams

# 字体设置（支持中文）
rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'DejaVu Sans', 'Arial', 'sans-serif']
rcParams['axes.unicode_minus'] = False

# 颜色方案
STAGE_COLORS = {
    'A': '#4C72B0', 'B': '#DD8452', 'C': '#55A868',
    'D': '#C44E52', 'E': '#8172B3', 'F': '#937860', 'G': '#DA8BC3'
}
SETUP_COLOR = '#AAAAAA'
MTO_COLOR = '#E74C3C'
MTS_COLOR = '#3498DB'


def plot_gantt(schedule_history, machines_order, save_path='gantt.png'):
    """
    绘制甘特图
    schedule_history: list of dicts {order_id, machine, stage, start, end, type, product_type}
    machines_order: 设备列表（纵轴顺序）
    """
    if not schedule_history:
        print("警告：调度历史为空，无法绘制甘特图")
        return

    fig, ax = plt.subplots(figsize=(24, max(8, min(len(machines_order) * 0.55, 20))))

    machine_y = {m: i for i, m in enumerate(machines_order)}
    bar_height = 0.6

    for item in schedule_history:
        y = machine_y.get(item['machine'], 0)
        duration = item['end'] - item['start']
        if duration <= 0:
            continue

        if item['type'] == 'setup':
            color = SETUP_COLOR
            label_text = f"换料"
            alpha = 0.7
            hatch = '//'
        else:
            color = STAGE_COLORS.get(item['stage'], '#888888')
            label_text = f"{item['order_id']}\n{item['product_type']}"
            alpha = 0.85
            hatch = ''

        rect = mpatches.FancyBboxPatch(
            (item['start'] / 60, y - bar_height / 2),
            duration / 60, bar_height,
            boxstyle="round,pad=0.02",
            facecolor=color, alpha=alpha, edgecolor='white',
            linewidth=0.5, hatch=hatch
        )
        ax.add_patch(rect)

        # 标注文字（宽度足够时才标）
        if duration / 60 > 5 and item['type'] == 'process':
            ax.text(
                (item['start'] + duration / 2) / 60, y,
                f"{item['order_id']}", ha='center', va='center',
                fontsize=5.5, color='white', fontweight='bold'
            )
        elif duration / 60 > 3 and item['type'] == 'setup':
            ax.text(
                (item['start'] + duration / 2) / 60, y,
                '换料', ha='center', va='center',
                fontsize=5, color='#444444'
            )

    ax.set_yticks(range(len(machines_order)))
    ax.set_yticklabels(machines_order, fontsize=8)
    ax.set_xlabel('时间 (分钟)', fontsize=11)
    ax.set_title('车间调度甘特图（灰色为换料时间）', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(left=0, right=max(
        (item['end'] / 60 for item in schedule_history if item), default=100
    ) * 1.02)

    # 图例
    legend_patches = [mpatches.Patch(color=v, label=f'工序{k}') for k, v in STAGE_COLORS.items()]
    legend_patches.append(mpatches.Patch(color=SETUP_COLOR, label='换料', hatch='//'))
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8, ncol=4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"甘特图已保存: {save_path}")


def plot_boxplot(history, save_path='boxplot.png'):
    """
    绘制完工时间、拖期总和的箱线图
    history: {'makespans_raw': [[ep1,ep2,...], ...], 'delays_raw': [...]}
    """
    makespans = [v for ep_list in history.get('makespans_raw', [[]]) for v in ep_list]
    delays = [v for ep_list in history.get('delays_raw', [[]]) for v in ep_list]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].boxplot(makespans, patch_artist=True,
                    boxprops=dict(facecolor='#4C72B0', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    axes[0].set_title('最大完工时间分布 (分钟)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('完工时间 (分钟)')
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].boxplot(delays, patch_artist=True,
                    boxprops=dict(facecolor='#DD8452', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    axes[1].set_title('拖期总和分布 (分钟)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('拖期总和 (分钟)')
    axes[1].grid(axis='y', alpha=0.3)

    # 添加均值标注
    for ax, data in zip(axes, [makespans, delays]):
        if data:
            mean_val = np.mean(data)
            ax.axhline(mean_val, color='blue', linestyle='--', alpha=0.5, label=f'均值={mean_val:.1f}')
            ax.legend(fontsize=9)

    plt.suptitle('调度性能指标箱线图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"箱线图已保存: {save_path}")


def plot_training_curves(history, save_path='training_curves.png'):
    """
    绘制训练迭代曲线：奖励、完工时间、拖期、Loss、Entropy
    """
    keys = ['reward', 'makespan', 'total_delay', 'loss', 'entropy']
    titles = ['奖励函数', '最大完工时间 (分钟)', '拖期总和 (分钟)', '损失值 (Loss)', '熵 (Entropy)']
    colors = ['#2ECC71', '#3498DB', '#E74C3C', '#9B59B6', '#F39C12']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (key, title, color) in enumerate(zip(keys, titles, colors)):
        data = history.get(key, [])
        if not data:
            axes[i].text(0.5, 0.5, '暂无数据', ha='center', va='center',
                         transform=axes[i].transAxes, fontsize=12)
            axes[i].set_title(title)
            continue

        x = np.arange(1, len(data) + 1)
        axes[i].plot(x, data, color=color, linewidth=1.5, alpha=0.8)

        # 平滑曲线
        if len(data) >= 10:
            window = max(5, len(data) // 20)
            smoothed = np.convolve(data, np.ones(window) / window, mode='valid')
            x_smooth = np.arange(window, len(data) + 1)
            axes[i].plot(x_smooth, smoothed, color=color, linewidth=2.5,
                         linestyle='-', alpha=1.0, label='平滑曲线')

        axes[i].set_title(title, fontsize=11, fontweight='bold')
        axes[i].set_xlabel('迭代次数')
        axes[i].grid(alpha=0.3)
        if len(data) >= 10:
            axes[i].legend(fontsize=8)

    # 隐藏多余的子图
    axes[-1].axis('off')

    plt.suptitle('PPO 训练迭代曲线', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"迭代曲线已保存: {save_path}")


def generate_delay_table(metrics, save_path='delay_table.png'):
    """
    生成过期订单拖期时间表（图片形式）
    """
    order_delays = metrics.get('order_delays', [])
    overdue = [d for d in order_delays if d['delay_sec'] > 0]

    fig, ax = plt.subplots(figsize=(14, max(4, len(overdue) * 0.5 + 2)))
    ax.axis('off')

    if not overdue:
        ax.text(0.5, 0.5, '无过期订单 ✓', ha='center', va='center',
                transform=ax.transAxes, fontsize=16, color='green', fontweight='bold')
        plt.title('过期订单拖期时间表', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"拖期时间表已保存: {save_path}")
        return

    # 表格数据
    col_labels = ['订单编号', '生产方式', '交货时间(min)', '完工时间(min)', '拖期(min)', '拖期(hour)']
    table_data = []
    for d in sorted(overdue, key=lambda x: -x['delay_sec']):
        table_data.append([
            d['order_id'],
            d['mode'],
            f"{d['deadline_sec']/60:.1f}",
            f"{d['completion_time']/60:.1f}",
            f"{d['delay_min']:.1f}",
            f"{d['delay_min']/60:.2f}",
        ])

    # 颜色区分 MTO/MTS
    cell_colors = []
    for row in table_data:
        color = '#FADBD8' if row[1] == 'MTO' else '#D6EAF8'
        cell_colors.append([color] * len(col_labels))

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        cellColours=cell_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # 表头颜色
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2C3E50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    plt.title(f'过期订单拖期时间表（共 {len(overdue)} 个过期订单）\n'
              f'红色=MTO  蓝色=MTS',
              fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"拖期时间表已保存: {save_path}")


def save_delay_excel(metrics, save_path='delay_table.xlsx'):
    """保存拖期时间表为 Excel"""
    order_delays = metrics.get('order_delays', [])
    rows = []
    for d in order_delays:
        rows.append({
            '订单编号': d['order_id'],
            '生产方式': d['mode'],
            '交货时间(min)': round(d['deadline_sec'] / 60, 1),
            '完工时间(min)': round(d['completion_time'] / 60, 1),
            '拖期(min)': round(d['delay_min'], 1),
            '拖期(hour)': round(d['delay_min'] / 60, 2),
            '是否逾期': '是' if d['delay_sec'] > 0 else '否',
        })
    df = pd.DataFrame(rows)
    df.to_excel(save_path, index=False)
    print(f"拖期Excel已保存: {save_path}")
    return df


if __name__ == '__main__':
    # 测试：生成空数据图表
    dummy_history = {
        'reward': list(np.random.randn(20).cumsum() - 10),
        'makespan': list(5000 - np.arange(20) * 50 + np.random.randn(20) * 200),
        'total_delay': list(3000 - np.arange(20) * 30 + np.random.randn(20) * 100),
        'loss': list(np.exp(-np.arange(20) * 0.1) + np.random.rand(20) * 0.1),
        'entropy': list(2.0 - np.arange(20) * 0.05 + np.random.rand(20) * 0.1),
        'makespans_raw': [[4000 + np.random.randn() * 500 for _ in range(5)] for _ in range(20)],
        'delays_raw': [[2000 + np.random.randn() * 300 for _ in range(5)] for _ in range(20)],
    }
    dummy_metrics = {
        'makespan_min': 4500,
        'total_delay': 1800 * 60,
        'order_delays': [
            {'order_id': 'O-01', 'mode': 'MTS', 'deadline_sec': 1440*60,
             'completion_time': 1600*60, 'delay_sec': 160*60, 'delay_min': 160},
            {'order_id': 'O-02', 'mode': 'MTO', 'deadline_sec': 1440*60,
             'completion_time': 1500*60, 'delay_sec': 60*60, 'delay_min': 60},
            {'order_id': 'O-03', 'mode': 'MTS', 'deadline_sec': 2880*60,
             'completion_time': 2800*60, 'delay_sec': 0, 'delay_min': 0},
        ]
    }

    import os
    os.makedirs('/tmp/test_viz', exist_ok=True)
    plot_boxplot(dummy_history, '/tmp/test_viz/boxplot.png')
    plot_training_curves(dummy_history, '/tmp/test_viz/training_curves.png')
    generate_delay_table(dummy_metrics, '/tmp/test_viz/delay_table.png')
    print("可视化模块测试成功 ✓")
