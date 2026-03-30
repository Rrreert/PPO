"""
可视化模块：甘特图、箱线图、迭代曲线、拖期时间表
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

STAGE_COLORS = {
    'A': '#4E79A7', 'B': '#F28E2B', 'C': '#E15759',
    'D': '#76B7B2', 'E': '#59A14F', 'F': '#EDC948', 'G': '#B07AA1',
}
SETUP_COLOR = '#CCCCCC'


# ══════════════════════════════════════════════════════════════════
def plot_gantt(env, save_path='gantt.png'):
    """甘特图：横轴时间（分钟），纵轴设备，区分换料与加工，颜色区分工序。"""
    log = env.schedule_log
    if not log:
        print("调度日志为空，无法绘制甘特图")
        return

    devices = env.all_devices
    dev_y = {d: i for i, d in enumerate(devices)}
    n_dev = len(devices)

    fig, ax = plt.subplots(figsize=(20, max(8, n_dev * 0.5)))

    for entry in log:
        d     = entry['device']
        start = entry['start'] / 60
        end   = entry['end'] / 60
        dur   = end - start
        y     = dev_y[d]
        if entry['is_setup']:
            rect = mpatches.FancyBboxPatch(
                (start, y - 0.35), dur, 0.7,
                boxstyle='round,pad=0.01',
                facecolor=SETUP_COLOR, edgecolor='gray', linewidth=0.5,
                alpha=0.85
            )
            ax.add_patch(rect)
            if dur > 2:
                ax.text(start + dur/2, y, '换料',
                        ha='center', va='center', fontsize=6, color='#555555')
        else:
            color = STAGE_COLORS.get(entry['stage'], '#999999')
            rect  = mpatches.FancyBboxPatch(
                (start, y - 0.35), dur, 0.7,
                boxstyle='round,pad=0.01',
                facecolor=color, edgecolor='white', linewidth=0.5,
                alpha=0.85
            )
            ax.add_patch(rect)
            if dur > 3:
                ax.text(start + dur/2, y, entry['order_id'],
                        ha='center', va='center', fontsize=5.5, color='white',
                        fontweight='bold')

    # 坐标轴
    max_time = max(e['end'] for e in log) / 60
    ax.set_xlim(0, max_time * 1.02)
    ax.set_ylim(-0.7, n_dev - 0.3)
    ax.set_yticks(range(n_dev))
    ax.set_yticklabels(devices, fontsize=8)
    ax.set_xlabel('时间（分钟）', fontsize=11)
    ax.set_title('车间调度甘特图', fontsize=13, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    # 图例
    legend_patches = [mpatches.Patch(color=c, label=f'工序{s}') for s, c in STAGE_COLORS.items()]
    legend_patches.append(mpatches.Patch(color=SETUP_COLOR, label='换料时间'))
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8, ncol=4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"甘特图已保存：{save_path}")


# ══════════════════════════════════════════════════════════════════
def plot_boxplot(eval_results, save_path='boxplot.png'):
    """最大完工时间和拖期总和的箱线图。"""
    makespans = [r['makespan'] for r in eval_results]
    mto_delays = [r['mto_delay'] for r in eval_results]
    mts_delays = [r['mts_delay'] for r in eval_results]
    total_delays = [m + t for m, t in zip(mto_delays, mts_delays)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 完工时间
    bp1 = axes[0].boxplot(makespans, patch_artist=True,
                          boxprops=dict(facecolor='#4E79A7', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2))
    axes[0].set_title('最大完工时间分布', fontsize=12)
    axes[0].set_ylabel('时间（分钟）', fontsize=10)
    axes[0].set_xticklabels(['Makespan'])
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # 拖期
    data_bp = [mto_delays, mts_delays, total_delays]
    bp2 = axes[1].boxplot(data_bp, patch_artist=True,
                          labels=['MTO拖期', 'MTS拖期', '总拖期'],
                          boxprops=dict(alpha=0.7),
                          medianprops=dict(color='red', linewidth=2))
    colors = ['#F28E2B', '#E15759', '#B07AA1']
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    axes[1].set_title('拖期总和分布', fontsize=12)
    axes[1].set_ylabel('时间（分钟）', fontsize=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle(f'评估结果箱线图（n={len(eval_results)}次）', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"箱线图已保存：{save_path}")


# ══════════════════════════════════════════════════════════════════
def plot_training_curves(history, save_path='training_curves.png'):
    """训练迭代曲线：奖励、完工时间、拖期、Loss、Entropy。"""
    def smooth(data, window=10):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    metrics = [
        ('reward',        '奖励函数',         '#4E79A7'),
        ('makespan',      '最大完工时间(min)', '#F28E2B'),
        ('mto_delay',     'MTO拖期总和(min)', '#E15759'),
        ('mts_delay',     'MTS拖期总和(min)', '#59A14F'),
        ('loss_order',    'Loss（订单/设备智能体）', '#B07AA1'),
        ('entropy_order', 'Entropy（订单/设备智能体）', '#76B7B2'),
    ]

    for ax, (key, title, color) in zip(axes, metrics):
        raw = history[key]
        if key == 'loss_order':
            # 双折线：订单+设备
            raw2 = history['loss_machine']
            s1 = smooth(raw)
            s2 = smooth(raw2)
            x1 = np.linspace(0, len(raw)-1, len(s1))
            x2 = np.linspace(0, len(raw2)-1, len(s2))
            ax.plot(raw, alpha=0.2, color='#B07AA1')
            ax.plot(raw2, alpha=0.2, color='#76B7B2')
            ax.plot(x1, s1, color='#B07AA1', linewidth=2, label='订单智能体')
            ax.plot(x2, s2, color='#76B7B2', linewidth=2, label='设备智能体')
            ax.legend(fontsize=8)
        elif key == 'entropy_order':
            raw2 = history['entropy_machine']
            s1 = smooth(raw)
            s2 = smooth(raw2)
            x1 = np.linspace(0, len(raw)-1, len(s1))
            x2 = np.linspace(0, len(raw2)-1, len(s2))
            ax.plot(raw, alpha=0.2, color='#B07AA1')
            ax.plot(raw2, alpha=0.2, color='#76B7B2')
            ax.plot(x1, s1, color='#B07AA1', linewidth=2, label='订单智能体')
            ax.plot(x2, s2, color='#76B7B2', linewidth=2, label='设备智能体')
            ax.legend(fontsize=8)
        else:
            s = smooth(raw)
            x = np.linspace(0, len(raw)-1, len(s))
            ax.plot(raw, alpha=0.2, color=color)
            ax.plot(x, s, color=color, linewidth=2)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Episode', fontsize=9)
        ax.grid(linestyle='--', alpha=0.4)

    plt.suptitle('PPO 训练迭代曲线', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"迭代曲线已保存：{save_path}")


# ══════════════════════════════════════════════════════════════════
def export_delay_table(env, data, save_path='delay_table.csv'):
    """过期订单的拖期时间表。"""
    rows = []
    for i, order in enumerate(data['orders']):
        ft = env.order_finish_time[i]
        if ft is None:
            continue
        delay_sec = max(0.0, ft - order['due_time'])
        rows.append({
            '订单编号':   order['order_id'],
            '产品型号':   order['product_type'],
            '产品数量':   order['quantity'],
            '生产方式':   order['order_type'],
            '交货时间(min)': order['due_time'] / 60,
            '完工时间(min)': round(ft / 60, 2),
            '拖期时间(min)': round(delay_sec / 60, 2),
            '是否拖期':   '是' if delay_sec > 0 else '否',
        })

    df = pd.DataFrame(rows)
    delayed = df[df['拖期时间(min)'] > 0].sort_values('拖期时间(min)', ascending=False)
    all_df  = df.sort_values('订单编号')

    all_df.to_csv(save_path, index=False, encoding='utf-8-sig')

    print(f"\n拖期时间表已保存：{save_path}")
    print(f"共 {len(delayed)} 个订单拖期")
    if not delayed.empty:
        print(delayed[['订单编号', '生产方式', '完工时间(min)', '拖期时间(min)']].to_string(index=False))
    return all_df


# ══════════════════════════════════════════════════════════════════
def print_summary(env, data):
    """打印调度结果摘要。"""
    reward, makespan, mto_d, mts_d = env.compute_reward()
    print("\n" + "="*50)
    print("       调度结果摘要")
    print("="*50)
    print(f"  最大完工时间  : {makespan/60:.2f} 分钟")
    print(f"  MTO 拖期总和  : {mto_d/60:.2f} 分钟")
    print(f"  MTS 拖期总和  : {mts_d/60:.2f} 分钟")
    print(f"  奖励函数值    : {reward:.2f}")
    print("="*50)
