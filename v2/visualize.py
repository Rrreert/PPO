"""结果可视化输出"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as _fm
import pandas as pd
import numpy as np

# 自动选择支持中文的字体
_cjk_font = next((f.name for f in _fm.fontManager.ttflist
                  if 'Noto Sans CJK' in f.name or 'WenQuanYi' in f.name), 'DejaVu Sans')
plt.rcParams['font.family'] = _cjk_font
plt.rcParams['axes.unicode_minus'] = False

OP_COLORS = {
    'A': '#4E79A7', 'B': '#F28E2B', 'C': '#E15759',
    'D': '#76B7B2', 'E': '#59A14F', 'F': '#EDC948', 'G': '#B07AA1'
}

def plot_gantt(env, event_log, output_path):
    """甘特图：标注换料时间、设备故障、新订单"""
    fig, ax = plt.subplots(figsize=(24, 14))

    devices = sorted(env.devices.keys())
    dev_y = {d: i for i, d in enumerate(devices)}
    n_dev = len(devices)

    new_order_ids = set()
    failure_intervals = {}  # dev -> [(start,end)]

    for ev in event_log:
        if ev['type'] == 'new_order':
            new_order_ids.add(ev['order'])
        if ev['type'] == 'breakdown':
            dev = ev['dev']
            failure_intervals.setdefault(dev, []).append((ev['fail_time']/60, ev['recover_time']/60))

    for dev_id, dev in env.devices.items():
        y = dev_y[dev_id]
        for h in dev.history:
            start_min = h['start'] / 60
            end_min = h['end'] / 60
            op = h['op']
            order_id = h['order_id']
            color = OP_COLORS.get(op, '#999999')
            is_new = order_id in new_order_ids or '_split' in order_id
            edgecolor = 'red' if is_new else 'white'
            lw = 1.5 if is_new else 0.5

            ax.barh(y, end_min - start_min, left=start_min, height=0.7,
                    color=color, edgecolor=edgecolor, linewidth=lw, alpha=0.85)

            # 换料时间标注
            if h.get('changeover_start') is not None:
                co_s = h['changeover_start'] / 60
                co_e = h['changeover_end'] / 60
                ax.barh(y, co_e - co_s, left=co_s, height=0.7,
                        color='#CCCCCC', edgecolor='orange', linewidth=1.0,
                        hatch='//', alpha=0.9)

            # 标注订单ID
            mid = (start_min + end_min) / 2
            dur = end_min - start_min
            if dur > 20:
                ax.text(mid, y, f"{order_id}\n{op}", ha='center', va='center',
                        fontsize=5.5, fontweight='bold', color='black')

        # 故障区域
        for (fs, fe) in failure_intervals.get(dev_id, []):
            ax.barh(y, fe - fs, left=fs, height=0.8,
                    color='black', alpha=0.5, edgecolor='red', linewidth=1.5,
                    hatch='xx')

    ax.set_yticks(range(n_dev))
    ax.set_yticklabels(devices, fontsize=9)
    ax.set_xlabel('时间 (分钟)', fontsize=11)
    ax.set_title('车间调度甘特图', fontsize=14, fontweight='bold')

    legend_elements = [mpatches.Patch(color=c, label=f'工序{op}') for op, c in OP_COLORS.items()]
    legend_elements += [
        mpatches.Patch(facecolor='#CCCCCC', edgecolor='orange', hatch='//', label='换料时间'),
        mpatches.Patch(facecolor='black', edgecolor='red', hatch='xx', alpha=0.5, label='设备故障'),
        mpatches.Patch(facecolor='white', edgecolor='red', linewidth=2, label='新插入订单'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=2)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  甘特图已保存: {output_path}")


def plot_boxplots(eval_results, output_path):
    """箱线图：奖励函数、最大完工时间、拖期总和"""
    rewards = [r['reward'] for r in eval_results]
    makespans = [r['makespan'] for r in eval_results]
    tardinesses = [r['total_tardiness'] for r in eval_results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    metrics = [
        (rewards, '奖励函数', '#4E79A7'),
        (makespans, '最大完工时间 (分钟)', '#F28E2B'),
        (tardinesses, '拖期总和 (分钟)', '#E15759'),
    ]
    for ax, (vals, title, color) in zip(axes, metrics):
        bp = ax.boxplot(vals, patch_artist=True, widths=0.5,
                        medianprops=dict(color='black', linewidth=2))
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.7)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('值', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        # 添加均值点
        ax.scatter([1], [np.mean(vals)], color='red', zorder=5, s=60, marker='D', label='均值')
        ax.legend(fontsize=8)

    plt.suptitle('评估结果箱线图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  箱线图已保存: {output_path}")


def plot_training_curves(history, output_path):
    """训练曲线图"""
    eps = [h['episode'] for h in history]
    metrics = [
        ('reward', '奖励函数', '#4E79A7'),
        ('makespan', '最大完工时间 (分钟)', '#F28E2B'),
        ('mto_tardiness', 'MTO拖期总和 (分钟)', '#E15759'),
        ('mts_tardiness', 'MTS拖期总和 (分钟)', '#76B7B2'),
        ('loss', 'Loss', '#59A14F'),
        ('entropy', 'Entropy', '#EDC948'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, (key, title, color) in zip(axes, metrics):
        vals = [h[key] for h in history]
        ax.plot(eps, vals, color=color, linewidth=1.5, alpha=0.8, label=title)
        # 移动平均
        w = min(10, len(vals))
        ma = np.convolve(vals, np.ones(w)/w, mode='valid')
        ax.plot(eps[w-1:], ma, color='black', linewidth=2, linestyle='--', label='移动均值')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('训练轮次', fontsize=9)
        ax.grid(linestyle='--', alpha=0.4)
        ax.legend(fontsize=8)

    plt.suptitle('训练过程迭代曲线', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  训练曲线已保存: {output_path}")


def save_training_table(history, output_path):
    """训练数据表格"""
    df = pd.DataFrame(history)
    df.columns = ['轮次', '奖励函数', '最大完工时间(分钟)', 'MTO拖期总和(分钟)',
                  'MTS拖期总和(分钟)', '拖期总和加权(分钟)', 'Loss', 'Entropy']
    df = df.round(3)
    df.to_excel(output_path, index=False)
    print(f"  训练表格已保存: {output_path}")


def save_tardiness_table(env, output_path):
    """过期订单拖期时间表"""
    rows = []
    for oid, o in env.orders.items():
        tard = o.tardiness(env.current_time)
        ft = o.finish_time / 60 if o.finish_time else None
        rows.append({
            '订单编号': o.id,
            '产品型号': o.product_type,
            '产品数量': o.qty,
            '生产方式': o.mode,
            '交货时间(分钟)': o.due / 60,
            '完工时间(分钟)': round(ft, 2) if ft else '未完成',
            '拖期时间(分钟)': round(tard / 60, 2),
            '是否拖期': '是' if tard > 0 else '否',
        })
    df = pd.DataFrame(rows)
    df = df[df['拖期时间(分钟)'] > 0].sort_values('拖期时间(分钟)', ascending=False)
    df.to_excel(output_path, index=False)
    print(f"  拖期时间表已保存: {output_path}")
    return df
