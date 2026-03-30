"""
主入口 main.py
车间调度 PPO 强化学习系统
使用方法:
  python main.py --mode train --iterations 500
  python main.py --mode inference --load_model
  python main.py --mode quick_test  # 快速验证流程（少量迭代）
"""
import os
import sys
import argparse
import torch
import numpy as np

# 设置路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

from data_loader import load_all_data
from environment import SchedulingEnv
from ppo_trainer import PPOTrainer, PPOConfig
from visualization import (
    plot_gantt, plot_boxplot, plot_training_curves,
    generate_delay_table, save_delay_excel
)

DATA_PATH = '/mnt/user-data/uploads/data.xlsx'


def build_machines_order(data):
    """构造甘特图纵轴设备顺序（按工序A->B->C->D->E->F->G）"""
    order = []
    for stage in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        order.extend(data['stage_to_machines'].get(stage, []))
    return order


def run_quick_test(data_path=DATA_PATH):
    """
    快速流程验证：仅跑 3 次迭代，验证代码能完整跑通
    """
    print("\n" + "="*60)
    print("  快速验证模式 (quick_test)")
    print("="*60)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    data = load_all_data(data_path)
    print(f"  ✓ 订单数: {len(data['orders'])}")
    print(f"  ✓ 设备总数: {sum(len(v) for v in data['stage_to_machines'].values())}")

    # 2. 初始化环境
    print("\n[2/5] 初始化环境...")
    env = SchedulingEnv(data)
    state = env.reset()
    print(f"  ✓ 订单特征维度: {state['order_features'].shape}")
    print(f"  ✓ 设备特征维度: {state['machine_features'].shape}")

    # 3. 构建并测试网络
    print("\n[3/5] 构建 PPO 网络...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  ✓ 使用设备: {device}")

    config = PPOConfig()
    config.n_episodes_per_update = 2  # 快速测试减少采集量
    config.n_epochs = 2
    config.batch_size = 16

    trainer = PPOTrainer(data, config=config, device=device)
    total_params = (sum(p.numel() for p in trainer.order_agent.parameters()) +
                    sum(p.numel() for p in trainer.machine_agent.parameters()))
    print(f"  ✓ 双智能体参数总量: {total_params:,}")

    # 4. 训练（仅3次迭代）
    print("\n[4/5] 运行 PPO 训练（3次迭代验证）...")
    history = trainer.train(n_iterations=3, log_interval=1)
    print("  ✓ 训练流程正常")

    # 5. 推理并生成输出
    print("\n[5/5] 运行推理并生成可视化输出...")
    schedule_history, metrics = trainer.run_inference()

    print(f"\n  调度结果：")
    print(f"  ✓ 最大完工时间: {metrics['makespan_min']:.1f} 分钟")
    print(f"  ✓ MTO拖期总和: {metrics['mto_delay']/60:.1f} 分钟")
    print(f"  ✓ MTS拖期总和: {metrics['mts_delay']/60:.1f} 分钟")
    overdue_count = sum(1 for d in metrics['order_delays'] if d['delay_sec'] > 0)
    print(f"  ✓ 逾期订单数: {overdue_count}/{len(metrics['order_delays'])}")

    machines_order = build_machines_order(data)

    # 生成甘特图
    gantt_path = os.path.join(OUTPUT_DIR, 'gantt.png')
    plot_gantt(schedule_history, machines_order, save_path=gantt_path)

    # 生成箱线图
    boxplot_path = os.path.join(OUTPUT_DIR, 'boxplot.png')
    plot_boxplot(history, save_path=boxplot_path)

    # 生成迭代曲线
    curve_path = os.path.join(OUTPUT_DIR, 'training_curves.png')
    plot_training_curves(history, save_path=curve_path)

    # 生成拖期时间表
    delay_img_path = os.path.join(OUTPUT_DIR, 'delay_table.png')
    generate_delay_table(metrics, save_path=delay_img_path)

    # 保存拖期 Excel
    delay_excel_path = os.path.join(OUTPUT_DIR, 'delay_table.xlsx')
    df_delay = save_delay_excel(metrics, save_path=delay_excel_path)

    print(f"\n  ✓ 所有输出已保存至: {OUTPUT_DIR}/")
    print("\n" + "="*60)
    print("  快速验证完成！代码流程全部正常 ✓")
    print("="*60)
    return history, metrics


def run_full_train(n_iterations=500, data_path=DATA_PATH):
    """完整训练模式"""
    print("\n" + "="*60)
    print(f"  完整训练模式 ({n_iterations} 迭代)")
    print("="*60)

    data = load_all_data(data_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = PPOConfig()

    trainer = PPOTrainer(data, config=config, device=device)
    history = trainer.train(n_iterations=n_iterations, log_interval=20)

    # 保存模型
    trainer.save_models(
        path_order=os.path.join(OUTPUT_DIR, 'order_agent.pth'),
        path_machine=os.path.join(OUTPUT_DIR, 'machine_agent.pth')
    )

    # 推理
    schedule_history, metrics = trainer.run_inference()
    machines_order = build_machines_order(data)

    # 生成全部可视化
    plot_gantt(schedule_history, machines_order,
               save_path=os.path.join(OUTPUT_DIR, 'gantt.png'))
    plot_boxplot(history, save_path=os.path.join(OUTPUT_DIR, 'boxplot.png'))
    plot_training_curves(history, save_path=os.path.join(OUTPUT_DIR, 'training_curves.png'))
    generate_delay_table(metrics, save_path=os.path.join(OUTPUT_DIR, 'delay_table.png'))
    save_delay_excel(metrics, save_path=os.path.join(OUTPUT_DIR, 'delay_table.xlsx'))

    print("\n完整训练结束！")
    print(f"最大完工时间: {metrics['makespan_min']:.1f} 分钟")
    print(f"拖期总和: {metrics['total_delay']/60:.1f} 分钟")
    return history, metrics


def run_inference_only(data_path=DATA_PATH):
    """仅推理模式（需要预训练好的模型）"""
    data = load_all_data(data_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = PPOTrainer(data, device=device)
    trainer.load_models(
        path_order=os.path.join(OUTPUT_DIR, 'order_agent.pth'),
        path_machine=os.path.join(OUTPUT_DIR, 'machine_agent.pth')
    )
    schedule_history, metrics = trainer.run_inference()
    machines_order = build_machines_order(data)
    plot_gantt(schedule_history, machines_order,
               save_path=os.path.join(OUTPUT_DIR, 'gantt_inference.png'))
    generate_delay_table(metrics, save_path=os.path.join(OUTPUT_DIR, 'delay_table_inference.png'))
    save_delay_excel(metrics, save_path=os.path.join(OUTPUT_DIR, 'delay_inference.xlsx'))
    return schedule_history, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='车间调度 PPO 系统')
    parser.add_argument('--mode', type=str, default='quick_test',
                        choices=['quick_test', 'train', 'inference'],
                        help='运行模式')
    parser.add_argument('--iterations', type=int, default=500,
                        help='训练迭代次数（train模式）')
    parser.add_argument('--data', type=str, default=DATA_PATH,
                        help='Excel数据文件路径')
    args = parser.parse_args()

    if args.mode == 'quick_test':
        run_quick_test(data_path=args.data)
    elif args.mode == 'train':
        run_full_train(n_iterations=args.iterations, data_path=args.data)
    elif args.mode == 'inference':
        run_inference_only(data_path=args.data)
