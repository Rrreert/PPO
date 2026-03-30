"""
主程序入口：车间调度 PPO 强化学习
用法：python main.py [--episodes 300] [--eval 20] [--data data.xlsx]
"""
import argparse
import os
import torch

from data_loader import load_all_data
from ppo_trainer import PPOTrainer
from visualization import (
    plot_gantt, plot_boxplot, plot_training_curves,
    export_delay_table, print_summary
)


def parse_args():
    parser = argparse.ArgumentParser(description='车间调度 PPO')
    parser.add_argument('--data',     default='/mnt/user-data/uploads/data.xlsx')
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--eval',     type=int, default=20)
    parser.add_argument('--outdir',   default='output')
    parser.add_argument('--seed',     type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    print("加载数据...")
    data = load_all_data(args.data)
    print(f"  订单数: {len(data['orders'])}")
    print(f"  设备数: {len(data['device_compatibility'])}")
    print(f"  产品型号数: {len(data['product_types'])}")

    # ── 训练 ──────────────────────────────────────────────────────
    trainer = PPOTrainer(data)
    best_env = trainer.train(n_episodes=args.episodes, log_interval=20)

    # ── 评估 ──────────────────────────────────────────────────────
    print(f"\n开始评估 ({args.eval} 次)...")
    eval_results = trainer.evaluate(n_runs=args.eval)

    # ── 输出 ──────────────────────────────────────────────────────
    print_summary(best_env, data)

    gantt_path   = os.path.join(args.outdir, 'gantt.png')
    box_path     = os.path.join(args.outdir, 'boxplot.png')
    curve_path   = os.path.join(args.outdir, 'training_curves.png')
    delay_path   = os.path.join(args.outdir, 'delay_table.csv')

    plot_gantt(best_env, save_path=gantt_path)
    plot_boxplot(eval_results, save_path=box_path)
    plot_training_curves(trainer.history, save_path=curve_path)
    export_delay_table(best_env, data, save_path=delay_path)

    print(f"\n所有输出已保存到 '{args.outdir}/' 目录")


if __name__ == '__main__':
    main()
