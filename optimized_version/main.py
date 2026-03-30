"""
主程序入口：车间调度 PPO 强化学习（优化版）

用法示例：
  python main.py                          # 默认 3000 episodes
  python main.py --episodes 3000 --eval 30
  python main.py --resume                 # 从上次检查点继续训练
  python main.py --resume --episodes 1000 # 继续训练 1000 个 episode

优化项（自动启用）：
  1. 并行环境采样      - N_ENVS = CPU核数//2，同时跑多个 episode
  2. 增大 batch size   - MINI_BATCH=512，COLLECT_INTERVAL=16
  3. 混合精度训练 AMP  - CUDA 下自动启用 autocast + GradScaler
  4. 向量化前向传播    - 候选集合整体一次 Linear，无逐条循环
"""
import argparse
import os
import pickle
import torch

from data_loader import load_all_data
from ppo_trainer import PPOTrainer, N_ENVS, DEVICE, MINI_BATCH, COLLECT_INTERVAL
from networks import HIDDEN
from visualization import (
    plot_gantt, plot_boxplot, plot_training_curves,
    export_delay_table, print_summary,
)


def parse_args():
    p = argparse.ArgumentParser(description='车间调度 PPO（优化版）')
    p.add_argument('--data',     default='data.xlsx',   help='Excel 数据文件路径')
    p.add_argument('--episodes', type=int, default=3000, help='训练 episode 总数')
    p.add_argument('--eval',     type=int, default=30,   help='评估次数（用于箱线图）')
    p.add_argument('--outdir',   default='output',       help='输出目录')
    p.add_argument('--seed',     type=int, default=42)
    p.add_argument('--resume',   action='store_true',
                   help='从 order_net.pt / machine_net.pt 检查点继续训练')
    p.add_argument('--log',      type=int, default=100,  help='日志打印间隔')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # ── 打印配置 ──────────────────────────────────────────────────
    print("=" * 55)
    print("  车间调度 PPO 强化学习（优化版）")
    print("=" * 55)
    print(f"  设备          : {DEVICE}")
    print(f"  并行环境数     : {N_ENVS}")
    print(f"  隐藏层维度     : {HIDDEN}")
    print(f"  Mini-batch    : {MINI_BATCH}")
    print(f"  Collect间隔   : {COLLECT_INTERVAL}")
    print(f"  混合精度 AMP  : {'启用' if DEVICE.type == 'cuda' else '仅 CUDA 下启用'}")
    print("=" * 55)

    # ── 加载数据 ──────────────────────────────────────────────────
    print(f"\n加载数据: {args.data}")
    data = load_all_data(args.data)
    print(f"  订单数: {len(data['orders'])}  设备数: {len(data['device_compatibility'])}")

    # ── 初始化训练器 ──────────────────────────────────────────────
    trainer = PPOTrainer(data)

    if args.resume:
        if os.path.exists('order_net.pt') and os.path.exists('machine_net.pt'):
            trainer.order_net.load_state_dict(torch.load('order_net.pt', map_location=DEVICE))
            trainer.machine_net.load_state_dict(torch.load('machine_net.pt', map_location=DEVICE))
            print("已加载网络权重检查点")
        if os.path.exists('history.pkl'):
            with open('history.pkl', 'rb') as f:
                trainer.history = pickle.load(f)
            print(f"已加载历史记录（{len(trainer.history['reward'])} episodes）")
        if os.path.exists('best_env.pkl'):
            with open('best_env.pkl', 'rb') as f:
                prev_best = pickle.load(f)
            print("已加载上次最佳调度方案")
        else:
            prev_best = None
    else:
        prev_best = None

    # ── 训练 ──────────────────────────────────────────────────────
    print(f"\n开始训练 {args.episodes} 个 episode...")
    best_env = trainer.train(n_episodes=args.episodes, log_interval=args.log)

    # 与历史最佳比较
    if prev_best is not None:
        r_new, _, _, _ = best_env.compute_reward()
        r_old, _, _, _ = prev_best.compute_reward()
        if r_old > r_new:
            print(f"历史最佳更优（{r_old:.1f} > {r_new:.1f}），保留历史最佳")
            best_env = prev_best

    # 保存检查点
    torch.save(trainer.order_net.state_dict(),   'order_net.pt')
    torch.save(trainer.machine_net.state_dict(), 'machine_net.pt')
    with open('history.pkl', 'wb') as f:
        pickle.dump(trainer.history, f)
    with open('best_env.pkl', 'wb') as f:
        pickle.dump(best_env, f)
    print("检查点已保存")

    # ── 评估 ──────────────────────────────────────────────────────
    print(f"\n评估阶段（{args.eval} 次）...")
    eval_results = trainer.evaluate(n_runs=args.eval)

    # ── 输出 ──────────────────────────────────────────────────────
    print_summary(best_env, data)

    plot_gantt(best_env,
               save_path=os.path.join(args.outdir, 'gantt.png'))
    plot_boxplot(eval_results,
                 save_path=os.path.join(args.outdir, 'boxplot.png'))
    plot_training_curves(trainer.history,
                         save_path=os.path.join(args.outdir, 'training_curves.png'))
    export_delay_table(best_env, data,
                       save_path=os.path.join(args.outdir, 'delay_table.csv'))

    print(f"\n所有输出已保存至 '{args.outdir}/' 目录")


if __name__ == '__main__':
    main()
