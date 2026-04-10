"""主程序入口"""
import sys, os
sys.path.insert(0, '/home/claude/scheduler')

import torch
import numpy as np
import random

from data_loader import load_all_data
from environment import ShopFloorEnv
from model import SchedulingAgent
from ppo_trainer import train, evaluate, run_episode
from visualize import (plot_gantt, plot_boxplots, plot_training_curves,
                       save_training_table, save_tardiness_table)

OUTPUT_DIR = '/mnt/user-data/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    set_seed(42)
    print("=" * 60)
    print("  基于异构图的动态车间调度系统 (PPO)")
    print("=" * 60)

    print("\n[1/5] 加载数据...")
    data = load_all_data()
    print(f"  订单: {len(data['orders'])}条, 新订单: {len(data['new_orders'])}条")
    print(f"  设备: {len(data['all_devices'])}台, 型号: {len(data['product_types'])}种")
    print(f"  故障事件: {len(data['maintenance'])}条")

    print("\n[2/5] PPO训练 (100轮)...")
    agent, history = train(data, num_episodes=100)

    print("\n[3/5] 评估（20次运行）...")
    eval_results, _ = evaluate(data, agent, n_runs=20)

    # 单独跑一次greedy episode做甘特图
    env_gantt = ShopFloorEnv(data)
    env_gantt.reset()
    run_episode(env_gantt, agent, data, greedy=True)

    makespans = [r['makespan'] for r in eval_results]
    tardinesses = [r['total_tardiness'] for r in eval_results]
    print(f"  完工时间: 均值={np.mean(makespans):.0f}min, 标准差={np.std(makespans):.0f}min")
    print(f"  拖期总和: 均值={np.mean(tardinesses):.0f}min, 标准差={np.std(tardinesses):.0f}min")

    print("\n[4/5] 生成输出文件...")
    plot_gantt(env_gantt, env_gantt.event_log,
               os.path.join(OUTPUT_DIR, '1_甘特图.png'))
    plot_boxplots(eval_results,
                  os.path.join(OUTPUT_DIR, '2_箱线图.png'))
    plot_training_curves(history,
                         os.path.join(OUTPUT_DIR, '3_训练迭代曲线.png'))
    save_training_table(history,
                        os.path.join(OUTPUT_DIR, '3_训练数据表格.xlsx'))
    df_tard = save_tardiness_table(env_gantt,
                                   os.path.join(OUTPUT_DIR, '4_拖期时间表.xlsx'))

    if len(df_tard) > 0:
        print(f"  共 {len(df_tard)} 个订单拖期，最大拖期: {df_tard['拖期时间(分钟)'].max():.1f}min")
    else:
        print("  所有订单均按时完成！")

    print("\n[5/5] 完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()
