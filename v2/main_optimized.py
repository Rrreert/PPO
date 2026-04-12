"""主程序入口 - 优化版本"""
import sys
import os
import torch
import numpy as np
import random

from data_loader import load_all_data
from environment import ShopFloorEnv
from model_optimized import SchedulingAgent
from ppo_trainer_optimized import train, evaluate, run_episode
from visualize import (plot_gantt, plot_boxplots, plot_training_curves,
                       save_training_table, save_tardiness_table)

OUTPUT_DIR = './outputs_optimized'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='PPO训练测试（优化版）')
    parser.add_argument('--num_episodes', type=int, default=400,
                       help='训练轮数（默认400，推荐300-500）')
    parser.add_argument('--eval_episodes', type=int, default=50,
                       help='评估轮数（默认50）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()

    set_seed(args.seed)
    print("=" * 60)
    print("  基于异构图的动态车间调度系统 (PPO-V3优化版)")
    print("=" * 60)
    print(f"  训练轮数: {args.num_episodes}")
    print(f"  评估轮数: {args.eval_episodes}")
    print(f"  随机种子: {args.seed}")

    print("\n[1/5] 加载数据...")
    data = load_all_data()
    print(f"  订单: {len(data['orders'])}条, 新订单: {len(data['new_orders'])}条")
    print(f"  设备: {len(data['all_devices'])}台, 型号: {len(data['product_types'])}种")
    print(f"  故障事件: {len(data['maintenance'])}条")

    print(f"\n[2/5] PPO-V3训练 ({args.num_episodes}轮)...")
    print("  主要优化:")
    print("    ✓ 改进的密集奖励塑形")
    print("    ✓ 经验回放缓冲区")
    print("    ✓ 双Critic网络架构")
    print("    ✓ 自适应学习率调度")
    print("    ✓ 多维度启发式过滤")
    print()
    
    agent, history = train(data, num_episodes=args.num_episodes)

    print(f"\n[3/5] 评估（{args.eval_episodes}次运行）...")
    eval_results, _ = evaluate(data, agent, n_runs=args.eval_episodes)

    # 单独跑一次greedy episode做甘特图
    env_gantt = ShopFloorEnv(data)
    env_gantt.reset()
    run_episode(env_gantt, agent, data, greedy=True)

    makespans = [r['makespan'] for r in eval_results]
    tardinesses = [r['total_tardiness'] for r in eval_results]
    
    print(f"  完工时间: 均值={np.mean(makespans):.0f}min, 标准差={np.std(makespans):.0f}min")
    print(f"            中位数={np.median(makespans):.0f}min, 最优={np.min(makespans):.0f}min")
    print(f"  拖期总和: 均值={np.mean(tardinesses):.0f}min, 标准差={np.std(tardinesses):.0f}min")
    print(f"            中位数={np.median(tardinesses):.0f}min, 最优={np.min(tardinesses):.0f}min")

    # 与基线对比
    baseline_ms = 3910
    baseline_tard = 956
    improvement_ms = (baseline_ms - np.mean(makespans)) / baseline_ms * 100
    improvement_tard = (baseline_tard - np.mean(tardinesses)) / baseline_tard * 100
    
    print(f"\n  性能改进:")
    print(f"    完工时间: {improvement_ms:+.1f}% (vs EDD基线)")
    print(f"    拖期总和: {improvement_tard:+.1f}% (vs EDD基线)")

    print("\n[4/5] 生成输出文件...")
    plot_gantt(env_gantt, env_gantt.event_log,
               os.path.join(OUTPUT_DIR, '1_甘特图_优化版.png'))
    print(f"  ✓ 甘特图已保存: {OUTPUT_DIR}/1_甘特图_优化版.png")
    
    plot_boxplots(eval_results,
                  os.path.join(OUTPUT_DIR, '2_箱线图_优化版.png'))
    print(f"  ✓ 箱线图已保存: {OUTPUT_DIR}/2_箱线图_优化版.png")
    
    plot_training_curves(history,
                         os.path.join(OUTPUT_DIR, '3_训练迭代曲线_优化版.png'))
    print(f"  ✓ 训练曲线已保存: {OUTPUT_DIR}/3_训练迭代曲线_优化版.png")
    
    save_training_table(history,
                        os.path.join(OUTPUT_DIR, '3_训练数据表格_优化版.xlsx'))
    print(f"  ✓ 训练表格已保存: {OUTPUT_DIR}/3_训练数据表格_优化版.xlsx")
    
    df_tard = save_tardiness_table(env_gantt,
                                   os.path.join(OUTPUT_DIR, '4_拖期时间表_优化版.xlsx'))
    print(f"  ✓ 拖期表格已保存: {OUTPUT_DIR}/4_拖期时间表_优化版.xlsx")

    if len(df_tard) > 0:
        print(f"\n  拖期分析:")
        print(f"    拖期订单数: {len(df_tard)}")
        print(f"    最大拖期: {df_tard['拖期时间(分钟)'].max():.1f}min")
        print(f"    平均拖期: {df_tard['拖期时间(分钟)'].mean():.1f}min")
        
        mto_tard = df_tard[df_tard['订单类型'] == 'MTO']['拖期时间(分钟)'].sum()
        mts_tard = df_tard[df_tard['订单类型'] == 'MTS']['拖期时间(分钟)'].sum()
        print(f"    MTO拖期总和: {mto_tard:.1f}min")
        print(f"    MTS拖期总和: {mts_tard:.1f}min")
    else:
        print("\n  🎉 所有订单均按时完成！")

    print("\n[5/5] 完成！")
    print("=" * 60)
    
    # 保存模型
    model_path = os.path.join(OUTPUT_DIR, 'best_agent.pth')
    torch.save(agent.state_dict(), model_path)
    print(f"模型已保存至: {model_path}")

if __name__ == '__main__':
    main()
