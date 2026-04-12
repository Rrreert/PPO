"""
对比测试脚本 - 原版 vs 优化版

运行此脚本可以直观看到优化前后的性能差异
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os

def quick_comparison_test():
    """快速对比测试（各运行50轮）"""
    from data_loader import load_all_data
    from environment import ShopFloorEnv
    
    print("=" * 70)
    print("  快速对比测试：原版 vs 优化版")
    print("=" * 70)
    print("\n加载数据...")
    data = load_all_data()
    
    # 设置随机种子确保公平对比
    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    results_comparison = {
        '原版': {'makespans': [], 'tardiness': []},
        '优化版': {'makespans': [], 'tardiness': []}
    }
    
    # 测试原版
    print("\n" + "=" * 70)
    print("[1/2] 测试原版（50轮）...")
    print("=" * 70)
    set_seed(42)
    
    from model import SchedulingAgent as OriginalAgent
    from ppo_trainer import train as original_train
    
    agent_orig, history_orig = original_train(data, num_episodes=50)
    
    for h in history_orig[-10:]:  # 取最后10轮
        results_comparison['原版']['makespans'].append(h['makespan'])
        results_comparison['原版']['tardiness'].append(h['total_tardiness'])
    
    # 测试优化版
    print("\n" + "=" * 70)
    print("[2/2] 测试优化版（50轮）...")
    print("=" * 70)
    set_seed(42)
    
    from model_optimized import SchedulingAgent as OptimizedAgent
    from ppo_trainer_optimized import train as optimized_train
    
    agent_opt, history_opt = optimized_train(data, num_episodes=50)
    
    for h in history_opt[-10:]:  # 取最后10轮
        results_comparison['优化版']['makespans'].append(h['makespan'])
        results_comparison['优化版']['tardiness'].append(h['total_tardiness'])
    
    # 统计分析
    print("\n" + "=" * 70)
    print("对比结果汇总")
    print("=" * 70)
    
    baseline_ms = 3910
    baseline_tard = 956
    
    for version in ['原版', '优化版']:
        ms_data = results_comparison[version]['makespans']
        tard_data = results_comparison[version]['tardiness']
        
        ms_mean = np.mean(ms_data)
        ms_std = np.std(ms_data)
        ms_best = np.min(ms_data)
        
        tard_mean = np.mean(tard_data)
        tard_std = np.std(tard_data)
        tard_best = np.min(tard_data)
        
        ms_improve = (baseline_ms - ms_mean) / baseline_ms * 100
        tard_improve = (baseline_tard - tard_mean) / baseline_tard * 100
        
        print(f"\n【{version}】(最后10轮统计)")
        print(f"  完工时间:")
        print(f"    平均值: {ms_mean:.0f} min (vs 基线: {ms_improve:+.1f}%)")
        print(f"    标准差: {ms_std:.0f} min")
        print(f"    最优值: {ms_best:.0f} min")
        print(f"  拖期总和:")
        print(f"    平均值: {tard_mean:.0f} min (vs 基线: {tard_improve:+.1f}%)")
        print(f"    标准差: {tard_std:.0f} min")
        print(f"    最优值: {tard_best:.0f} min")
    
    # 计算改进幅度
    ms_orig = np.mean(results_comparison['原版']['makespans'])
    ms_opt = np.mean(results_comparison['优化版']['makespans'])
    tard_orig = np.mean(results_comparison['原版']['tardiness'])
    tard_opt = np.mean(results_comparison['优化版']['tardiness'])
    
    ms_delta = (ms_orig - ms_opt) / ms_orig * 100
    tard_delta = (tard_orig - tard_opt) / tard_orig * 100
    
    print("\n" + "-" * 70)
    print("优化版相对原版的改进:")
    print(f"  完工时间: {ms_delta:+.1f}%")
    print(f"  拖期总和: {tard_delta:+.1f}%")
    print("-" * 70)
    
    # 绘制对比图
    print("\n生成对比图表...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 完工时间对比
    ax1 = axes[0]
    positions = [1, 2]
    box_data = [results_comparison['原版']['makespans'], 
                results_comparison['优化版']['makespans']]
    bp1 = ax1.boxplot(box_data, positions=positions, widths=0.5,
                      patch_artist=True, showmeans=True)
    
    colors = ['lightcoral', 'lightblue']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    
    ax1.axhline(y=baseline_ms, color='r', linestyle='--', 
                label=f'EDD Baseline ({baseline_ms}min)', alpha=0.7)
    ax1.set_ylabel('Makespan (min)', fontsize=11)
    ax1.set_title('Makespan Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(positions)
    ax1.set_xticklabels(['Original', 'Optimized'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 拖期对比
    ax2 = axes[1]
    box_data2 = [results_comparison['原版']['tardiness'], 
                 results_comparison['优化版']['tardiness']]
    bp2 = ax2.boxplot(box_data2, positions=positions, widths=0.5,
                      patch_artist=True, showmeans=True)
    
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.axhline(y=baseline_tard, color='r', linestyle='--', 
                label=f'EDD Baseline ({baseline_tard}min)', alpha=0.7)
    ax2.set_ylabel('Total Tardiness (min)', fontsize=11)
    ax2.set_title('Tardiness Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(positions)
    ax2.set_xticklabels(['Original', 'Optimized'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = './comparison_result.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 对比图已保存: {output_path}")
    
    # 生成训练曲线对比
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    
    # Makespan曲线
    ax = axes2[0, 0]
    ax.plot([h['episode'] for h in history_orig], 
            [h['makespan'] for h in history_orig],
            'o-', label='Original', alpha=0.6, markersize=3)
    ax.plot([h['episode'] for h in history_opt], 
            [h['makespan'] for h in history_opt],
            's-', label='Optimized', alpha=0.6, markersize=3)
    ax.axhline(y=baseline_ms, color='r', linestyle='--', label='EDD Baseline', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Makespan (min)')
    ax.set_title('Makespan Training Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Tardiness曲线
    ax = axes2[0, 1]
    ax.plot([h['episode'] for h in history_orig], 
            [h['total_tardiness'] for h in history_orig],
            'o-', label='Original', alpha=0.6, markersize=3)
    ax.plot([h['episode'] for h in history_opt], 
            [h['total_tardiness'] for h in history_opt],
            's-', label='Optimized', alpha=0.6, markersize=3)
    ax.axhline(y=baseline_tard, color='r', linestyle='--', label='EDD Baseline', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Tardiness (min)')
    ax.set_title('Tardiness Training Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss曲线
    ax = axes2[1, 0]
    ax.plot([h['episode'] for h in history_orig], 
            [h['loss'] for h in history_orig],
            'o-', label='Original', alpha=0.6, markersize=3)
    ax.plot([h['episode'] for h in history_opt], 
            [h['loss'] for h in history_opt],
            's-', label='Optimized', alpha=0.6, markersize=3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Entropy曲线
    ax = axes2[1, 1]
    ax.plot([h['episode'] for h in history_orig], 
            [h['entropy'] for h in history_orig],
            'o-', label='Original', alpha=0.6, markersize=3)
    ax.plot([h['episode'] for h in history_opt], 
            [h['entropy'] for h in history_opt],
            's-', label='Optimized', alpha=0.6, markersize=3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Entropy')
    ax.set_title('Entropy Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path2 = './comparison_curves.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✓ 训练曲线对比已保存: {output_path2}")
    
    print("\n" + "=" * 70)
    print("快速对比测试完成！")
    print("=" * 70)
    
    return results_comparison


if __name__ == '__main__':
    results = quick_comparison_test()
