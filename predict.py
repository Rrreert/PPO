"""
predict.py — 加载 ppo_final.pt，对新订单数据进行调度预测
=============================================================
用法：
    python predict.py --model ppo_final.pt --data new_data.xlsx --output ./predict_out

new_data.xlsx 格式要求：
  - 工艺数据页（设备通用性、生产加工时间、换料时间、最短剩余加工时间）与训练时完全一致
  - 只需替换「订单」Sheet 的内容，列顺序：
        订单编号 | 产品型号 | 数量 | 生产方式(MTO/MTS) | 交货期(分钟)
  - 产品型号范围：01~15；数量为正整数；交货期单位为分钟

输出：
    gantt.png          — 甘特图
    tardiness_table.csv — 拖期时间表
    metrics.txt        — 关键指标汇总
"""

import argparse, os, sys
import torch
import numpy as np

# ────────────────────────────────────────────────────────────────
# 1. 参数解析
# ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='PPO 车间调度 — 新订单预测')
parser.add_argument('--model',  default='ppo_final.pt',   help='模型文件路径')
parser.add_argument('--data',   default='data.xlsx',      help='订单 Excel 路径')
parser.add_argument('--output', default='./predict_out',  help='输出目录')
parser.add_argument('--eval',   type=int, default=1,
                    help='推理次数：1=贪心确定性结果；>1=随机采样取最优')
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

# ────────────────────────────────────────────────────────────────
# 2. 把新 Excel 路径注入到 data_loader
#    （data_loader.py 里 EXCEL_PATH 是硬编码的，这里用环境变量覆盖）
# ────────────────────────────────────────────────────────────────
os.environ['PPO_DATA_PATH'] = os.path.abspath(args.data)

# 打猴子补丁：在 import data_loader 之前替换路径常量
import importlib, types

_orig_data_loader = None
def _patch_data_loader():
    import data_loader
    data_loader.EXCEL_PATH = os.environ['PPO_DATA_PATH']
    # 强制重新加载（清除模块级缓存变量）
    importlib.reload(data_loader)

_patch_data_loader()

# ────────────────────────────────────────────────────────────────
# 3. 导入项目模块（data_loader 已指向新 Excel）
# ────────────────────────────────────────────────────────────────
from models import (
    OrderPolicyNet, OrderValueNet,
    MachinePolicyNet, MachineValueNet,
)
from environment import WorkshopEnv
from trainer import run_episode, RolloutBuffer
from visualization import plot_gantt, make_tardiness_table

# ────────────────────────────────────────────────────────────────
# 4. 加载模型
# ────────────────────────────────────────────────────────────────
print(f"[1/4] 加载模型：{args.model}")
ckpt = torch.load(args.model, map_location='cpu')

order_policy   = OrderPolicyNet();   order_policy.load_state_dict(ckpt['order_policy'])
order_value    = OrderValueNet();    order_value.load_state_dict(ckpt['order_value'])
machine_policy = MachinePolicyNet(); machine_policy.load_state_dict(ckpt['machine_policy'])
machine_value  = MachineValueNet();  machine_value.load_state_dict(ckpt['machine_value'])

for net in [order_policy, order_value, machine_policy, machine_value]:
    net.eval()

print("      模型加载完成 ✓")

# ────────────────────────────────────────────────────────────────
# 5. 推理
#    eval=1  → 贪心（training=False），结果确定，适合生产部署
#    eval>1  → 多次随机采样，取 reward 最大的那次，适合离线优化
# ────────────────────────────────────────────────────────────────
print(f"[2/4] 开始推理（eval={args.eval}）...")

env = WorkshopEnv()
buf = RolloutBuffer()

best_reward  = float('-inf')
best_env     = None

n_runs = args.eval if args.eval > 1 else 1
use_sampling = args.eval > 1   # >1 时用随机采样，1 时用贪心

for i in range(n_runs):
    buf.clear()
    reward = run_episode(
        env, order_policy, order_value,
        machine_policy, machine_value,
        buf, training=use_sampling,
    )
    if reward > best_reward:
        best_reward = reward
        # 深拷贝当前环境状态（保留甘特图数据）
        import copy
        best_env = copy.deepcopy(env)
    if n_runs > 1:
        print(f"      第 {i+1}/{n_runs} 次：reward={reward:.1f}")

print(f"      推理完成，最优 reward={best_reward:.1f} ✓")

# ────────────────────────────────────────────────────────────────
# 6. 生成输出
# ────────────────────────────────────────────────────────────────
print("[3/4] 生成甘特图...")
gantt_data = best_env.get_gantt_data()
plot_gantt(gantt_data, save_path=f"{args.output}/gantt.png")

print("[4/4] 生成拖期时间表 & 指标汇总...")
df_tard = make_tardiness_table(best_env, save_path=f"{args.output}/tardiness_table.csv")
metrics = best_env.get_metrics()

# 写指标文件
summary = (
    f"=== 调度结果汇总 ===\n"
    f"Makespan（总完工时间）: {metrics['makespan']:.1f} 分钟\n"
    f"MTO 拖期总和:           {metrics['mto_tardiness']:.1f} 分钟\n"
    f"MTS 拖期总和:           {metrics['mts_tardiness']:.1f} 分钟\n"
    f"拖期订单数:             {len(df_tard)}\n"
)
with open(f"{args.output}/metrics.txt", 'w', encoding='utf-8') as f:
    f.write(summary)

print()
print("=" * 45)
print(summary)
print(f"所有输出已保存至：{args.output}")
print("=" * 45)
