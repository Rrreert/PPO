"""
主入口：训练 PPO 并生成全部输出
"""
import os, sys
sys.path.insert(0, '/kaggle/working/PPO')

OUTPUT_DIR = '/kaggle/working/PPO/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

from trainer import train, run_episode, RolloutBuffer, N_EPISODES
from visualize import (
    plot_gantt, plot_boxplots, plot_training_curves,
    make_tardiness_table, run_multiple_episodes,
)

print("=" * 60)
print("  车间调度 PPO 训练启动")
print(f"  训练轮次: {N_EPISODES}")
print("=" * 60)

# ---- 训练 ----
history, env, order_policy, order_value, machine_policy, machine_value = train(
    n_episodes=N_EPISODES,
    save_path=OUTPUT_DIR,
)

print("\n[1/4] 生成甘特图...")
# 运行一次最终评估 episode（贪心）
dummy_buf = RolloutBuffer()
dummy_buf.clear()
run_episode(env, order_policy, order_value,
            machine_policy, machine_value, dummy_buf, training=False)
gantt_data = env.get_gantt_data()
plot_gantt(gantt_data, save_path=f"{OUTPUT_DIR}/gantt.png")

print("\n[2/4] 生成箱线图（评估 30 个 episode）...")
makespans, mto_tards, mts_tards = run_multiple_episodes(
    30, env, order_policy, order_value, machine_policy, machine_value
)
plot_boxplots(makespans, mto_tards, mts_tards,
              save_path=f"{OUTPUT_DIR}/boxplot.png")

print("\n[3/4] 生成训练迭代曲线...")
plot_training_curves(history, save_path=f"{OUTPUT_DIR}/training_curves.png")

print("\n[4/4] 生成拖期时间表...")
# 用最后一次 episode 结果
dummy_buf.clear()
run_episode(env, order_policy, order_value,
            machine_policy, machine_value, dummy_buf, training=False)
df_tard = make_tardiness_table(env, save_path=f"{OUTPUT_DIR}/tardiness_table.csv")

print("\n" + "=" * 60)
print("  所有输出已保存至:", OUTPUT_DIR)
m = env.get_metrics()
print(f"  最终 Makespan:      {m['makespan']:.1f} 分钟")
print(f"  MTO 拖期总和:       {m['mto_tardiness']:.1f} 分钟")
print(f"  MTS 拖期总和:       {m['mts_tardiness']:.1f} 分钟")
print(f"  过期订单数:         {len(df_tard)}")
print("=" * 60)
