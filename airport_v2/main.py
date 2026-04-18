"""
main.py
主程序：构建图 → Dijkstra → 训练PPO → 推理 → 可视化对比
"""
import os, sys, time
os.chdir("/kaggle/working/PPO/airport_v2/")
sys.path.insert(0, "/kaggle/working/PPO/airport_v2/")

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from airport_graph import build_graph
from flight_data import load_flights
from dijkstra_solver import run_dijkstra
from ppo_solver import run_ppo
from visualize import generate_full_report, generate_route_detail

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='PPO训练测试')
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                       help='训练轮数')
    args = parser.parse_args()

    print("=" * 60)
    print("  Hongqiao Airport Taxi Optimization")
    print("  PPO vs Dijkstra Comparison")
    print("=" * 60)

    # ── 1. 构建机场图 ─────────────────────────
    print("\n[Step 1] Building airport graph...")
    G, pos = build_graph("虹桥点.csv")

    # ── 2. 加载航班数据 ───────────────────────
    print("\n[Step 2] Loading flight data...")
    flights = load_flights(
        "出港航班0407.csv", "进港航班0407.csv",
        set(G.nodes()), pos
    )

    # ── 3. Dijkstra 基线 ──────────────────────
    print("\n[Step 3] Running Dijkstra baseline...")
    t0 = time.time()
    dijk_summary = run_dijkstra(G, flights, pos)
    print(f"  Dijkstra time: {time.time()-t0:.1f}s")

    # ── 4. PPO 训练 + 推理 ────────────────────
    print("\n[Step 4] PPO training and inference...")
    t1 = time.time()
    ppo_summary = run_ppo(
        G, flights, pos,
        total_timesteps=args.total_timesteps,
        force_retrain=True
    )
    print(f"  PPO total time: {time.time()-t1:.1f}s")

    # ── 5. 打印汇总对比 ───────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS COMPARISON")
    print("=" * 60)
    metrics_keys = [
        ('avg_time_s',     'Avg Time (s)'),
        ('avg_distance_m', 'Avg Distance (m)'),
        ('avg_co2_kg',     'Avg CO2 (kg)'),
        ('conflicts',      'Total Conflicts'),
    ]
    for k, label in metrics_keys:
        dv = dijk_summary[k]
        pv = ppo_summary[k]
        diff = (dv - pv) / max(abs(dv), 1e-9) * 100
        arrow = "↓" if diff > 0 else "↑"
        print(f"  {label:25s} | Dijkstra: {dv:10.2f} | PPO: {pv:10.2f} "
              f"| {arrow}{abs(diff):5.1f}%")

    # ── 6. 可视化 ─────────────────────────────
    print("\n[Step 5] Generating visualizations...")
    generate_full_report(dijk_summary, ppo_summary, G, pos,
                         output_path="/kaggle/working/PPO/airport_v2/taxi_comparison_report.png")
    generate_route_detail(dijk_summary, G, pos,
                          output_path="/kaggle/working/PPO/airport_v2/dijkstra_routes.png",
                          title_prefix="Dijkstra")
    generate_route_detail(ppo_summary, G, pos,
                          output_path="/kaggle/working/PPO/airport_v2/ppo_routes.png",
                          title_prefix="PPO")

    print("\n[Done] All outputs saved to /kaggle/working/PPO/airport_v2/")
    return dijk_summary, ppo_summary


if __name__ == "__main__":
    main()
