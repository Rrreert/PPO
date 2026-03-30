"""
PPO 训练器（优化版）
优化项：
  1. 并行环境采样：multiprocessing.Pool 同时跑 N_ENVS 个 episode
  2. 增大 batch size：MINI_BATCH=512，COLLECT_INTERVAL=16
  3. 混合精度训练（AMP）：autocast + GradScaler
  4. 向量化前向：由 networks.py 内部保证
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
import numpy as np
import multiprocessing as mp
from torch.distributions import Categorical

from environment import ShopFloorEnv
from networks import (
    OrderSelectionNet, MachineAssignmentNet,
    compute_order_heuristics, compute_machine_heuristics,
    state_to_tensors, DEVICE,
)

# ── PPO 超参数 ────────────────────────────────────────────────────
LR               = 3e-4
GAMMA            = 0.99
GAE_LAMBDA       = 0.95
CLIP_EPS         = 0.2
ENTROPY_COEF     = 0.005   # 训练后期降低随机探索
VF_COEF          = 0.5
MAX_GRAD_NORM    = 0.5
PPO_EPOCHS       = 2
MINI_BATCH       = 512     # 128 → 512，充分利用 GPU 带宽
COLLECT_INTERVAL = 16      # 积累更多轨迹后再统一更新

# ── 并行环境数量 ──────────────────────────────────────────────────
# 自动设为 CPU 核数的一半（至少 1），避免争抢资源
N_ENVS = max(1, mp.cpu_count() // 2)


# ══════════════════════════════════════════════════════════════════
#  Worker 函数（运行在子进程，不依赖 CUDA）
# ══════════════════════════════════════════════════════════════════
def _worker_run_episode(args):
    """
    子进程中运行单个 episode（CPU 推理）。
    网络权重通过 state_dict 传入，避免 CUDA tensor 无法 pickle 的问题。
    """
    order_sd, machine_sd, data, seed = args
    torch.manual_seed(seed)

    # 在子进程中重建网络（CPU）
    _dev = torch.device('cpu')
    order_net   = OrderSelectionNet().to(_dev)
    machine_net = MachineAssignmentNet().to(_dev)
    order_net.load_state_dict(order_sd)
    machine_net.load_state_dict(machine_sd)
    order_net.eval()
    machine_net.eval()

    env   = ShopFloorEnv(data)
    state = env.reset()

    traj_order   = []
    traj_machine = []

    def _to_cpu(state):
        of = torch.as_tensor(state['order_feat'],  dtype=torch.float32).unsqueeze(0)
        df = torch.as_tensor(state['device_feat'], dtype=torch.float32).unsqueeze(0)
        gf = torch.as_tensor(state['global_feat'], dtype=torch.float32).unsqueeze(0)
        return of, df, gf

    while not env.is_done():
        schedulable = env.get_schedulable()
        if not schedulable:
            t = env.advance_time()
            if t is None:
                break
            state = env._get_state()
            continue

        while schedulable:
            of, df, gf = _to_cpu(state)
            h_order = compute_order_heuristics(env, schedulable)
            h_order = h_order.to(_dev)

            with torch.no_grad():
                logits_o, val_o = order_net(
                    of, df, gf, [s[0] for s in schedulable], h_order)

            dist_o   = Categorical(logits=logits_o)
            action_o = dist_o.sample()
            logp_o   = dist_o.log_prob(action_o)

            chosen_order_idx, chosen_stage = schedulable[action_o.item()]

            traj_order.append({
                'of': of, 'df': df, 'gf': gf,
                'cand_indices': [s[0] for s in schedulable],
                'h_scores': h_order,
                'action':   action_o,
                'log_prob': logp_o.detach(),
                'value':    val_o.detach(),
            })

            idle_devices = env.get_compatible_idle_devices(
                chosen_order_idx, chosen_stage)
            if not idle_devices:
                schedulable = env.get_schedulable()
                state = env._get_state()
                continue

            h_machine = compute_machine_heuristics(
                env, chosen_order_idx, chosen_stage, idle_devices)
            h_machine = h_machine.to(_dev)
            dev_cand  = [env.device_index[d] for d in idle_devices]

            with torch.no_grad():
                logits_m, val_m = machine_net(
                    of, df, gf, dev_cand, h_machine)

            dist_m   = Categorical(logits=logits_m)
            action_m = dist_m.sample()
            logp_m   = dist_m.log_prob(action_m)

            traj_machine.append({
                'of': of, 'df': df, 'gf': gf,
                'cand_indices': dev_cand,
                'h_scores': h_machine,
                'action':   action_m,
                'log_prob': logp_m.detach(),
                'value':    val_m.detach(),
            })

            env.dispatch(chosen_order_idx, chosen_stage,
                         idle_devices[action_m.item()])
            state      = env._get_state()
            schedulable = env.get_schedulable()

        t = env.advance_time()
        if t is None:
            break
        state = env._get_state()

    reward, makespan, mto_d, mts_d = env.compute_reward()
    return reward, makespan, mto_d, mts_d, traj_order, traj_machine, env


# ══════════════════════════════════════════════════════════════════
#  PPO 训练器
# ══════════════════════════════════════════════════════════════════
class PPOTrainer:
    def __init__(self, data):
        self.data        = data
        self.order_net   = OrderSelectionNet().to(DEVICE)
        self.machine_net = MachineAssignmentNet().to(DEVICE)

        self.opt_order   = optim.Adam(self.order_net.parameters(),   lr=LR)
        self.opt_machine = optim.Adam(self.machine_net.parameters(), lr=LR)

        # ── 混合精度 GradScaler（仅 CUDA 下有效）──────────────────
        self.scaler_order   = GradScaler('cuda', enabled=DEVICE.type == 'cuda')
        self.scaler_machine = GradScaler('cuda', enabled=DEVICE.type == 'cuda')

        self.history = {
            'reward': [], 'makespan': [], 'mto_delay': [], 'mts_delay': [],
            'loss_order': [], 'loss_machine': [],
            'entropy_order': [], 'entropy_machine': [],
        }

        print(f"设备: {DEVICE} | 并行环境数: {N_ENVS} | "
              f"HIDDEN={256} | MINI_BATCH={MINI_BATCH} | "
              f"COLLECT_INTERVAL={COLLECT_INTERVAL}")

    # ── 串行 episode（评估 / 单机测试用）─────────────────────────
    def _run_episode_serial(self, collect=True):
        env   = ShopFloorEnv(self.data)
        state = env.reset()
        traj_order   = []
        traj_machine = []

        while not env.is_done():
            schedulable = env.get_schedulable()
            if not schedulable:
                t = env.advance_time()
                if t is None:
                    break
                state = env._get_state()
                continue

            while schedulable:
                of, df, gf = state_to_tensors(state)
                h_order    = compute_order_heuristics(env, schedulable)

                ctx = autocast(device_type=DEVICE.type,
                               enabled=DEVICE.type == 'cuda')
                with ctx:
                    logits_o, val_o = self.order_net(
                        of, df, gf, [s[0] for s in schedulable], h_order)

                dist_o   = Categorical(logits=logits_o)
                action_o = dist_o.sample()
                logp_o   = dist_o.log_prob(action_o)
                chosen_order_idx, chosen_stage = schedulable[action_o.item()]

                if collect:
                    traj_order.append({
                        'of': of.detach(), 'df': df.detach(), 'gf': gf.detach(),
                        'cand_indices': [s[0] for s in schedulable],
                        'h_scores': h_order.detach(),
                        'action':   action_o.detach(),
                        'log_prob': logp_o.detach(),
                        'value':    val_o.detach(),
                    })

                idle_devices = env.get_compatible_idle_devices(
                    chosen_order_idx, chosen_stage)
                if not idle_devices:
                    schedulable = env.get_schedulable()
                    state = env._get_state()
                    continue

                h_machine = compute_machine_heuristics(
                    env, chosen_order_idx, chosen_stage, idle_devices)
                dev_cand  = [env.device_index[d] for d in idle_devices]

                with ctx:
                    logits_m, val_m = self.machine_net(
                        of, df, gf, dev_cand, h_machine)

                dist_m   = Categorical(logits=logits_m)
                action_m = dist_m.sample()
                logp_m   = dist_m.log_prob(action_m)

                if collect:
                    traj_machine.append({
                        'of': of.detach(), 'df': df.detach(), 'gf': gf.detach(),
                        'cand_indices': dev_cand,
                        'h_scores': h_machine.detach(),
                        'action':   action_m.detach(),
                        'log_prob': logp_m.detach(),
                        'value':    val_m.detach(),
                    })

                env.dispatch(chosen_order_idx, chosen_stage,
                             idle_devices[action_m.item()])
                state      = env._get_state()
                schedulable = env.get_schedulable()

            t = env.advance_time()
            if t is None:
                break
            state = env._get_state()

        reward, makespan, mto_d, mts_d = env.compute_reward()
        return reward, makespan, mto_d, mts_d, traj_order, traj_machine, env

    # ── 并行 episode 采样 ─────────────────────────────────────────
    def _run_episodes_parallel(self, n_envs, base_seed):
        """
        用 multiprocessing.Pool 并行运行 n_envs 个 episode。
        网络权重转为 CPU state_dict 传入子进程。
        """
        # 主进程网络移到 CPU 做 state_dict（避免 CUDA tensor pickle 错误）
        order_sd   = {k: v.cpu() for k, v in
                      self.order_net.state_dict().items()}
        machine_sd = {k: v.cpu() for k, v in
                      self.machine_net.state_dict().items()}

        args = [
            (order_sd, machine_sd, self.data, base_seed + i)
            for i in range(n_envs)
        ]

        # spawn 模式避免 CUDA fork 问题
        ctx = mp.get_context('spawn')
        with ctx.Pool(n_envs) as pool:
            results = pool.map(_worker_run_episode, args)

        return results

    # ── GAE 优势估计 ──────────────────────────────────────────────
    def _compute_returns(self, trajectory, final_reward):
        n = len(trajectory)
        if n == 0:
            return [], []
        rewards = [0.0] * n
        rewards[-1] = final_reward
        values = [t['value'].item() for t in trajectory]

        advantages = [0.0] * n
        gae = 0.0
        for t in reversed(range(n)):
            next_val = values[t + 1] if t + 1 < n else 0.0
            delta    = rewards[t] + GAMMA * next_val - values[t]
            gae      = delta + GAMMA * GAE_LAMBDA * gae
            advantages[t] = gae

        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    # ── PPO 更新（AMP + 大 batch）────────────────────────────────
    def _ppo_update(self, net, optimizer, scaler, trajectory,
                    advantages, returns):
        if not trajectory:
            return 0.0, 0.0

        n        = len(trajectory)
        indices  = np.arange(n)
        adv_arr  = np.array(advantages, dtype=np.float32)
        ret_arr  = np.array(returns,    dtype=np.float32)
        adv_norm = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)
        ret_norm = (ret_arr - ret_arr.mean()) / (ret_arr.std() + 1e-8)

        total_loss = 0.0
        total_entr = 0.0
        n_updates  = 0

        use_amp = DEVICE.type == 'cuda'

        for _ in range(PPO_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, n, MINI_BATCH):
                batch_idx = indices[start:start + MINI_BATCH]
                pg_losses, vf_losses, entropies = [], [], []

                adv_tensor = torch.tensor(adv_norm, device=DEVICE)
                ret_tensor = torch.tensor(ret_norm, device=DEVICE)
                for bi in batch_idx:
                    t   = trajectory[bi]
                    adv = adv_tensor[bi]
                    ret = ret_tensor[bi]

                    # ── 混合精度前向 ──────────────────────────────
                    with autocast(DEVICE.type, enabled=use_amp):
                        logits, value = net(
                            t['of'].to(DEVICE), t['df'].to(DEVICE),
                            t['gf'].to(DEVICE),
                            t['cand_indices'], t['h_scores'].to(DEVICE))

                        dist    = Categorical(logits=logits)
                        new_lp  = dist.log_prob(t['action'].to(DEVICE))
                        entropy = dist.entropy()

                        ratio = torch.exp(new_lp - t['log_prob'].to(DEVICE))
                        surr1 = ratio * adv
                        surr2 = torch.clamp(ratio, 1 - CLIP_EPS,
                                            1 + CLIP_EPS) * adv
                        pg_losses.append(-torch.min(surr1, surr2))
                        vf_losses.append(F.mse_loss(value, ret))
                        entropies.append(entropy)

                pg_loss = torch.stack(pg_losses).mean()
                vf_loss = torch.stack(vf_losses).mean()
                ent     = torch.stack(entropies).mean()
                loss    = pg_loss + VF_COEF * vf_loss - ENTROPY_COEF * ent

                optimizer.zero_grad()
                # ── GradScaler 反向（AMP）────────────────────────
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                total_entr += ent.item()
                n_updates  += 1

        return total_loss / max(n_updates, 1), total_entr / max(n_updates, 1)

    # ── 主训练循环 ────────────────────────────────────────────────
    def train(self, n_episodes=3000, log_interval=50):
        print(f"训练开始，共 {n_episodes} 个 episode")
        best_reward = -float('inf')
        best_env    = None

        buf_traj_o, buf_traj_m = [], []
        buf_adv_o,  buf_adv_m  = [], []
        buf_ret_o,  buf_ret_m  = [], []

        ep = 0
        seed_base = 0

        while ep < n_episodes:
            # ── 并行采样 N_ENVS 个 episode ────────────────────────
            remaining = n_episodes - ep
            n_this    = min(N_ENVS, remaining)

            if n_this > 1:
                results = self._run_episodes_parallel(n_this, seed_base)
            else:
                r = self._run_episode_serial()
                results = [r]

            seed_base += n_this

            for (reward, makespan, mto_d, mts_d,
                 traj_o, traj_m, env) in results:

                adv_o, ret_o = self._compute_returns(traj_o, reward)
                adv_m, ret_m = self._compute_returns(traj_m, reward)

                buf_traj_o.extend(traj_o); buf_adv_o.extend(adv_o)
                buf_ret_o.extend(ret_o)
                buf_traj_m.extend(traj_m); buf_adv_m.extend(adv_m)
                buf_ret_m.extend(ret_m)

                self.history['reward'].append(reward)
                self.history['makespan'].append(makespan / 60)
                self.history['mto_delay'].append(mto_d / 60)
                self.history['mts_delay'].append(mts_d / 60)
                self.history['loss_order'].append(0.0)
                self.history['loss_machine'].append(0.0)
                self.history['entropy_order'].append(0.0)
                self.history['entropy_machine'].append(0.0)

                if reward > best_reward:
                    best_reward = reward
                    best_env    = env

                ep += 1

            # ── 积累够 COLLECT_INTERVAL 后统一更新 ────────────────
            if ep % COLLECT_INTERVAL == 0 or ep >= n_episodes:
                loss_o, entr_o = self._ppo_update(
                    self.order_net,   self.opt_order,
                    self.scaler_order,
                    buf_traj_o, buf_adv_o, buf_ret_o)
                loss_m, entr_m = self._ppo_update(
                    self.machine_net, self.opt_machine,
                    self.scaler_machine,
                    buf_traj_m, buf_adv_m, buf_ret_m)

                # 把本批次的 loss/entropy 回填到历史记录
                fill_start = max(0, ep - COLLECT_INTERVAL)
                for idx in range(fill_start, ep):
                    if idx < len(self.history['loss_order']):
                        self.history['loss_order'][idx]    = loss_o
                        self.history['loss_machine'][idx]  = loss_m
                        self.history['entropy_order'][idx] = entr_o
                        self.history['entropy_machine'][idx] = entr_m

                buf_traj_o.clear(); buf_adv_o.clear(); buf_ret_o.clear()
                buf_traj_m.clear(); buf_adv_m.clear(); buf_ret_m.clear()

            # ── 日志 ──────────────────────────────────────────────
            if ep % log_interval == 0:
                recent = self.history
                avg_r  = np.mean(recent['reward'][-log_interval:])
                avg_mk = np.mean(recent['makespan'][-log_interval:])
                avg_mto= np.mean(recent['mto_delay'][-log_interval:])
                lo_vals= [v for v in recent['loss_order'][-log_interval:] if v]
                eo_vals= [v for v in recent['entropy_order'][-log_interval:] if v]
                avg_lo = np.mean(lo_vals) if lo_vals else 0.0
                avg_eo = np.mean(eo_vals) if eo_vals else 0.0
                print(f"Ep {ep:5d} | Reward={avg_r:10.1f} | "
                      f"Makespan={avg_mk:.1f}min | "
                      f"MTO_delay={avg_mto:.1f}min | "
                      f"Loss={avg_lo:.4f} | Entr={avg_eo:.3f}")

        print(f"\n训练完成！最佳奖励={best_reward:.1f}")
        return best_env

    # ── 评估 ──────────────────────────────────────────────────────
    def evaluate(self, n_runs=30):
        results = []
        for i in range(n_runs):
            reward, makespan, mto_d, mts_d, _, _, env = \
                self._run_episode_serial(collect=False)
            results.append({
                'reward':    reward,
                'makespan':  makespan / 60,
                'mto_delay': mto_d / 60,
                'mts_delay': mts_d / 60,
                'env':       env,
            })
        return results
