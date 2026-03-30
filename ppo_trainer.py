"""
PPO 训练器：双智能体协同，事件驱动调度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

from environment import ShopFloorEnv
from networks import (
    OrderSelectionNet, MachineAssignmentNet,
    compute_order_heuristics, compute_machine_heuristics, state_to_tensors,
    DEVICE
)

# ─── PPO 超参数 ─────────────────────────────────────────────────────
LR           = 3e-4
GAMMA        = 0.99
GAE_LAMBDA   = 0.95
CLIP_EPS     = 0.2
ENTROPY_COEF = 0.005
VF_COEF      = 0.5
MAX_GRAD_NORM= 0.5
PPO_EPOCHS   = 2
MINI_BATCH   = 128
COLLECT_INTERVAL = 8   # 每收集4个episode的轨迹后做一次PPO更新


class PPOTrainer:
    def __init__(self, data):
        self.data = data
        self.order_net   = OrderSelectionNet().to(DEVICE)
        self.machine_net = MachineAssignmentNet().to(DEVICE)

        self.opt_order   = optim.Adam(self.order_net.parameters(),   lr=LR)
        self.opt_machine = optim.Adam(self.machine_net.parameters(), lr=LR)

        # 记录训练曲线
        self.history = {
            'reward': [], 'makespan': [], 'mto_delay': [], 'mts_delay': [],
            'loss_order': [], 'loss_machine': [],
            'entropy_order': [], 'entropy_machine': [],
        }

    # ──────────────────────────────────────────────────────────────
    def _run_episode(self, collect=True):
        """
        运行一个完整调度 episode。
        collect=True 时收集轨迹用于训练。
        返回 (reward, makespan, mto_delay, mts_delay, trajectories)
        """
        env = ShopFloorEnv(self.data)
        state = env.reset()

        # 轨迹缓冲区
        traj_order   = []   # (of, df, gf, cand_idx, h_scores, action, log_prob, value)
        traj_machine = []

        while not env.is_done():
            schedulable = env.get_schedulable()

            if not schedulable:
                # 没有可调度任务，推进时间
                t = env.advance_time()
                if t is None:
                    break
                state = env._get_state()
                continue

            # 事件驱动决策循环
            while schedulable:
                of, df, gf = state_to_tensors(state)

                # ── 订单选择 ──
                h_order = compute_order_heuristics(env, schedulable)
                cand_order_idx = list(range(len(schedulable)))

                with torch.no_grad() if not collect else torch.enable_grad():
                    logits_o, val_o = self.order_net(
                        of, df, gf,
                        [s[0] for s in schedulable],
                        h_order
                    )

                dist_o = Categorical(logits=logits_o)
                action_o = dist_o.sample()
                logp_o   = dist_o.log_prob(action_o)

                chosen_order_idx, chosen_stage = schedulable[action_o.item()]

                if collect:
                    traj_order.append({
                        'of': of.detach(), 'df': df.detach(), 'gf': gf.detach(),
                        'cand_indices': [s[0] for s in schedulable],
                        'h_scores': h_order.detach(),
                        'action': action_o.detach(),
                        'log_prob': logp_o.detach(),
                        'value': val_o.detach(),
                    })

                # ── 设备分配 ──
                idle_devices = env.get_compatible_idle_devices(chosen_order_idx, chosen_stage)
                if not idle_devices:
                    schedulable = env.get_schedulable()
                    state = env._get_state()
                    continue

                h_machine = compute_machine_heuristics(
                    env, chosen_order_idx, chosen_stage, idle_devices
                )
                dev_cand_idx = [env.device_index[d] for d in idle_devices]

                with torch.no_grad() if not collect else torch.enable_grad():
                    logits_m, val_m = self.machine_net(
                        of, df, gf, dev_cand_idx, h_machine
                    )

                dist_m   = Categorical(logits=logits_m)
                action_m = dist_m.sample()
                logp_m   = dist_m.log_prob(action_m)
                chosen_device = idle_devices[action_m.item()]

                if collect:
                    traj_machine.append({
                        'of': of.detach(), 'df': df.detach(), 'gf': gf.detach(),
                        'cand_indices': dev_cand_idx,
                        'h_scores': h_machine.detach(),
                        'action': action_m.detach(),
                        'log_prob': logp_m.detach(),
                        'value': val_m.detach(),
                    })

                # ── 执行调度 ──
                env.dispatch(chosen_order_idx, chosen_stage, chosen_device)
                state = env._get_state()
                schedulable = env.get_schedulable()

            # 推进时间
            t = env.advance_time()
            if t is None:
                break
            state = env._get_state()

        reward, makespan, mto_delay, mts_delay = env.compute_reward()
        return reward, makespan, mto_delay, mts_delay, traj_order, traj_machine, env

    # ──────────────────────────────────────────────────────────────
    def _compute_returns(self, trajectory, final_reward):
        """使用 GAE 计算优势和回报（终端奖励）。"""
        n = len(trajectory)
        if n == 0:
            return [], []
        rewards_list = [0.0] * n
        rewards_list[-1] = final_reward   # 仅终端有奖励
        values = [t['value'].item() for t in trajectory]

        advantages = [0.0] * n
        gae = 0.0
        for t in reversed(range(n)):
            next_val = values[t+1] if t+1 < n else 0.0
            delta = rewards_list[t] + GAMMA * next_val - values[t]
            gae   = delta + GAMMA * GAE_LAMBDA * gae
            advantages[t] = gae

        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    # ──────────────────────────────────────────────────────────────
    def _ppo_update(self, net, optimizer, trajectory, advantages, returns, is_order):
        """PPO 更新（全局归一化优势，批量前向加速）。"""
        if not trajectory:
            return 0.0, 0.0

        n = len(trajectory)
        indices = np.arange(n)

        adv_arr  = np.array(advantages, dtype=np.float32)
        ret_arr  = np.array(returns,    dtype=np.float32)
        adv_norm = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)
        ret_norm = (ret_arr - ret_arr.mean()) / (ret_arr.std() + 1e-8)

        total_loss = 0.0
        total_entr = 0.0
        n_updates  = 0

        for _ in range(PPO_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, n, MINI_BATCH):
                batch_idx = indices[start:start + MINI_BATCH]
                pg_losses, vf_losses, entropies = [], [], []

                for bi in batch_idx:
                    t   = trajectory[bi]
                    adv = torch.tensor(float(adv_norm[bi]), dtype=torch.float32, device=DEVICE)
                    ret = torch.tensor(float(ret_norm[bi]), dtype=torch.float32, device=DEVICE)

                    logits, value = net(t['of'], t['df'], t['gf'],
                                       t['cand_indices'], t['h_scores'])
                    dist    = Categorical(logits=logits)
                    new_lp  = dist.log_prob(t['action'])
                    entropy = dist.entropy()

                    ratio  = torch.exp(new_lp - t['log_prob'])
                    surr1  = ratio * adv
                    surr2  = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
                    pg_losses.append(-torch.min(surr1, surr2))
                    vf_losses.append(F.mse_loss(value, ret))
                    entropies.append(entropy)

                pg_loss = torch.stack(pg_losses).mean()
                vf_loss = torch.stack(vf_losses).mean()
                ent     = torch.stack(entropies).mean()
                loss    = pg_loss + VF_COEF * vf_loss - ENTROPY_COEF * ent

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                total_loss += loss.item()
                total_entr += ent.item()
                n_updates  += 1

        return total_loss / max(n_updates, 1), total_entr / max(n_updates, 1)

    # ──────────────────────────────────────────────────────────────
    def train(self, n_episodes=300, log_interval=10):
        print(f"训练开始，共 {n_episodes} 个 episode，使用设备: {DEVICE}")
        best_reward = -float('inf')
        best_env    = None

        buf_traj_o, buf_traj_m = [], []
        buf_adv_o,  buf_adv_m  = [], []
        buf_ret_o,  buf_ret_m  = [], []

        for ep in range(1, n_episodes + 1):
            reward, makespan, mto_d, mts_d, traj_o, traj_m, env = self._run_episode()

            adv_o, ret_o = self._compute_returns(traj_o, reward)
            adv_m, ret_m = self._compute_returns(traj_m, reward)

            buf_traj_o.extend(traj_o); buf_adv_o.extend(adv_o); buf_ret_o.extend(ret_o)
            buf_traj_m.extend(traj_m); buf_adv_m.extend(adv_m); buf_ret_m.extend(ret_m)

            loss_o = loss_m = entr_o = entr_m = 0.0
            if ep % COLLECT_INTERVAL == 0:
                loss_o, entr_o = self._ppo_update(
                    self.order_net,   self.opt_order,   buf_traj_o, buf_adv_o, buf_ret_o, True)
                loss_m, entr_m = self._ppo_update(
                    self.machine_net, self.opt_machine, buf_traj_m, buf_adv_m, buf_ret_m, False)
                buf_traj_o.clear(); buf_adv_o.clear(); buf_ret_o.clear()
                buf_traj_m.clear(); buf_adv_m.clear(); buf_ret_m.clear()

            self.history['reward'].append(reward)
            self.history['makespan'].append(makespan / 60)
            self.history['mto_delay'].append(mto_d / 60)
            self.history['mts_delay'].append(mts_d / 60)
            self.history['loss_order'].append(loss_o)
            self.history['loss_machine'].append(loss_m)
            self.history['entropy_order'].append(entr_o)
            self.history['entropy_machine'].append(entr_m)

            if reward > best_reward:
                best_reward = reward
                best_env    = env

            if ep % log_interval == 0:
                avg_r  = np.mean(self.history['reward'][-log_interval:])
                avg_mk = np.mean(self.history['makespan'][-log_interval:])
                avg_lo = np.mean([v for v in self.history['loss_order'][-log_interval:] if v != 0])  or 0
                avg_eo = np.mean([v for v in self.history['entropy_order'][-log_interval:] if v != 0]) or 0
                print(f"Ep {ep:4d} | Reward={avg_r:10.1f} | Makespan={avg_mk:.1f}min | "
                      f"MTO_delay={np.mean(self.history['mto_delay'][-log_interval:]):.1f}min | "
                      f"Loss={avg_lo:.4f} | Entr={avg_eo:.3f}")

        print(f"\n训练完成！最佳奖励={best_reward:.1f}")
        return best_env

    # ──────────────────────────────────────────────────────────────
    def evaluate(self, n_runs=20):
        """评估阶段：运行多次，收集指标用于箱线图。"""
        results = []
        for _ in range(n_runs):
            reward, makespan, mto_d, mts_d, _, _, env = self._run_episode(collect=False)
            results.append({
                'reward': reward,
                'makespan': makespan / 60,
                'mto_delay': mto_d / 60,
                'mts_delay': mts_d / 60,
                'env': env
            })
        return results
