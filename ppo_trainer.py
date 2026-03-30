"""
PPO 训练模块
实现双智能体 PPO 训练流程：
- 事件驱动的 episode 采样
- GAE 优势估计
- 裁剪式 PPO 更新
- 完整的训练日志记录
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import defaultdict

from ppo_network import (
    OrderSelectionPPO, MachineAssignPPO,
    state_to_tensors,
    ORDER_FEAT_DIM, MACHINE_FEAT_DIM, GLOBAL_FEAT_DIM,
    ORDER_HEURISTIC_DIM, MACHINE_HEURISTIC_DIM
)
from environment import SchedulingEnv, STAGE_INDEX, STATUS_WAITING


# ─── PPO 超参数 ───────────────────────────────────────────────────────────────
class PPOConfig:
    lr_order = 3e-4
    lr_machine = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_eps = 0.2
    entropy_coef = 0.01
    value_coef = 0.5
    max_grad_norm = 0.5
    n_epochs = 4         # 每次更新的 epoch 数
    batch_size = 32
    n_episodes_per_update = 5   # 每次更新收集的 episode 数


# ─── 经验缓冲区 ───────────────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.order_transitions = []    # 订单选择智能体的转移
        self.machine_transitions = []  # 设备分配智能体的转移

    def add_order(self, order_feats, machine_mean, global_feat,
                  heuristic_scores, action_idx, log_prob, value, reward, done):
        self.order_transitions.append({
            'order_feats': order_feats,
            'machine_mean': machine_mean,
            'global_feat': global_feat,
            'heuristic_scores': heuristic_scores,
            'action_idx': action_idx,
            'log_prob': log_prob,
            'value': value,
            'reward': reward,
            'done': done,
        })

    def add_machine(self, order_feat, machine_feats, global_feat,
                    heuristic_scores, action_idx, log_prob, value, reward, done):
        self.machine_transitions.append({
            'order_feat': order_feat,
            'machine_feats': machine_feats,
            'global_feat': global_feat,
            'heuristic_scores': heuristic_scores,
            'action_idx': action_idx,
            'log_prob': log_prob,
            'value': value,
            'reward': reward,
            'done': done,
        })


def compute_gae(rewards, values, dones, gamma, gae_lambda):
    """计算 GAE 优势估计"""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        if dones[t]:
            next_val = 0.0
        else:
            next_val = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    return advantages, returns


# ─── 主训练器 ─────────────────────────────────────────────────────────────────
class PPOTrainer:
    def __init__(self, data, config=None, device=None):
        self.data = data
        self.config = config or PPOConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"训练设备: {self.device}")

        self.env = SchedulingEnv(data)

        self.order_agent = OrderSelectionPPO().to(self.device)
        self.machine_agent = MachineAssignPPO().to(self.device)

        self.order_optimizer = optim.Adam(self.order_agent.parameters(), lr=self.config.lr_order)
        self.machine_optimizer = optim.Adam(self.machine_agent.parameters(), lr=self.config.lr_machine)

        # 训练记录
        self.history = defaultdict(list)

    # ─── Episode 采样 ─────────────────────────────────────────────────────────
    def collect_episode(self):
        """运行一个完整 episode，收集轨迹"""
        state = self.env.reset()
        buffer = RolloutBuffer()
        total_decisions = 0
        max_steps = 10000  # 防死循环保护

        for _ in range(max_steps):
            if self.env.is_done():
                break

            available = self.env._get_available_orders()
            if not available:
                # 没有可调度订单，推进时间到下一事件
                self.env.step_to_next_event()
                state = self.env._get_state()
                continue

            tensors = state_to_tensors(state, self.device)
            global_feat = tensors['global_features']
            all_machine_feats = tensors['machine_features']
            machine_mean = all_machine_feats.mean(0)

            # ── 订单选择 ──────────────────────────────────────────────────────
            order_indices, stages = zip(*available)
            order_indices = list(order_indices)
            candidate_feats = tensors['order_features'][order_indices]  # [n_cand, feat]

            h_scores_list = [
                self.env.get_heuristic_scores_order(i) for i in order_indices
            ]
            h_scores = torch.FloatTensor(np.stack(h_scores_list)).to(self.device)

            with torch.no_grad():
                probs = self.order_agent.get_action_probs(
                    candidate_feats, machine_mean, global_feat, h_scores)
                value_o = self.order_agent.get_value(
                    candidate_feats, machine_mean, global_feat)

            dist = torch.distributions.Categorical(probs)
            action_o = dist.sample()
            log_prob_o = dist.log_prob(action_o)

            chosen_order_idx = order_indices[action_o.item()]
            chosen_stage = stages[action_o.item()]

            # ── 设备分配 ──────────────────────────────────────────────────────
            avail_machines = self.env._get_available_machines(chosen_order_idx, chosen_stage)
            order_feat = tensors['order_features'][chosen_order_idx]

            # 获取可用设备的特征（从全局设备矩阵中找对应行）
            machine_idx_map = {m_id: i for i, m_id in enumerate(self.env._machine_list)}
            m_feats_list = [
                self.env.machines[m].stage  # 仅用于索引
                for m in avail_machines
            ]
            m_feat_tensors = torch.stack([
                all_machine_feats[machine_idx_map[m]]
                for m in avail_machines
            ])  # [n_avail_m, MACHINE_FEAT_DIM]

            h_m_scores = torch.FloatTensor(
                self.env.get_heuristic_scores_machine(chosen_order_idx, avail_machines)
            ).to(self.device)

            with torch.no_grad():
                probs_m = self.machine_agent.get_action_probs(
                    order_feat, m_feat_tensors, global_feat, h_m_scores)
                value_m = self.machine_agent.get_value(
                    order_feat, m_feat_tensors, global_feat)

            dist_m = torch.distributions.Categorical(probs_m)
            action_m = dist_m.sample()
            log_prob_m = dist_m.log_prob(action_m)

            chosen_machine = avail_machines[action_m.item()]

            # ── 执行动作 ──────────────────────────────────────────────────────
            self.env.assign(chosen_order_idx, chosen_machine)
            total_decisions += 1

            # 中间奖励（稀疏，仅终端有值）
            step_reward = 0.0
            done = self.env.is_done()

            # 存入缓冲
            buffer.add_order(
                candidate_feats.cpu().numpy(),
                machine_mean.cpu().numpy(),
                global_feat.cpu().numpy(),
                h_scores.cpu().numpy(),
                action_o.item(), log_prob_o.item(),
                value_o.item(), step_reward, float(done)
            )
            buffer.add_machine(
                order_feat.cpu().numpy(),
                m_feat_tensors.cpu().numpy(),
                global_feat.cpu().numpy(),
                h_m_scores.cpu().numpy(),
                action_m.item(), log_prob_m.item(),
                value_m.item(), step_reward, float(done)
            )

            state = self.env._get_state()

            # 如果没有更多可调度订单，推进时间
            if not self.env._get_available_orders() and not self.env.is_done():
                self.env.step_to_next_event()
                state = self.env._get_state()

        # 终端奖励
        terminal_reward = self.env.compute_reward()
        if buffer.order_transitions:
            buffer.order_transitions[-1]['reward'] = terminal_reward
            buffer.order_transitions[-1]['done'] = 1.0
        if buffer.machine_transitions:
            buffer.machine_transitions[-1]['reward'] = terminal_reward
            buffer.machine_transitions[-1]['done'] = 1.0

        metrics = self.env.get_metrics()
        return buffer, metrics

    # ─── PPO 更新 ─────────────────────────────────────────────────────────────
    def _ppo_update_agent(self, agent, optimizer, transitions,
                          feat_key_list, action_key, is_order_agent):
        """通用 PPO 更新函数"""
        if not transitions:
            return 0.0, 0.0, 0.0

        rewards = np.array([t['reward'] for t in transitions])
        values = np.array([t['value'] for t in transitions])
        dones = np.array([t['done'] for t in transitions])
        advantages, returns = compute_gae(
            rewards, values, dones, self.config.gamma, self.config.gae_lambda)

        # 标准化优势
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.config.n_epochs):
            # 小批量随机更新
            indices = np.random.permutation(len(transitions))
            for start in range(0, len(indices), self.config.batch_size):
                batch_idx = indices[start: start + self.config.batch_size]

                for bi in batch_idx:
                    t = transitions[bi]
                    adv = torch.FloatTensor([advantages[bi]]).to(self.device)
                    ret = torch.FloatTensor([returns[bi]]).to(self.device)
                    old_log_prob = torch.FloatTensor([t['log_prob']]).to(self.device)
                    action = t[action_key]

                    global_feat = torch.FloatTensor(t['global_feat']).to(self.device)
                    h_scores = torch.FloatTensor(t['heuristic_scores']).to(self.device)

                    if is_order_agent:
                        order_feats = torch.FloatTensor(t['order_feats']).to(self.device)
                        machine_mean = torch.FloatTensor(t['machine_mean']).to(self.device)
                        probs = agent.get_action_probs(order_feats, machine_mean, global_feat, h_scores)
                        value = agent.get_value(order_feats, machine_mean, global_feat)
                    else:
                        order_feat = torch.FloatTensor(t['order_feat']).to(self.device)
                        machine_feats = torch.FloatTensor(t['machine_feats']).to(self.device)
                        probs = agent.get_action_probs(order_feat, machine_feats, global_feat, h_scores)
                        value = agent.get_value(order_feat, machine_feats, global_feat)

                    dist = torch.distributions.Categorical(probs)
                    new_log_prob = dist.log_prob(torch.tensor(action).to(self.device))
                    entropy = dist.entropy()

                    ratio = torch.exp(new_log_prob - old_log_prob)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(value.unsqueeze(0), ret)

                    loss = (policy_loss
                            + self.config.value_coef * value_loss
                            - self.config.entropy_coef * entropy)

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), self.config.max_grad_norm)
                    optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.item()
                    n_updates += 1

        if n_updates == 0:
            return 0.0, 0.0, 0.0
        return (total_policy_loss / n_updates,
                total_value_loss / n_updates,
                total_entropy / n_updates)

    # ─── 训练主循环 ───────────────────────────────────────────────────────────
    def train(self, n_iterations=100, log_interval=10):
        """
        主训练循环
        n_iterations: 更新迭代次数
        log_interval: 每隔多少次打印日志
        """
        print(f"\n{'='*60}")
        print(f"开始训练 | 迭代次数: {n_iterations} | 设备: {self.device}")
        print(f"{'='*60}\n")

        for iteration in range(1, n_iterations + 1):
            # 采集多个 episode
            all_order_trans = []
            all_machine_trans = []
            episode_rewards = []
            episode_makespans = []
            episode_delays = []

            for _ in range(self.config.n_episodes_per_update):
                buffer, metrics = self.collect_episode()
                all_order_trans.extend(buffer.order_transitions)
                all_machine_trans.extend(buffer.machine_transitions)
                episode_rewards.append(self.env.compute_reward())
                episode_makespans.append(metrics['makespan_min'])
                episode_delays.append(metrics['total_delay'] / 60.0)

            # PPO 更新
            o_ploss, o_vloss, o_ent = self._ppo_update_agent(
                self.order_agent, self.order_optimizer,
                all_order_trans, None, 'action_idx', is_order_agent=True)
            m_ploss, m_vloss, m_ent = self._ppo_update_agent(
                self.machine_agent, self.machine_optimizer,
                all_machine_trans, None, 'action_idx', is_order_agent=False)

            # 记录
            avg_reward = np.mean(episode_rewards)
            avg_makespan = np.mean(episode_makespans)
            avg_delay = np.mean(episode_delays)
            total_loss = o_ploss + o_vloss + m_ploss + m_vloss
            avg_entropy = (o_ent + m_ent) / 2

            self.history['reward'].append(avg_reward)
            self.history['makespan'].append(avg_makespan)
            self.history['total_delay'].append(avg_delay)
            self.history['loss'].append(total_loss)
            self.history['entropy'].append(avg_entropy)
            self.history['makespans_raw'].append(episode_makespans)
            self.history['delays_raw'].append(episode_delays)

            if iteration % log_interval == 0 or iteration == 1:
                print(f"Iter {iteration:4d}/{n_iterations} | "
                      f"奖励: {avg_reward:8.4f} | "
                      f"完工时间: {avg_makespan:8.1f}min | "
                      f"拖期: {avg_delay:8.1f}min | "
                      f"Loss: {total_loss:.4f} | "
                      f"Entropy: {avg_entropy:.4f}")

        print(f"\n训练完成！共 {n_iterations} 次迭代")
        return self.history

    def save_models(self, path_order='order_agent.pth', path_machine='machine_agent.pth'):
        torch.save(self.order_agent.state_dict(), path_order)
        torch.save(self.machine_agent.state_dict(), path_machine)
        print(f"模型已保存: {path_order}, {path_machine}")

    def load_models(self, path_order='order_agent.pth', path_machine='machine_agent.pth'):
        self.order_agent.load_state_dict(torch.load(path_order, map_location=self.device))
        self.machine_agent.load_state_dict(torch.load(path_machine, map_location=self.device))
        print("模型已加载")

    def run_inference(self):
        """运行一次推理，返回完整调度结果（贪心策略）"""
        state = self.env.reset()
        self.order_agent.eval()
        self.machine_agent.eval()

        with torch.no_grad():
            for _ in range(10000):
                if self.env.is_done():
                    break
                available = self.env._get_available_orders()
                if not available:
                    self.env.step_to_next_event()
                    state = self.env._get_state()
                    continue

                tensors = state_to_tensors(state, self.device)
                global_feat = tensors['global_features']
                all_machine_feats = tensors['machine_features']
                machine_mean = all_machine_feats.mean(0)

                order_indices, stages = zip(*available)
                order_indices = list(order_indices)
                candidate_feats = tensors['order_features'][order_indices]
                h_scores = torch.FloatTensor(np.stack([
                    self.env.get_heuristic_scores_order(i) for i in order_indices
                ])).to(self.device)

                probs = self.order_agent.get_action_probs(
                    candidate_feats, machine_mean, global_feat, h_scores)
                action_o = probs.argmax().item()  # 贪心

                chosen_order_idx = order_indices[action_o]
                chosen_stage = stages[action_o]

                avail_machines = self.env._get_available_machines(chosen_order_idx, chosen_stage)
                order_feat = tensors['order_features'][chosen_order_idx]
                machine_idx_map = {m_id: i for i, m_id in enumerate(self.env._machine_list)}
                m_feat_tensors = torch.stack([
                    all_machine_feats[machine_idx_map[m]] for m in avail_machines
                ])
                h_m = torch.FloatTensor(
                    self.env.get_heuristic_scores_machine(chosen_order_idx, avail_machines)
                ).to(self.device)

                probs_m = self.machine_agent.get_action_probs(
                    order_feat, m_feat_tensors, global_feat, h_m)
                action_m = probs_m.argmax().item()
                chosen_machine = avail_machines[action_m]

                self.env.assign(chosen_order_idx, chosen_machine)
                state = self.env._get_state()

                if not self.env._get_available_orders() and not self.env.is_done():
                    self.env.step_to_next_event()
                    state = self.env._get_state()

        self.order_agent.train()
        self.machine_agent.train()
        return self.env.schedule_history, self.env.get_metrics()


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/claude/workshop_scheduling')
    from data_loader import load_all_data

    data = load_all_data('/mnt/user-data/uploads/data.xlsx')
    trainer = PPOTrainer(data)

    # 测试单个 episode 采样
    print("测试 episode 采样...")
    buffer, metrics = trainer.collect_episode()
    print(f"  订单决策步数: {len(buffer.order_transitions)}")
    print(f"  设备决策步数: {len(buffer.machine_transitions)}")
    print(f"  完工时间: {metrics['makespan_min']:.1f} 分钟")
    print(f"  拖期总和: {metrics['total_delay']/60:.1f} 分钟")
    print("PPO 训练模块测试成功 ✓")
