"""优化版 PPO 训练循环"""
import torch
import torch.nn.functional as F
import numpy as np
from model import SchedulingAgent, build_hetero_graph
from environment import ShopFloorEnv

GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VF_COEF = 0.5
LR = 1e-3
MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 2


def get_global_feat(env):
    total_busy = sum(d.total_busy_time for d in env.devices.values())
    total_tardiness = env.total_tardiness()
    return torch.tensor([
        min(total_busy / 1e8, 10.0),
        min(total_tardiness / 1e8, 10.0),
    ], dtype=torch.float)


def run_episode(env, agent, data, greedy=False, device_str='cpu', max_decisions=300):
    agent.eval()
    trajectory = []
    prev_reward_val = env.reward_value()
    decisions = 0

    while decisions < max_decisions:
        pairs = env.get_schedulable_pairs()
        if not pairs:
            advanced = env.advance_to_next_event()
            if not advanced:
                break
            if env.is_terminal():
                break
            continue

        graph, order_idx, device_idx = build_hetero_graph(env, data)
        global_feat = get_global_feat(env)

        graph_pairs = []
        valid_pairs = []
        for (oid, did, op) in pairs:
            if oid in order_idx and did in device_idx:
                graph_pairs.append((order_idx[oid], device_idx[did]))
                valid_pairs.append((oid, did, op))

        if not graph_pairs:
            env.advance_to_next_event()
            continue

        with torch.no_grad():
            logits, value = agent(graph, graph_pairs, global_feat)

        if greedy:
            action_idx = logits.argmax().item()
            log_prob = torch.tensor(0.0)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action_idx = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action_idx))

        chosen = valid_pairs[action_idx]
        env.assign(chosen[0], chosen[1], chosen[2])
        decisions += 1

        curr_reward_val = env.reward_value()
        reward = (prev_reward_val - curr_reward_val) / 1e6
        prev_reward_val = curr_reward_val

        trajectory.append({
            'graph': graph, 'graph_pairs': graph_pairs,
            'global_feat': global_feat, 'action_idx': action_idx,
            'log_prob': log_prob,
            'value': value.item() if hasattr(value, 'item') else float(value),
            'reward': reward,
        })

        if not env.get_schedulable_pairs():
            while True:
                advanced = env.advance_to_next_event()
                if not advanced:
                    break
                if env.is_terminal() or env.get_schedulable_pairs():
                    break

        if env.is_terminal():
            break

    mto = env.total_tardiness('MTO')
    mts = env.total_tardiness('MTS')
    ms = env.makespan()
    terminal_reward = -(0.7 * mto + 0.3 * mts + ms) / 1e8
    if trajectory:
        trajectory[-1]['reward'] += terminal_reward

    return trajectory, ms, mto, mts


def compute_gae(trajectory):
    rewards = [t['reward'] for t in trajectory]
    values = [t['value'] for t in trajectory]
    T = len(rewards)
    advantages = [0.0] * T
    last_adv = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + GAMMA * next_val - values[t]
        advantages[t] = delta + GAMMA * LAMBDA * last_adv
        last_adv = advantages[t]
    returns = [advantages[t] + values[t] for t in range(T)]
    return advantages, returns


def ppo_update(agent, optimizer, trajectory):
    if len(trajectory) < 2:
        return 0.0, 0.0

    advantages, returns = compute_gae(trajectory)
    adv_tensor = torch.tensor(advantages, dtype=torch.float)
    adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
    ret_tensor = torch.tensor(returns, dtype=torch.float)
    old_log_probs = torch.stack([t['log_prob'] for t in trajectory]).detach()

    total_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    agent.train()
    sample_size = min(len(trajectory), 32)

    for epoch in range(PPO_EPOCHS):
        indices = np.random.choice(len(trajectory), sample_size, replace=False)
        for i in indices:
            t = trajectory[i]
            logits, value = agent(t['graph'], t['graph_pairs'], t['global_feat'])
            if len(logits) == 0:
                continue

            dist = torch.distributions.Categorical(logits=logits)
            new_log_prob = dist.log_prob(torch.tensor(t['action_idx']))
            entropy = dist.entropy()

            ratio = torch.exp(new_log_prob - old_log_probs[i])
            adv = adv_tensor[i]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = F.mse_loss(value, ret_tensor[i].detach())
            loss = actor_loss + VF_COEF * critic_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_loss += loss.item()
            total_entropy += entropy.item()
            n_updates += 1

    if n_updates == 0:
        return 0.0, 0.0
    return total_loss / n_updates, total_entropy / n_updates


def train(data, num_episodes=100, device_str='cpu'):
    agent = SchedulingAgent()
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)
    scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    history = []
    print(f"开始训练，共 {num_episodes} 轮...")

    for ep in range(num_episodes):
        env = ShopFloorEnv(data)
        env.reset()
        trajectory, ms, mto, mts = run_episode(env, agent, data, device_str=device_str)
        reward_sum = sum(t['reward'] for t in trajectory)
        loss, entropy = ppo_update(agent, optimizer, trajectory)
        scheduler_lr.step()

        record = {
            'episode': ep + 1,
            'reward': reward_sum,
            'makespan': ms / 60,
            'mto_tardiness': mto / 60,
            'mts_tardiness': mts / 60,
            'total_tardiness': (0.7 * mto + 0.3 * mts) / 60,
            'loss': loss,
            'entropy': entropy,
        }
        history.append(record)

        if (ep + 1) % 5 == 0:
            print(f"  Ep {ep+1:4d} | Makespan: {ms/60:7.0f}min | "
                  f"MTO: {mto/60:7.0f}min | MTS: {mts/60:7.0f}min | "
                  f"Loss: {loss:.4f} | Ent: {entropy:.3f}")

    return agent, history


def evaluate(data, agent, n_runs=20):
    results = []
    last_env = None
    for i in range(n_runs):
        env = ShopFloorEnv(data)
        env.reset()
        _, ms, mto, mts = run_episode(env, agent, data, greedy=(i % 2 == 0))
        results.append({
            'makespan': ms / 60,
            'mto_tardiness': mto / 60,
            'mts_tardiness': mts / 60,
            'total_tardiness': (0.7 * mto + 0.3 * mts) / 60,
            'reward': -(0.7 * mto + 0.3 * mts + ms) / 60,
        })
        if i == 0:
            last_env = env
    return results, last_env
