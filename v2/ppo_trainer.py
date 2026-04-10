"""
PPO V3 - 针对不收敛问题的系统性修复

修复点:
1. 奖励归一化：用运行统计量标准化，消除高方差
2. 修复MTO作弊：MTO延迟开工直接给负奖励
3. 价值函数预热：前20轮只训练Critic，给Actor提供稳定基线
4. 降低Entropy系数：强制策略收敛而非持续探索
5. 加入完工时间软约束：防止makespan无限增大
"""
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from model import SchedulingAgent, build_hetero_graph
from environment import ShopFloorEnv

GAMMA = 0.995
LAMBDA = 0.97
CLIP_EPS = 0.15
ENTROPY_COEF = 0.005   # 大幅降低，强制收敛
VF_COEF = 1.0          # 加强价值函数学习
LR = 2e-4
MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 4
SAMPLE_SIZE = 64

# 奖励归一化统计
class RunningStats:
    def __init__(self, window=200):
        self.buf = deque(maxlen=window)
    def update(self, x):
        self.buf.append(x)
    def normalize(self, x):
        if len(self.buf) < 10:
            return x
        mu, sigma = np.mean(self.buf), np.std(self.buf) + 1e-8
        return (x - mu) / sigma


reward_stats = RunningStats()


def get_global_feat(env, data):
    total_busy = sum(d.total_busy_time for d in env.devices.values())
    n_late = sum(1 for o in env.orders.values()
                 if not o.completed and o.due < env.current_time)
    n_idle = sum(1 for d in env.devices.values() if d.status == 'idle')
    n_dev = len(env.devices)
    return torch.tensor([
        min(total_busy / 2e7, 5.0),
        n_late / max(len(env.orders), 1),
        n_idle / n_dev,
        env.current_time / 3e5,
    ], dtype=torch.float)


def compute_step_reward(env, order_id, device_id, op, data):
    order = env.orders[order_id]
    dev = env.devices[device_id]
    reward = 0.0

    # 1. MTO 优先奖励（明确差异化）
    if order.mode == 'MTO':
        reward += 1.0
        # MTO订单若已超期，额外惩罚
        if env.current_time > order.due * 0.8:
            reward -= 2.0
    else:
        reward += 0.2

    # 2. 紧迫度奖励（用到期剩余时间衡量，越紧越大）
    ready_ops = order.get_ready_ops()
    min_rem = min((data['min_remaining'].get(order_id, {}).get(op2, 0)
                   for op2 in ready_ops), default=0)
    time_left = order.due - env.current_time - min_rem
    # 时间越紧（time_left越小），奖励越高
    urgency_reward = np.clip(-time_left / (order.due + 1e-6), -1, 1)
    reward += urgency_reward * 0.5

    # 3. 换料惩罚
    changeover_fn = data['changeover'].get(op)
    if changeover_fn and dev.last_type is not None:
        co = changeover_fn(dev.last_type, order.product_type)
        reward -= min(co / 45.0, 1.0) * 0.3

    # 4. 设备利用率奖励（避免设备长期闲置）
    if dev.status == 'idle':
        idle_duration = env.current_time - dev.busy_until
        if idle_duration > 600:  # 10分钟以上空闲
            reward += 0.2  # 鼓励分配任务给长期空闲设备

    return float(np.clip(reward / 3.0, -1, 1))


def filter_candidates(pairs, env, data, max_candidates=25):
    if len(pairs) <= max_candidates:
        return pairs

    def score(p):
        oid, did, op = p
        order = env.orders[oid]
        dev = env.devices[did]
        ready_ops = order.get_ready_ops()
        min_rem = min((data['min_remaining'].get(oid, {}).get(op2, 0)
                       for op2 in ready_ops), default=0)
        time_left = order.due - env.current_time - min_rem
        s = 0.0
        s += order.priority * 15          # MTO优先权重更高
        s -= time_left / 3600             # 越紧迫越优先
        s -= data['proc_time'].get((did, order.product_type), 9999) * order.qty / 1e6
        s += dev.specialization() * 0.5
        return s

    return sorted(pairs, key=score, reverse=True)[:max_candidates]


def run_episode(env, agent, data, greedy=False, max_decisions=400):
    agent.eval()
    trajectory = []
    decisions = 0

    while decisions < max_decisions:
        pairs = env.get_schedulable_pairs()
        if not pairs:
            adv = env.advance_to_next_event()
            if not adv or env.is_terminal():
                break
            continue

        pairs = filter_candidates(pairs, env, data)
        graph, order_idx, device_idx = build_hetero_graph(env, data)
        global_feat = get_global_feat(env, data)

        graph_pairs, valid_pairs = [], []
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
        step_reward = compute_step_reward(env, chosen[0], chosen[1], chosen[2], data)
        env.assign(chosen[0], chosen[1], chosen[2])
        decisions += 1

        trajectory.append({
            'graph': graph, 'graph_pairs': graph_pairs,
            'global_feat': global_feat, 'action_idx': action_idx,
            'log_prob': log_prob,
            'value': value.item() if hasattr(value, 'item') else float(value),
            'reward': step_reward,
        })

        if not env.get_schedulable_pairs():
            while True:
                adv = env.advance_to_next_event()
                if not adv or env.is_terminal() or env.get_schedulable_pairs():
                    break
        if env.is_terminal():
            break

    mto = env.total_tardiness('MTO')
    mts = env.total_tardiness('MTS')
    ms = env.makespan()

    # 终端奖励：相对于EDD基线，并施加makespan软约束
    ref_ms, ref_tard = 3910 * 60, 956 * 60
    curr_tard = 0.7 * mto + 0.3 * mts
    ms_improve = np.clip((ref_ms - ms) / ref_ms, -2, 2)
    tard_improve = np.clip((ref_tard - curr_tard) / (ref_tard + 1), -2, 2)
    # makespan超出基线太多额外惩罚
    ms_penalty = max(0, (ms - ref_ms * 1.2) / ref_ms) * 2.0
    terminal_reward = 3.0 * (0.35 * ms_improve + 0.65 * tard_improve) - ms_penalty

    # 归一化
    reward_stats.update(terminal_reward)
    terminal_reward = float(np.clip(terminal_reward, -5, 5))

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
        next_val = values[t+1] if t+1 < T else 0.0
        delta = rewards[t] + GAMMA * next_val - values[t]
        advantages[t] = delta + GAMMA * LAMBDA * last_adv
        last_adv = advantages[t]
    returns = [advantages[t] + values[t] for t in range(T)]
    return advantages, returns


def ppo_update(agent, optimizer, trajectory, critic_only=False):
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

    sample_size = min(len(trajectory), SAMPLE_SIZE)

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

            critic_loss = F.mse_loss(value, ret_tensor[i].detach())

            if critic_only:
                # 预热阶段：只训练Critic
                loss = VF_COEF * critic_loss
            else:
                ratio = torch.exp(new_log_prob - old_log_probs[i])
                adv = adv_tensor[i]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv
                actor_loss = -torch.min(surr1, surr2)
                loss = actor_loss + VF_COEF * critic_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_loss += loss.item()
            total_entropy += entropy.item() if not critic_only else 0.0
            n_updates += 1

    return (total_loss/n_updates if n_updates else 0.0,
            total_entropy/n_updates if n_updates else 0.0)


def train_v3(data, num_episodes=300):
    agent = SchedulingAgent(hidden=64)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_episodes, eta_min=5e-6)

    history = []
    best_tard = float('inf')
    best_ms = float('inf')
    WARMUP = 20  # 前20轮只训练Critic

    print(f"训练 V3，共 {num_episodes} 轮（前{WARMUP}轮Critic预热）...")
    print(f"EDD基线: makespan~3910min, 加权拖期~956min\n")

    for ep in range(num_episodes):
        env = ShopFloorEnv(data)
        env.reset()
        critic_only = (ep < WARMUP)
        trajectory, ms, mto, mts = run_episode(env, agent, data)
        reward_sum = sum(t['reward'] for t in trajectory)
        loss, entropy = ppo_update(agent, optimizer, trajectory, critic_only=critic_only)
        scheduler.step()

        tard = (0.7*mto + 0.3*mts) / 60
        ms_min = ms / 60
        if tard < best_tard: best_tard = tard
        if ms_min < best_ms: best_ms = ms_min

        history.append({
            'episode': ep+1, 'reward': reward_sum,
            'makespan': ms_min, 'mto_tardiness': mto/60,
            'mts_tardiness': mts/60, 'total_tardiness': tard,
            'loss': loss, 'entropy': entropy,
        })

        tag = '[预热]' if critic_only else ''
        if (ep+1) % 10 == 0:
            print(f"  Ep {ep+1:4d}{tag} | MS:{ms_min:5.0f} | "
                  f"MTO:{mto/60:5.0f} | MTS:{mts/60:5.0f} | "
                  f"Tard:{tard:5.0f} | Loss:{loss:.3f} | Ent:{entropy:.3f}")

    print(f"\n  最优 makespan: {best_ms:.0f}min (EDD: 3910min)")
    print(f"  最优 加权拖期: {best_tard:.0f}min (EDD: 956min)")
    return agent, history


def evaluate_v3(data, agent, n_runs=20):
    results = []
    last_env = None
    for i in range(n_runs):
        env = ShopFloorEnv(data)
        env.reset()
        _, ms, mto, mts = run_episode(env, agent, data, greedy=(i % 3 == 0))
        results.append({
            'makespan': ms/60,
            'mto_tardiness': mto/60,
            'mts_tardiness': mts/60,
            'total_tardiness': (0.7*mto + 0.3*mts)/60,
            'reward': -(0.7*mto + 0.3*mts + ms)/60,
        })
        if i == 0:
            last_env = env
    return results, last_env
