"""
PPO V2 - 针对车间调度的改进版本

改进点:
1. 奖励重塑: 用启发式引导 (reward shaping with EDD baseline)
2. 即时奖励增强: 每步给出更有意义的密集奖励
3. 动作空间裁剪: 过滤明显劣势动作，缩小有效候选集
4. 更大学习率 + 更多训练轮次
5. 归一化改善: 使用全局统计量归一化
"""
import torch
import torch.nn.functional as F
import numpy as np
from model import SchedulingAgent, build_hetero_graph
from environment import ShopFloorEnv

GAMMA = 0.995        # 更高折扣因子，更看重长期
LAMBDA = 0.97
CLIP_EPS = 0.15
ENTROPY_COEF = 0.02  # 增加探索
VF_COEF = 0.5
LR = 3e-4
MAX_GRAD_NORM = 1.0
PPO_EPOCHS = 3
SAMPLE_SIZE = 48


def get_global_feat(env, data):
    total_busy = sum(d.total_busy_time for d in env.devices.values())
    # 当前已拖期订单数
    n_late = sum(1 for o in env.orders.values()
                 if not o.completed and o.due < env.current_time)
    # 当前空闲设备比例
    n_idle = sum(1 for d in env.devices.values() if d.status == 'idle')
    n_dev = len(env.devices)
    return torch.tensor([
        min(total_busy / 2e7, 5.0),
        n_late / max(len(env.orders), 1),
        n_idle / n_dev,
        env.current_time / 3e5,  # 时间进度
    ], dtype=torch.float)


def compute_step_reward(env, order_id, device_id, op, data, prev_metrics):
    """
    密集即时奖励：
    - 拖期减少奖励
    - 设备利用率奖励
    - MTO优先奖励
    - 完成工序奖励
    """
    order = env.orders[order_id]
    dev = env.devices[device_id]

    reward = 0.0

    # 1. 优先级奖励: MTO订单优先
    priority_bonus = 0.3 if order.mode == 'MTO' else 0.1
    reward += priority_bonus

    # 2. 紧迫度奖励: 越紧迫越应该优先
    urgency = order.urgency(env.current_time, data['min_remaining'])
    if urgency > 0:  # 已经拖期或即将拖期
        reward += min(urgency / 86400, 1.0) * 0.5

    # 3. 换料惩罚
    changeover_fn = data['changeover'].get(op)
    if changeover_fn and dev.last_type is not None:
        co = changeover_fn(dev.last_type, order.product_type)
        reward -= min(co / 45.0, 1.0) * 0.2  # 换料时间越长，惩罚越大

    # 4. 设备专用性奖励: 优先用专用设备
    reward += dev.specialization() * 0.1

    # 归一化到 [-1, 1]
    reward = np.clip(reward / 2.0, -1, 1)
    return float(reward)


def filter_candidates(pairs, env, data, max_candidates=30):
    """
    启发式过滤: 只保留最有希望的候选(订单,设备)对
    优先考虑: MTO > 高紧迫度 > 短加工时间
    """
    if len(pairs) <= max_candidates:
        return pairs

    def score(p):
        oid, did, op = p
        order = env.orders[oid]
        dev = env.devices[did]
        urgency = order.urgency(env.current_time, data['min_remaining'])
        proc = data['proc_time'].get((did, order.product_type), 9999)
        s = 0.0
        s += order.priority * 10          # MTO优先
        s += min(urgency / 3600, 5)       # 紧迫度
        s -= proc * order.qty / 1e6       # 加工时间短优先
        s += dev.specialization()          # 专用设备
        return s

    sorted_pairs = sorted(pairs, key=score, reverse=True)
    return sorted_pairs[:max_candidates]


def run_episode(env, agent, data, greedy=False, max_decisions=400):
    agent.eval()
    trajectory = []
    decisions = 0
    prev_metrics = {'mto': env.total_tardiness('MTO'),
                    'mts': env.total_tardiness('MTS'),
                    'ms': env.makespan()}

    while decisions < max_decisions:
        pairs = env.get_schedulable_pairs()
        if not pairs:
            adv = env.advance_to_next_event()
            if not adv or env.is_terminal():
                break
            continue

        # 候选裁剪
        pairs = filter_candidates(pairs, env, data, max_candidates=30)

        graph, order_idx, device_idx = build_hetero_graph(env, data)
        global_feat = get_global_feat(env, data)

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

        # 计算步骤奖励 (在assign之前)
        step_reward = compute_step_reward(env, chosen[0], chosen[1], chosen[2],
                                          data, prev_metrics)

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

    # 终端奖励：与EDD启发式比较的相对改进量
    mto = env.total_tardiness('MTO')
    mts = env.total_tardiness('MTS')
    ms = env.makespan()

    # 参考基线（EDD启发式约: makespan=3910*60, tard=956*60）
    ref_ms = 3910 * 60
    ref_tard = 956 * 60
    curr_tard = 0.7 * mto + 0.3 * mts

    # 相对于基线的改进
    ms_improve = (ref_ms - ms) / ref_ms
    tard_improve = (ref_tard - curr_tard) / (ref_tard + 1)
    terminal_reward = 2.0 * (0.4 * ms_improve + 0.6 * tard_improve)
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

            ratio = torch.exp(new_log_prob - old_log_probs[i])
            adv = adv_tensor[i]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv
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

    return (total_loss/n_updates if n_updates else 0.0,
            total_entropy/n_updates if n_updates else 0.0)


def train(data, num_episodes=200):
    agent = SchedulingAgent(hidden=64)  # 稍大的网络
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)
    # Warmup + cosine decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_episodes, eta_min=1e-5)

    history = []
    best_tard = float('inf')
    best_ms = float('inf')
    print(f"开始训练，共 {num_episodes} 轮...")
    print(f"  EDD基线: makespan~3910min, 加权拖期~956min")
    print()

    for ep in range(num_episodes):
        env = ShopFloorEnv(data)
        env.reset()
        trajectory, ms, mto, mts = run_episode(env, agent, data)
        reward_sum = sum(t['reward'] for t in trajectory)
        loss, entropy = ppo_update(agent, optimizer, trajectory)
        scheduler.step()

        tard = (0.7*mto + 0.3*mts) / 60
        ms_min = ms / 60
        if tard < best_tard:
            best_tard = tard
        if ms_min < best_ms:
            best_ms = ms_min

        record = {
            'episode': ep+1,
            'reward': reward_sum,
            'makespan': ms_min,
            'mto_tardiness': mto/60,
            'mts_tardiness': mts/60,
            'total_tardiness': tard,
            'loss': loss,
            'entropy': entropy,
        }
        history.append(record)

        if (ep+1) % 10 == 0:
            print(f"  Ep {ep+1:4d} | MS: {ms_min:6.0f}min | "
                  f"MTO: {mto/60:6.0f} | MTS: {mts/60:6.0f} | "
                  f"Tard: {tard:6.0f} | Loss: {loss:.3f} | Ent: {entropy:.3f}")

    print(f"\n  最优 makespan: {best_ms:.0f}min (EDD基线: 3910min)")
    print(f"  最优 加权拖期: {best_tard:.0f}min (EDD基线: 956min)")
    return agent, history


def evaluate(data, agent, n_runs=20):
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
            'total_tardiness': (0.7*mto+0.3*mts)/60,
            'reward': -(0.7*mto+0.3*mts+ms)/60,
        })
        if i == 0:
            last_env = env
    return results, last_env
