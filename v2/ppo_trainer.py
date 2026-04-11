"""
PPO V4 - 快速修复版本

主要改进:
1. 重新设计reward function (对齐step和terminal reward)
2. 移除奖励归一化 (保持物理意义)
3. 取消Critic预热，改用经验回放
4. 调整超参数 (提高探索、学习率)
5. 增加详细监控指标
"""
import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from model import SchedulingAgent, build_hetero_graph
from environment import ShopFloorEnv

# === 优化后的超参数 ===
GAMMA = 0.995
LAMBDA = 0.97
CLIP_EPS = 0.2              # 提高到标准值
ENTROPY_COEF = 0.02         # 提高4倍，鼓励探索
VF_COEF = 0.5               # 降低，避免Value主导
LR = 5e-4                   # 提高学习率
MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 6              # 增加更新次数
SAMPLE_SIZE = 128           # 增大采样
REPLAY_BUFFER_SIZE = 30     # 经验回放池大小


def get_global_feat(env, data):
    """全局特征"""
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


def compute_step_reward_v2(env, order_id, device_id, op, data):
    """
    优化后的step reward:
    - 取消归一化，保持物理意义
    - 用指数函数放大对拖期的敏感度
    - 奖励高效设备选择
    """
    order = env.orders[order_id]
    dev = env.devices[device_id]
    reward = 0.0
    
    # 1. 基础奖励: 完成一个操作
    reward += 1.0
    
    # 2. 紧迫度感知(强化版)
    ready_ops = order.get_ready_ops()
    min_rem = min((data['min_remaining'].get(order_id, {}).get(op2, 0)
                   for op2 in ready_ops), default=0)
    time_left = order.due - env.current_time - min_rem
    
    # 关键改进: 用指数函数放大紧迫性
    if time_left < 0:  # 已经拖期了
        delay_hours = -time_left / 3600
        reward -= (np.exp(min(delay_hours / 24, 10)) - 1) * 2.0  # 拖期越多惩罚越重
    else:
        # 还没拖期，但越接近deadline奖励越高
        ratio = 1.0 - time_left / (order.due + 1e-6)
        urgency_bonus = np.exp(ratio * 3) - 1  # 指数增长
        if order.mode == 'MTO':
            reward += urgency_bonus * 0.8
        else:
            reward += urgency_bonus * 0.3
    
    # 3. 设备效率奖励
    proc_time = data['proc_time'].get((device_id, order.product_type), 9999)
    # 计算平均加工时间
    all_proc_times = [data['proc_time'].get((did, order.product_type), 9999) 
                      for did in env.devices.keys()]
    valid_times = [t for t in all_proc_times if t < 9999]
    if valid_times:
        avg_time = np.mean(valid_times)
        if proc_time < avg_time:
            reward += 0.8  # 选更快的设备给奖励
        elif proc_time > avg_time * 1.5:
            reward -= 0.5  # 选太慢的设备惩罚
    
    # 4. 换料惩罚
    changeover_fn = data['changeover'].get(op)
    if changeover_fn and dev.last_type is not None:
        co = changeover_fn(dev.last_type, order.product_type)
        reward -= min(co / 45.0, 1.0) * 0.8
    
    # 5. 设备均衡使用
    if dev.status == 'idle':
        idle_duration = env.current_time - dev.busy_until
        if idle_duration > 600:  # 10分钟以上空闲
            reward += 0.5  # 鼓励使用长期空闲设备
    
    # 不归一化，保持reward的绝对大小
    return float(np.clip(reward, -15, 15))


def compute_terminal_reward_v2(env, ref_ms=3910*60, ref_tard=956*60):
    """
    优化后的terminal reward:
    - 与step reward统一为"最小化代价"范式
    - 放大惩罚信号
    """
    mto = env.total_tardiness('MTO')
    mts = env.total_tardiness('MTS')
    ms = env.makespan()
    
    curr_tard = 0.7 * mto + 0.3 * mts
    
    # 归一化代价
    ms_cost = (ms / ref_ms)
    tard_cost = (curr_tard / (ref_tard + 1))
    
    # 总代价 = makespan代价 + 拖期代价(权重更高)
    total_cost = 0.35 * ms_cost + 0.65 * tard_cost
    
    # 转换为奖励: 代价越小奖励越高
    # 相对于基线的改进百分比 * 100
    improvement = (1.0 - total_cost) * 100.0
    
    # 额外惩罚: makespan或拖期显著超标
    penalty = 0.0
    if ms > ref_ms * 1.3:
        penalty += (ms / ref_ms - 1.3) * 50.0
    if curr_tard > ref_tard * 2.0:
        penalty += (curr_tard / ref_tard - 2.0) * 50.0
    
    reward = improvement - penalty
    
    return float(np.clip(reward, -100, 100))


def filter_candidates(pairs, env, data, max_candidates=40):
    """
    候选过滤策略 - 增加随机性
    """
    if len(pairs) <= max_candidates:
        return pairs
    
    # 策略1: 完全随机(exploration)
    if random.random() < 0.3:
        return random.sample(pairs, max_candidates)
    
    # 策略2: 启发式评分(exploitation)
    def score(p):
        oid, did, op = p
        order = env.orders[oid]
        dev = env.devices[did]
        ready_ops = order.get_ready_ops()
        min_rem = min((data['min_remaining'].get(oid, {}).get(op2, 0)
                       for op2 in ready_ops), default=0)
        time_left = order.due - env.current_time - min_rem
        s = 0.0
        s += order.priority * 20          # MTO优先
        s -= time_left / 3600             # 越紧迫越优先
        s -= data['proc_time'].get((did, order.product_type), 9999) * order.qty / 1e6
        s += dev.specialization() * 0.5
        return s
    
    scored_pairs = [(score(p), p) for p in pairs]
    scored_pairs.sort(reverse=True, key=lambda x: x[0])
    
    # Top-k采样: 前60%确定性选择，后40%随机采样
    n_top = int(max_candidates * 0.6)
    n_rand = max_candidates - n_top
    
    top_pairs = [p for _, p in scored_pairs[:n_top]]
    rand_pairs = random.sample([p for _, p in scored_pairs[n_top:]], 
                               min(n_rand, len(scored_pairs) - n_top))
    
    return top_pairs + rand_pairs


def run_episode(env, agent, data, greedy=False, max_decisions=400):
    """运行一个episode"""
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
        step_reward = compute_step_reward_v2(env, chosen[0], chosen[1], chosen[2], data)
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

    # 计算terminal reward
    terminal_reward = compute_terminal_reward_v2(env)

    if trajectory:
        trajectory[-1]['reward'] += terminal_reward

    return trajectory, ms, mto, mts, terminal_reward


def compute_gae(trajectory):
    """GAE优势函数估计"""
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


def ppo_update(agent, optimizer, trajectories):
    """
    PPO更新 - 支持多条轨迹(经验回放)
    """
    if not trajectories:
        return 0.0, 0.0, 0.0, 0.0
    
    all_data = []
    for trajectory in trajectories:
        if len(trajectory) < 2:
            continue
        
        advantages, returns = compute_gae(trajectory)
        for i, t in enumerate(trajectory):
            all_data.append({
                'graph': t['graph'],
                'graph_pairs': t['graph_pairs'],
                'global_feat': t['global_feat'],
                'action_idx': t['action_idx'],
                'old_log_prob': t['log_prob'],
                'advantage': advantages[i],
                'return': returns[i],
            })
    
    if len(all_data) < 2:
        return 0.0, 0.0, 0.0, 0.0
    
    # 标准化优势函数
    advantages_list = [d['advantage'] for d in all_data]
    adv_mean = np.mean(advantages_list)
    adv_std = np.std(advantages_list) + 1e-8
    for d in all_data:
        d['advantage'] = (d['advantage'] - adv_mean) / adv_std
    
    total_loss = 0.0
    total_actor_loss = 0.0
    total_critic_loss = 0.0
    total_entropy = 0.0
    n_updates = 0
    
    agent.train()
    
    sample_size = min(len(all_data), SAMPLE_SIZE)
    
    for epoch in range(PPO_EPOCHS):
        indices = np.random.choice(len(all_data), sample_size, replace=False)
        for i in indices:
            d = all_data[i]
            logits, value = agent(d['graph'], d['graph_pairs'], d['global_feat'])
            if len(logits) == 0:
                continue
            
            dist = torch.distributions.Categorical(logits=logits)
            new_log_prob = dist.log_prob(torch.tensor(d['action_idx']))
            entropy = dist.entropy()
            
            # Actor loss (PPO clip)
            ratio = torch.exp(new_log_prob - d['old_log_prob'])
            adv = torch.tensor(d['advantage'], dtype=torch.float)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv
            actor_loss = -torch.min(surr1, surr2)
            
            # Critic loss
            ret = torch.tensor(d['return'], dtype=torch.float)
            critic_loss = F.mse_loss(value, ret.detach())
            
            # Total loss
            loss = actor_loss + VF_COEF * critic_loss - ENTROPY_COEF * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            
            total_loss += loss.item()
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
            n_updates += 1
    
    if n_updates == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    return (total_loss / n_updates,
            total_actor_loss / n_updates,
            total_critic_loss / n_updates,
            total_entropy / n_updates)


def train(data, num_episodes=500):
    """训练主循环"""
    agent = SchedulingAgent(hidden=64)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_episodes, eta_min=1e-5)
    
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
    history = []
    best_tard = float('inf')
    best_ms = float('inf')
    
    print(f"训练 V4 (经验回放 + 优化奖励)，共 {num_episodes} 轮...")
    print(f"EDD基线: makespan~3910min, 加权拖期~956min")
    print(f"超参数: ENTROPY={ENTROPY_COEF}, CLIP={CLIP_EPS}, LR={LR}, "
          f"PPO_EPOCHS={PPO_EPOCHS}, REPLAY_SIZE={REPLAY_BUFFER_SIZE}\n")
    
    for ep in range(num_episodes):
        # 运行一个episode
        env = ShopFloorEnv(data)
        env.reset()
        trajectory, ms, mto, mts, terminal_reward = run_episode(env, agent, data)
        
        # 添加到回放池
        replay_buffer.append(trajectory)
        
        # 计算指标
        reward_sum = sum(t['reward'] for t in trajectory)
        tard = (0.7*mto + 0.3*mts) / 60
        ms_min = ms / 60
        
        if tard < best_tard: 
            best_tard = tard
        if ms_min < best_ms: 
            best_ms = ms_min
        
        # 经验回放训练
        if len(replay_buffer) >= 5:
            # 采样最近的轨迹进行训练
            sample_size = min(8, len(replay_buffer))
            sampled_trajs = random.sample(list(replay_buffer), sample_size)
            loss, actor_loss, critic_loss, entropy = ppo_update(agent, optimizer, sampled_trajs)
        else:
            # 前几轮直接用当前轨迹
            loss, actor_loss, critic_loss, entropy = ppo_update(agent, optimizer, [trajectory])
        
        scheduler.step()
        
        # 记录详细指标
        avg_step_reward = np.mean([t['reward'] for t in trajectory]) if trajectory else 0
        value_estimates = [t['value'] for t in trajectory]
        advantages, _ = compute_gae(trajectory)
        
        history.append({
            'episode': ep+1,
            'reward': reward_sum,
            'makespan': ms_min,
            'mto_tardiness': mto/60,
            'mts_tardiness': mts/60,
            'total_tardiness': tard,
            'loss': loss,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy': entropy,
            'terminal_reward': terminal_reward,
            'avg_step_reward': avg_step_reward,
            'trajectory_length': len(trajectory),
            'value_estimate_mean': np.mean(value_estimates) if value_estimates else 0,
            'advantage_mean': np.mean(advantages) if advantages else 0,
            'advantage_std': np.std(advantages) if advantages else 0,
            'num_late_orders': sum(1 for o in env.orders.values() 
                                  if o.tardiness(env.current_time) > 0),
        })
        
        if (ep+1) % 10 == 0:
            print(f"  Ep {ep+1:4d} | MS:{ms_min:5.0f} | MTO:{mto/60:5.0f} | "
                  f"MTS:{mts/60:5.0f} | Tard:{tard:5.0f} | "
                  f"Loss:{loss:.3f} | ALoss:{actor_loss:.3f} | "
                  f"CLoss:{critic_loss:.3f} | Ent:{entropy:.3f} | "
                  f"TRwd:{terminal_reward:6.1f} | Late:{history[-1]['num_late_orders']:2d}")
    
    print(f"\n  最优 makespan: {best_ms:.0f}min (EDD: 3910min, "
          f"改进: {(1-best_ms/3910)*100:.1f}%)")
    print(f"  最优 加权拖期: {best_tard:.0f}min (EDD: 956min, "
          f"改进: {(1-best_tard/956)*100:.1f}%)")
    
    return agent, history


def evaluate(data, agent, n_runs=20):
    """评估agent性能"""
    results = []
    last_env = None
    for i in range(n_runs):
        env = ShopFloorEnv(data)
        env.reset()
        _, ms, mto, mts, _ = run_episode(env, agent, data, greedy=(i % 3 == 0))
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
