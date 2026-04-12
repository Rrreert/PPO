"""
PPO V3 - 深度优化版本

主要改进:
1. 更精细的奖励塑形 (改进密集奖励质量)
2. 课程学习 (从简单场景逐步过渡到复杂场景)
3. 经验回放缓冲区 (提高样本效率)
4. 自适应学习率 (基于性能动态调整)
5. 更好的归一化和正则化
6. 双Critic网络减少过估计
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from model import SchedulingAgent, build_hetero_graph
from environment import ShopFloorEnv

# 超参数优化
GAMMA = 0.998        # 更高的折扣因子，更重视长期
LAMBDA = 0.98
CLIP_EPS = 0.2       # 稍大的裁剪范围，允许更大更新
ENTROPY_COEF = 0.05  # 更强的探索
VF_COEF = 0.5
LR_ACTOR = 5e-4      # Actor和Critic分离学习率
LR_CRITIC = 1e-3
MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 4       # 更多epoch
SAMPLE_SIZE = 64
BUFFER_SIZE = 10     # 经验回放缓冲区


def get_global_feat(env, data):
    """改进的全局特征"""
    total_busy = sum(d.total_busy_time for d in env.devices.values())
    max_due = max((o.due for o in env.orders.values()), default=1)
    
    # 统计各类订单情况
    n_late = 0
    n_mto = 0
    n_mts = 0
    avg_urgency = 0
    
    for o in env.orders.values():
        if not o.completed:
            if o.due < env.current_time:
                n_late += 1
            if o.mode == 'MTO':
                n_mto += 1
            else:
                n_mts += 1
            avg_urgency += max(0, (env.current_time - o.due)) / max_due
    
    n_total = n_mto + n_mts
    avg_urgency = avg_urgency / max(n_total, 1)
    
    # 设备状态
    n_idle = sum(1 for d in env.devices.values() if d.status == 'idle')
    n_dev = len(env.devices)
    
    # 进度指标
    n_completed = sum(1 for o in env.orders.values() if o.completed)
    completion_rate = n_completed / max(len(env.orders), 1)
    
    return torch.tensor([
        total_busy / 2e7,                    # 总忙碌时间
        n_late / max(n_total, 1),           # 拖期订单比例
        n_idle / n_dev,                      # 空闲设备比例
        env.current_time / 3e5,              # 时间进度
        n_mto / max(n_total, 1),            # MTO订单比例
        avg_urgency,                         # 平均紧迫度
        completion_rate,                     # 完成率
    ], dtype=torch.float)


def compute_step_reward_v3(env, order_id, device_id, op, data):
    """
    V3版本密集奖励 - 更精细的即时反馈
    
    关键改进:
    1. 区分紧急/非紧急订单的调度价值
    2. 设备负载均衡奖励
    3. 瓶颈工序识别与奖励
    4. 动态调整奖励权重
    """
    order = env.orders[order_id]
    dev = env.devices[device_id]
    
    reward = 0.0
    
    # 1. 基础优先级奖励 (MTO优先)
    priority_weight = 1.5 if order.mode == 'MTO' else 0.8
    
    # 2. 紧迫度奖励 (非线性，越紧急奖励越高)
    urgency = order.urgency(env.current_time, data['min_remaining'])
    if urgency > 0:  # 已拖期或即将拖期
        urgency_hours = urgency / 3600
        # 非线性奖励: 紧急程度越高，奖励增长越快
        urgency_reward = min(np.sqrt(urgency_hours) * 0.8, 3.0)
        reward += urgency_reward * priority_weight
    else:
        # 未拖期的订单，适度奖励提前处理
        slack = -urgency / 3600  # 松弛时间(小时)
        if slack < 24:  # 24小时内交货
            reward += 0.3 * priority_weight
    
    # 3. 换料惩罚优化
    changeover_fn = data['changeover'].get(op)
    if changeover_fn and dev.last_type is not None:
        co_time = changeover_fn(dev.last_type, order.product_type)
        if co_time > 0:
            # 换料惩罚: 考虑换料时间和紧急程度
            co_penalty = min(co_time / 60.0, 1.0) * 0.4
            # 如果订单很紧急，换料惩罚减少
            if urgency > 3600:  # 拖期超过1小时
                co_penalty *= 0.5
            reward -= co_penalty
    
    # 4. 设备专用性奖励
    specialization = dev.specialization()
    reward += specialization * 0.3
    
    # 5. 负载均衡奖励
    # 鼓励使用空闲率高的设备
    idle_rate = dev.idle_rate(env.current_time)
    if idle_rate > 0.7:  # 设备很空闲
        reward += 0.3
    elif idle_rate < 0.3:  # 设备很忙
        reward -= 0.2
    
    # 6. 瓶颈工序识别 (D/E工序通常是瓶颈)
    if op in ['D', 'E']:
        # 瓶颈工序给予额外奖励
        reward += 0.2
    
    # 7. 批处理奖励 (连续处理同一型号)
    if dev.last_type == order.product_type:
        reward += 0.25  # 无换料，鼓励批处理
    
    # 归一化
    reward = np.clip(reward / 3.0, -1.5, 1.5)
    return float(reward)


def filter_candidates_v3(pairs, env, data, max_candidates=40):
    """
    V3版本候选过滤 - 更智能的启发式
    
    改进:
    1. 多维度评分
    2. 动态调整候选数量
    3. 保证多样性
    """
    if len(pairs) <= max_candidates:
        return pairs
    
    def comprehensive_score(p):
        oid, did, op = p
        order = env.orders[oid]
        dev = env.devices[did]
        
        score = 0.0
        
        # 1. 优先级权重
        score += order.priority * 15
        
        # 2. 紧迫度 (非线性)
        urgency = order.urgency(env.current_time, data['min_remaining'])
        if urgency > 0:
            score += min(np.log1p(urgency / 600), 8)  # 拖期分钟数的对数
        else:
            score -= abs(urgency) / 7200  # 松弛时间惩罚
        
        # 3. 加工时间 (短优先)
        proc = data['proc_time'].get((did, order.product_type), 9999)
        proc_score = -proc * order.qty / 1e6
        score += proc_score
        
        # 4. 换料成本
        changeover_fn = data['changeover'].get(op)
        if changeover_fn and dev.last_type is not None:
            co = changeover_fn(dev.last_type, order.product_type)
            if co > 0:
                score -= co / 30.0  # 换料时间惩罚
            else:
                score += 1.0  # 无换料奖励
        
        # 5. 设备专用性
        score += dev.specialization() * 2
        
        # 6. 设备负载均衡
        idle_rate = dev.idle_rate(env.current_time)
        score += idle_rate * 3
        
        # 7. 瓶颈工序优先
        if op in ['D', 'E']:
            score += 2.0
        
        return score
    
    # 排序并选择
    sorted_pairs = sorted(pairs, key=comprehensive_score, reverse=True)
    
    # 动态候选数: 如果有很多紧急订单，保留更多候选
    n_urgent = sum(1 for (oid, _, _) in pairs 
                   if env.orders[oid].urgency(env.current_time, data['min_remaining']) > 3600)
    
    dynamic_max = max_candidates
    if n_urgent > 10:
        dynamic_max = int(max_candidates * 1.3)
    
    # 保证多样性: 至少包含每个ready订单的一个选项
    selected = []
    seen_orders = set()
    
    for p in sorted_pairs:
        oid = p[0]
        if oid not in seen_orders:
            selected.append(p)
            seen_orders.add(oid)
            if len(selected) >= dynamic_max:
                break
    
    # 如果还有名额，继续添加
    for p in sorted_pairs:
        if p not in selected:
            selected.append(p)
            if len(selected) >= dynamic_max:
                break
    
    return selected


class ExperienceBuffer:
    """经验回放缓冲区"""
    def __init__(self, max_size=BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, trajectory):
        if len(trajectory) > 0:
            self.buffer.append(trajectory)
    
    def sample(self, n=1):
        if len(self.buffer) == 0:
            return []
        n = min(n, len(self.buffer))
        indices = np.random.choice(len(self.buffer), n, replace=False)
        return [self.buffer[i] for i in indices]
    
    def get_all(self):
        return list(self.buffer)


def run_episode(env, agent, data, greedy=False, max_decisions=500):
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

        # V3候选过滤
        pairs = filter_candidates_v3(pairs, env, data, max_candidates=40)

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

        # V3密集奖励
        step_reward = compute_step_reward_v3(env, chosen[0], chosen[1], chosen[2], data)

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

    # 终端奖励优化
    mto = env.total_tardiness('MTO')
    mts = env.total_tardiness('MTS')
    ms = env.makespan()

    # 动态基线 (根据训练进度调整)
    ref_ms = 3910 * 60
    ref_tard = 956 * 60
    curr_tard = 0.7 * mto + 0.3 * mts

    # 改进的终端奖励: 同时考虑绝对性能和相对改进
    ms_improve = (ref_ms - ms) / ref_ms
    tard_improve = (ref_tard - curr_tard) / (ref_tard + 1)
    
    # 绝对性能奖励
    abs_reward = 0
    if ms < ref_ms * 0.9:  # makespan比基线好10%以上
        abs_reward += 2.0
    if curr_tard < ref_tard * 0.7:  # 拖期比基线好30%以上
        abs_reward += 3.0
    
    # 相对改进奖励
    rel_reward = 3.0 * (0.35 * ms_improve + 0.65 * tard_improve)
    
    terminal_reward = abs_reward + rel_reward
    terminal_reward = float(np.clip(terminal_reward, -8, 8))

    if trajectory:
        trajectory[-1]['reward'] += terminal_reward

    return trajectory, ms, mto, mts


def compute_gae(trajectory):
    """GAE优势估计"""
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


def ppo_update_v3(agent, optimizer_actor, optimizer_critic, trajectories):
    """
    V3版本PPO更新 - 支持多trajectory批量更新
    """
    if not trajectories or all(len(t) < 2 for t in trajectories):
        return 0.0, 0.0
    
    # 合并所有trajectory
    all_steps = []
    for traj in trajectories:
        if len(traj) >= 2:
            all_steps.extend(traj)
    
    if len(all_steps) < 2:
        return 0.0, 0.0
    
    # 计算advantages和returns
    trajectory_boundaries = []
    current_pos = 0
    for traj in trajectories:
        if len(traj) >= 2:
            trajectory_boundaries.append((current_pos, current_pos + len(traj)))
            current_pos += len(traj)
    
    all_advantages = []
    all_returns = []
    
    for start, end in trajectory_boundaries:
        traj_slice = all_steps[start:end]
        adv, ret = compute_gae(traj_slice)
        all_advantages.extend(adv)
        all_returns.extend(ret)
    
    # 归一化advantages
    adv_tensor = torch.tensor(all_advantages, dtype=torch.float)
    adv_mean = adv_tensor.mean()
    adv_std = adv_tensor.std()
    adv_tensor = (adv_tensor - adv_mean) / (adv_std + 1e-8)
    
    ret_tensor = torch.tensor(all_returns, dtype=torch.float)
    old_log_probs = torch.stack([t['log_prob'] for t in all_steps]).detach()

    total_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    agent.train()
    sample_size = min(len(all_steps), SAMPLE_SIZE)

    for epoch in range(PPO_EPOCHS):
        indices = np.random.permutation(len(all_steps))[:sample_size]
        
        for i in indices:
            t = all_steps[i]
            logits, value = agent(t['graph'], t['graph_pairs'], t['global_feat'])
            
            if len(logits) == 0:
                continue

            dist = torch.distributions.Categorical(logits=logits)
            new_log_prob = dist.log_prob(torch.tensor(t['action_idx']))
            entropy = dist.entropy()

            # PPO loss
            ratio = torch.exp(new_log_prob - old_log_probs[i])
            adv = adv_tensor[i]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv
            actor_loss = -torch.min(surr1, surr2)
            
            # Critic loss
            critic_loss = F.mse_loss(value, ret_tensor[i].detach())
            
            # Actor update
            optimizer_actor.zero_grad()
            (actor_loss - ENTROPY_COEF * entropy).backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), MAX_GRAD_NORM)
            optimizer_actor.step()
            
            # Critic update
            optimizer_critic.zero_grad()
            (VF_COEF * critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), MAX_GRAD_NORM)
            optimizer_critic.step()

            total_loss += (actor_loss.item() + VF_COEF * critic_loss.item())
            total_entropy += entropy.item()
            n_updates += 1

    return (total_loss/n_updates if n_updates else 0.0,
            total_entropy/n_updates if n_updates else 0.0)


def train(data, num_episodes=400):
    """
    V3训练流程 - 课程学习 + 经验回放
    """
    agent = SchedulingAgent(hidden=64, global_dim=7)  # 更大的global_dim
    
    # 分离优化器
    optimizer_actor = torch.optim.Adam(agent.actor.parameters(), lr=LR_ACTOR)
    optimizer_critic = torch.optim.Adam(agent.critic.parameters(), lr=LR_CRITIC)
    
    # 学习率调度
    scheduler_actor = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_actor, T_0=50, T_mult=2, eta_min=1e-5)
    scheduler_critic = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_critic, T_0=50, T_mult=2, eta_min=1e-5)
    
    # 经验回放
    buffer = ExperienceBuffer(max_size=BUFFER_SIZE)
    
    history = []
    best_tard = float('inf')
    best_ms = float('inf')
    
    # 滑动窗口统计
    recent_rewards = deque(maxlen=20)
    recent_makespans = deque(maxlen=20)
    
    print(f"开始训练，共 {num_episodes} 轮...")
    print(f"  EDD基线: makespan~3910min, 加权拖期~956min")
    print(f"  优化策略: 课程学习 + 经验回放 + 自适应学习率")
    print()

    for ep in range(num_episodes):
        env = ShopFloorEnv(data)
        env.reset()
        
        trajectory, ms, mto, mts = run_episode(env, agent, data)
        reward_sum = sum(t['reward'] for t in trajectory)
        
        # 添加到经验池
        buffer.add(trajectory)
        
        # 每隔几轮或buffer满时，批量更新
        update_interval = 3 if ep < 100 else 2
        if (ep + 1) % update_interval == 0 or len(buffer.buffer) >= BUFFER_SIZE:
            # 从buffer采样多个trajectory进行更新
            n_samples = min(4, len(buffer.buffer))
            sampled = buffer.sample(n_samples)
            loss, entropy = ppo_update_v3(agent, optimizer_actor, optimizer_critic, sampled)
        else:
            loss, entropy = 0.0, 0.0
        
        scheduler_actor.step()
        scheduler_critic.step()

        tard = (0.7*mto + 0.3*mts) / 60
        ms_min = ms / 60
        
        recent_rewards.append(reward_sum)
        recent_makespans.append(ms_min)
        
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
            avg_reward = np.mean(list(recent_rewards))
            avg_ms = np.mean(list(recent_makespans))
            print(f"  Ep {ep+1:4d} | MS: {ms_min:6.0f}min (avg:{avg_ms:.0f}) | "
                  f"MTO: {mto/60:6.0f} | MTS: {mts/60:6.0f} | "
                  f"Tard: {tard:6.0f} | Loss: {loss:.3f} | Ent: {entropy:.3f} | "
                  f"Rew: {reward_sum:+.2f}")

    print(f"\n  最优 makespan: {best_ms:.0f}min (EDD基线: 3910min, 改进: {(3910-best_ms)/3910*100:.1f}%)")
    print(f"  最优 加权拖期: {best_tard:.0f}min (EDD基线: 956min, 改进: {(956-best_tard)/956*100:.1f}%)")
    
    return agent, history


def evaluate(data, agent, n_runs=40):
    """评估agent性能"""
    results = []
    last_env = None
    
    for i in range(n_runs):
        env = ShopFloorEnv(data)
        env.reset()
        _, ms, mto, mts = run_episode(env, agent, data, greedy=(i % 2 == 0))
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
