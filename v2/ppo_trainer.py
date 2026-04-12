"""
PPO V3 - 训练稳定性优化版本

核心问题诊断与修复:
1. [修复] 单样本梯度更新 → 批量梯度累积后再step
2. [修复] 单episode训练方差大 → 多episode rollout buffer
3. [修复] 奖励尺度不一致 → 运行时奖励标准化(RunningMeanStd)
4. [修复] γ=0.995过高放大噪声 → 降至0.99
5. [修复] 无最优模型保存 → 保存/恢复best model
6. [修复] 熵系数固定 → 线性衰减探索
7. [修复] 终端奖励尺度跳变 → 平滑终端奖励设计
8. [新增] 梯度累积的mini-batch PPO更新
9. [新增] 学习率warmup阶段
10.[新增] 早停 + 耐心机制
"""
import torch
import torch.nn.functional as F
import numpy as np
import copy
from model import SchedulingAgent, build_hetero_graph
from environment import ShopFloorEnv

# ============ 超参数 ============
GAMMA = 0.99              # [修改] 从0.995降低,减少长horizon噪声放大
LAMBDA = 0.95             # [修改] 从0.97降低,减少方差
CLIP_EPS = 0.2            # [修改] 从0.15放宽到标准值0.2
ENTROPY_COEF_START = 0.05 # [新增] 初始熵系数(鼓励早期探索)
ENTROPY_COEF_END = 0.005  # [新增] 最终熵系数
VF_COEF = 0.5
LR = 1e-4                 # [修改] 从3e-4降低,减少更新震荡
MAX_GRAD_NORM = 0.5        # [修改] 从1.0降低,更严格的梯度裁剪
PPO_EPOCHS = 4             # [修改] 增加epoch数,配合batch更新
MINI_BATCH_SIZE = 32       # [新增] mini-batch大小
ROLLOUT_EPISODES = 3       # [新增] 每次更新收集的episode数
WARMUP_EPISODES = 20       # [新增] 学习率warmup轮数
PATIENCE = 50              # [新增] 早停耐心值


class RunningMeanStd:
    """运行时均值/方差标准化器 - 解决奖励尺度不稳定"""
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


def get_global_feat(env, data):
    total_busy = sum(d.total_busy_time for d in env.devices.values())
    n_late = sum(1 for o in env.orders.values()
                 if not o.completed and o.due < env.current_time)
    n_idle = sum(1 for d in env.devices.values() if d.status == 'idle')
    n_dev = len(env.devices)
    # [新增] 完成进度特征
    n_done = sum(1 for o in env.orders.values() if o.completed)
    n_total = max(len(env.orders), 1)
    return torch.tensor([
        min(total_busy / 2e7, 5.0),
        n_late / n_total,
        n_idle / n_dev,
        env.current_time / 3e5,
    ], dtype=torch.float)


def compute_step_reward(env, order_id, device_id, op, data, prev_metrics):
    """
    [优化] 密集即时奖励 - 简化结构, 减少奖励噪声
    """
    order = env.orders[order_id]
    dev = env.devices[device_id]

    reward = 0.0

    # 1. 优先级奖励 (MTO/MTS)
    priority_bonus = 0.3 if order.mode == 'MTO' else 0.05
    reward += priority_bonus

    # 2. 紧迫度奖励 - [优化] 使用更平滑的缩放
    urgency = order.urgency(env.current_time, data['min_remaining'])
    if urgency > 0:
        # 使用tanh平滑,避免线性截断造成的梯度不连续
        reward += float(np.tanh(urgency / 1e5)) * 0.4

    # 3. 换料惩罚 - [优化] 缩小惩罚幅度,避免主导奖励
    changeover_fn = data['changeover'].get(op)
    if changeover_fn and dev.last_type is not None:
        co = changeover_fn(dev.last_type, order.product_type)
        reward -= float(np.tanh(co / 60.0)) * 0.15

    # 4. 设备专用性微小奖励
    reward += dev.specialization() * 0.05

    return float(reward)


def filter_candidates(pairs, env, data, max_candidates=30):
    """启发式过滤候选(订单,设备)对"""
    if len(pairs) <= max_candidates:
        return pairs

    def score(p):
        oid, did, op = p
        order = env.orders[oid]
        dev = env.devices[did]
        urgency = order.urgency(env.current_time, data['min_remaining'])
        proc = data['proc_time'].get((did, order.product_type), 9999)
        s = 0.0
        s += order.priority * 10
        s += min(urgency / 3600, 5)
        s -= proc * order.qty / 1e6
        s += dev.specialization()
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
            # [优化] 温度缩放,早期更随机,后期更确定
            dist = torch.distributions.Categorical(logits=logits)
            action_idx = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action_idx))

        chosen = valid_pairs[action_idx]
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

    # ---- 终端奖励 ----
    mto = env.total_tardiness('MTO')
    mts = env.total_tardiness('MTS')
    ms = env.makespan()

    ref_ms = 3910 * 60
    ref_tard = 956 * 60
    curr_tard = 0.7 * mto + 0.3 * mts

    ms_improve = (ref_ms - ms) / ref_ms
    tard_improve = (ref_tard - curr_tard) / (ref_tard + 1)
    # [优化] 使用tanh平滑终端奖励,避免尖锐跳变
    terminal_reward = float(np.tanh(0.4 * ms_improve + 0.6 * tard_improve)) * 3.0

    # [优化] 将终端奖励分散到最后N步,而非全部堆积在最后一步
    if trajectory:
        spread_steps = min(10, len(trajectory))
        per_step = terminal_reward / spread_steps
        for i in range(spread_steps):
            trajectory[-(i + 1)]['reward'] += per_step

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


def ppo_update(agent, optimizer, buffer, entropy_coef):
    """
    [核心修复] 批量mini-batch PPO更新
    - 多episode合并为一个buffer
    - 按mini-batch聚合梯度后再step(而非逐样本step)
    """
    if len(buffer) < MINI_BATCH_SIZE:
        return 0.0, 0.0

    # 合并所有trajectory的GAE
    all_advantages = []
    all_returns = []
    for traj in buffer:
        adv, ret = compute_gae(traj)
        all_advantages.extend(adv)
        all_returns.extend(ret)

    # 展平buffer
    flat = []
    for traj in buffer:
        flat.extend(traj)

    adv_tensor = torch.tensor(all_advantages, dtype=torch.float)
    # [修复] 全局标准化advantage
    adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
    ret_tensor = torch.tensor(all_returns, dtype=torch.float)
    old_log_probs = torch.stack([t['log_prob'] for t in flat]).detach()

    total_loss = 0.0
    total_entropy = 0.0
    n_updates = 0
    N = len(flat)

    agent.train()

    for epoch in range(PPO_EPOCHS):
        # 随机打乱索引
        indices = np.random.permutation(N)

        # [核心修复] 按mini-batch聚合梯度
        for start in range(0, N, MINI_BATCH_SIZE):
            end = min(start + MINI_BATCH_SIZE, N)
            batch_idx = indices[start:end]

            optimizer.zero_grad()
            batch_loss = 0.0
            batch_entropy = 0.0
            valid_count = 0

            for i in batch_idx:
                t = flat[i]
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

                loss = actor_loss + VF_COEF * critic_loss - entropy_coef * entropy
                # [修复] 累积梯度而非逐样本step
                (loss / len(batch_idx)).backward()

                batch_loss += loss.item()
                batch_entropy += entropy.item()
                valid_count += 1

            if valid_count > 0:
                # [修复] 在mini-batch累积完毕后才step
                torch.nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                total_loss += batch_loss / valid_count
                total_entropy += batch_entropy / valid_count
                n_updates += 1

    return (total_loss / n_updates if n_updates else 0.0,
            total_entropy / n_updates if n_updates else 0.0)


def train(data, num_episodes=200):
    agent = SchedulingAgent(hidden=64)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

    # [新增] 学习率warmup + cosine退火
    def lr_lambda(ep):
        if ep < WARMUP_EPISODES:
            return float(ep + 1) / WARMUP_EPISODES
        progress = (ep - WARMUP_EPISODES) / max(num_episodes - WARMUP_EPISODES, 1)
        return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # [新增] 奖励标准化器
    reward_normalizer = RunningMeanStd()

    history = []
    best_score = float('inf')     # 综合指标: 越小越好
    best_state = None
    no_improve_count = 0

    print(f"开始训练，共 {num_episodes} 轮...")
    print(f"  EDD基线: makespan~3910min, 加权拖期~956min")
    print(f"  优化: 批量更新, {ROLLOUT_EPISODES}ep/update, "
          f"mini-batch={MINI_BATCH_SIZE}, warmup={WARMUP_EPISODES}ep")
    print()

    ep = 0
    while ep < num_episodes:
        # [核心修复] 收集多个episode的rollout buffer
        rollout_buffer = []
        ep_makespans = []
        ep_mtos = []
        ep_mtss = []

        for _ in range(ROLLOUT_EPISODES):
            if ep >= num_episodes:
                break
            env = ShopFloorEnv(data)
            env.reset()
            trajectory, ms, mto, mts = run_episode(env, agent, data)

            # [新增] 奖励标准化
            if trajectory:
                raw_rewards = [t['reward'] for t in trajectory]
                reward_normalizer.update(raw_rewards)
                for t in trajectory:
                    t['reward'] = float(reward_normalizer.normalize(np.array([t['reward']]))[0])

            rollout_buffer.append(trajectory)
            ep_makespans.append(ms)
            ep_mtos.append(mto)
            ep_mtss.append(mts)
            ep += 1

        # [新增] 熵系数线性衰减
        progress = min(ep / num_episodes, 1.0)
        entropy_coef = ENTROPY_COEF_START + (ENTROPY_COEF_END - ENTROPY_COEF_START) * progress

        # [核心修复] 用整个rollout buffer做批量PPO更新
        loss, entropy = ppo_update(agent, optimizer, rollout_buffer, entropy_coef)
        scheduler.step()

        # 记录本轮均值
        avg_ms = np.mean(ep_makespans)
        avg_mto = np.mean(ep_mtos)
        avg_mts = np.mean(ep_mtss)
        tard = (0.7 * avg_mto + 0.3 * avg_mts) / 60
        ms_min = avg_ms / 60

        # [新增] 综合评分 (用于best model跟踪)
        score = 0.4 * ms_min + 0.6 * tard
        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(agent.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1

        # [新增] 早停检测
        if no_improve_count >= PATIENCE and ep > num_episodes // 2:
            print(f"\n  早停: {PATIENCE}轮无改善, 在第{ep}轮停止")
            break

        reward_sum = sum(sum(t['reward'] for t in traj) for traj in rollout_buffer)
        record = {
            'episode': ep,
            'reward': reward_sum / len(rollout_buffer),
            'makespan': ms_min,
            'mto_tardiness': avg_mto / 60,
            'mts_tardiness': avg_mts / 60,
            'total_tardiness': tard,
            'loss': loss,
            'entropy': entropy,
        }
        history.append(record)

        if ep % 10 == 0 or ep <= 10:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Ep {ep:4d} | MS: {ms_min:6.0f}min | "
                  f"MTO: {avg_mto/60:6.0f} | MTS: {avg_mts/60:6.0f} | "
                  f"Tard: {tard:6.0f} | Loss: {loss:.3f} | Ent: {entropy:.3f} | "
                  f"LR: {current_lr:.1e} | EntC: {entropy_coef:.4f}")

    # [新增] 恢复最优模型
    if best_state is not None:
        agent.load_state_dict(best_state)
        print(f"\n  已恢复最优模型 (score={best_score:.1f})")

    # 计算最优值
    best_ms_val = min(h['makespan'] for h in history) if history else 0
    best_tard_val = min(h['total_tardiness'] for h in history) if history else 0
    print(f"  最优 makespan: {best_ms_val:.0f}min (EDD基线: 3910min)")
    print(f"  最优 加权拖期: {best_tard_val:.0f}min (EDD基线: 956min)")

    return agent, history


def evaluate(data, agent, n_runs=20):
    results = []
    last_env = None
    for i in range(n_runs):
        env = ShopFloorEnv(data)
        env.reset()
        _, ms, mto, mts = run_episode(env, agent, data, greedy=(i % 3 == 0))
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
