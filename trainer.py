"""
PPO 训练主循环 - 高性能批量版本
核心优化：将 episode 内所有步的 policy forward 合并为一次批量计算
通过 padding + masking 处理不同大小的候选集
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from environment import WorkshopEnv, ORDERS, ALL_MACHINES, N_ORDERS, N_MACHINES, MIN_REMAINING
from models import (
    OrderPolicyNet, OrderValueNet,
    MachinePolicyNet, MachineValueNet,
    build_order_inputs, build_machine_inputs,
    ORDER_FEAT_DIM, MACHINE_FEAT_DIM, GLOBAL_FEAT_DIM,
    ORDER_CONTEXT_DIM, ORDER_HEURISTIC_DIM,
    MACHINE_CONTEXT_DIM, MACHINE_HEURISTIC_DIM,
)

DEVICE        = 'cpu'
LR            = 3e-4
GAMMA         = 0.99
LAM           = 0.95
CLIP_EPS      = 0.2
ENTROPY_COEF  = 0.05   # 提高熵系数，防止大动作空间下策略过早收敛到次优解
VALUE_COEF    = 0.25   # 降低 value loss 权重，policy loss 主导更新方向
MAX_GRAD_NORM = 0.5
N_EPOCHS      = 4
N_EPISODES    = 300
N_COLLECT     = 4      # 每次 PPO 更新前收集的 episode 数，降低梯度方差
FLAT_DIM = N_ORDERS*ORDER_FEAT_DIM + N_MACHINES*MACHINE_FEAT_DIM + GLOBAL_FEAT_DIM
REWARD_SCALE  = 100.0  # 终端奖励缩放（-7000 -> -70 量级）
SHAPING_SCALE = 1000.0 # predictive tardiness shaping 归一化基准（分钟）


def flat_state(obs):
    return np.concatenate([
        obs["order_features"].flatten(),
        obs["machine_features"].flatten(),
        obs["global_features"],
    ]).astype(np.float32)


class RolloutBuffer:
    """存储一个 episode 的全部轨迹"""
    def __init__(self):
        self.clear()

    def clear(self):
        # 订单智能体：存储原始 numpy，批量化时再处理
        self.o_ctx_np   = []   # list of np [n_cand, ORDER_CONTEXT_DIM]
        self.o_hs_np    = []   # list of np [n_cand, 2]
        self.o_actions  = []   # int
        self.o_logprobs = []   # float
        self.o_values   = []   # float
        self.o_flat_np  = []   # np [FLAT_DIM]
        # 设备智能体
        self.m_ctx_np   = []
        self.m_hs_np    = []
        self.m_actions  = []
        self.m_logprobs = []
        self.m_values   = []
        self.m_flat_np  = []
        # 奖励
        self.rewards    = []
        self.dones      = []

    def __len__(self):
        return len(self.rewards)


class RunningMeanStd:
    """
    Welford 在线算法维护 reward 序列的滑动均值和方差。
    用于归一化 rewards，保持 returns 与 v_pred 的尺度一致，
    避免 returns 归一化（破坏 value 学习目标）和不归一化（value loss 爆炸）的两难。
    """
    def __init__(self, epsilon=1e-4):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray):
        """用一批 rewards 更新统计量（Welford batch 更新）。"""
        batch_mean = x.mean()
        batch_var  = x.var()
        batch_count = len(x)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean  = self.mean + delta * batch_count / total
        self.var   = (self.var * self.count
                      + batch_var * batch_count
                      + delta**2 * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


def compute_gae(rewards, values, dones, last_value=0.0, gamma=GAMMA, lam=LAM):
    """GAE，支持截断 Bootstrap（last_value）。"""
    T   = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    g   = 0.0
    for t in reversed(range(T)):
        if dones[t]:
            nv = 0.0
            g  = 0.0
        elif t + 1 < T:
            nv = values[t + 1]
        else:
            nv = last_value   # episode 被截断时 Bootstrap
        delta  = rewards[t] + gamma * nv - values[t]
        g      = delta + gamma * lam * g
        adv[t] = g
    return adv, adv + np.array(values, dtype=np.float32)


def _pad_and_batch(ctx_list, hs_list, actions, heuristic_dim, context_dim):
    """
    将不同长度的候选集 pad 到统一大小
    返回: ctx_t [T,M,d], hs_t [T,M,h], mask [T,M], act_idx [T]
    """
    T       = len(ctx_list)
    sizes   = [c.shape[0] for c in ctx_list]
    max_sz  = max(sizes)

    ctx_pad = np.zeros((T, max_sz, context_dim),  dtype=np.float32)
    hs_pad  = np.zeros((T, max_sz, heuristic_dim), dtype=np.float32)
    mask    = np.zeros((T, max_sz), dtype=bool)

    for i, (c, h, sz) in enumerate(zip(ctx_list, hs_list, sizes)):
        ctx_pad[i, :sz] = c
        hs_pad[i,  :sz] = h
        mask[i,    :sz] = True

    act_clamped = [min(a, sz-1) for a, sz in zip(actions, sizes)]

    return (torch.FloatTensor(ctx_pad),
            torch.FloatTensor(hs_pad),
            torch.BoolTensor(mask),
            torch.LongTensor(act_clamped))


def ppo_update_batched(policy_net, value_net, optimizer,
                       ctx_list, hs_list, actions, old_logprobs_np,
                       flat_np, advantages, returns,
                       heuristic_dim, context_dim):
    """
    完全批量化的 PPO 更新：
    - policy 更新：pad 整个 episode → 一次 batch forward
    - value  更新：batch forward [T, flat_dim]
    """
    T = len(actions)
    if T == 0:
        return 0.0, 0.0, 0.0

    # 预先 pad 好
    ctx_t, hs_t, mask_t, act_idx = _pad_and_batch(
        ctx_list, hs_list, actions, heuristic_dim, context_dim)

    flat_t   = torch.FloatTensor(flat_np)
    ret_t    = torch.FloatTensor(returns)
    adv_t    = torch.FloatTensor(advantages)
    old_lp_t = torch.FloatTensor(old_logprobs_np)

    total_pl, total_vl, total_ent = 0.0, 0.0, 0.0

    for _ in range(N_EPOCHS):
        # ---- 批量 policy forward ----
        log_probs, entropy = policy_net(ctx_t, hs_t, mask=mask_t)  # [T, max_cand]
        new_lp = log_probs.gather(1, act_idx.unsqueeze(1)).squeeze(1)  # [T]

        ratio = (new_lp - old_lp_t).exp()
        s1    = ratio * adv_t
        s2    = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t
        p_loss= -torch.min(s1, s2).mean()

        # ---- 批量 value forward ----
        v_pred = value_net(flat_t)
        v_loss = F.mse_loss(v_pred, ret_t)

        loss = p_loss + VALUE_COEF * v_loss - ENTROPY_COEF * entropy
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(policy_net.parameters()) + list(value_net.parameters()),
            MAX_GRAD_NORM)
        optimizer.step()

        total_pl  += p_loss.item()
        total_vl  += v_loss.item()
        total_ent += entropy.item()

    return total_pl / N_EPOCHS, total_vl / N_EPOCHS, total_ent / N_EPOCHS


def _pred_tardiness_total(env):
    """
    预计完工拖期总和（分钟）：
    对每个未完成订单，取其当前最靠前 WAITING/IN_PROGRESS 工序，
    估算 current_time + MIN_REMAINING[该工序] 与 due_time 的差。
    从 step 1 起就有非零信号，真正实现 dense reward。
    """
    from environment import ST_DONE, ST_NOT_STARTED, OPS
    total = 0.0
    ct = env.current_time  # 秒
    for os in env.order_states:
        if os.op_status['G'] == ST_DONE:
            continue
        # 找最靠前的非完成工序
        for op in OPS:
            if os.op_status[op] != ST_DONE:
                mr = MIN_REMAINING[os.id].get(op, 0)  # 秒（最短剩余加工时间）
                pred_completion = ct + mr              # 预计完工时间（秒）
                tard = max(0.0, pred_completion - os.due_time) / 60.0  # 分钟
                total += tard
                break
    return total


def run_episode(env, order_policy, order_value,
                machine_policy, machine_value,
                buffer, training=True):
    """
    执行一个完整 episode。

    修复内容
    --------
    1. Shaping reward 改用 predictive tardiness，从第 1 步起就有信号
    2. episode 截断时记录 _truncated/_last_flat，供 GAE Bootstrap
    3. REWARD_SCALE=100，N_COLLECT 在 train() 层面控制
    """
    obs   = env.reset()
    steps = 0
    truncated = False

    prev_pred_tard = _pred_tardiness_total(env)

    while True:
        while not obs["schedulable"] and not env._all_done():
            if not env.advance_to_next_event():
                break
            obs = env._get_obs()

        schedulable = obs["schedulable"]
        if not schedulable:
            break

        fs   = flat_state(obs)
        fs_t = torch.FloatTensor(fs).unsqueeze(0)

        # ---- 订单选择 ----
        h_ord = env.heuristic_order_scores(schedulable)
        ctx_o = build_order_inputs(obs, schedulable)
        h_o_t = torch.FloatTensor(h_ord).unsqueeze(0)

        with torch.no_grad():
            lp_o, _ = order_policy(ctx_o, h_o_t)
            val_o   = order_value(fs_t).item()

        probs_o = lp_o[0].exp().cpu().numpy()
        probs_o = np.nan_to_num(probs_o, nan=1.0/len(schedulable))
        probs_o = np.clip(probs_o, 1e-10, None); probs_o /= probs_o.sum()
        cand_idx = (np.random.choice(len(schedulable), p=probs_o)
                    if training else int(np.argmax(probs_o)))

        sel_os, sel_op = schedulable[cand_idx]
        old_lp_o       = lp_o[0, cand_idx].item()

        # ---- 设备分配 ----
        free_m = env._machines_free_for(sel_op, sel_os.product_type)
        h_mach = env.heuristic_machine_scores(sel_os, sel_op, free_m)
        ctx_m  = build_machine_inputs(obs, sel_os, sel_op, free_m)
        h_m_t  = torch.FloatTensor(h_mach).unsqueeze(0)

        with torch.no_grad():
            lp_m, _ = machine_policy(ctx_m, h_m_t)
            val_m   = machine_value(fs_t).item()

        probs_m = lp_m[0].exp().cpu().numpy()
        probs_m = np.nan_to_num(probs_m, nan=1.0/len(free_m))
        probs_m = np.clip(probs_m, 1e-10, None); probs_m /= probs_m.sum()
        mach_idx = (np.random.choice(len(free_m), p=probs_m)
                    if training else int(np.argmax(probs_m)))

        chosen_m = free_m[mach_idx]
        old_lp_m = lp_m[0, mach_idx].item()

        # ---- 执行动作 ----
        obs, _, done = env.step(sel_os, sel_op, chosen_m)

        if training:
            # ---- Shaping reward：预计完工拖期增量 ----
            curr_pred_tard = _pred_tardiness_total(env)
            delta_tard     = curr_pred_tard - prev_pred_tard   # 正值 = 变差
            shaping_r      = -delta_tard / SHAPING_SCALE
            prev_pred_tard = curr_pred_tard

            buffer.o_ctx_np.append(ctx_o[0].numpy())
            buffer.o_hs_np.append(h_o_t[0].numpy())
            buffer.o_actions.append(cand_idx)
            buffer.o_logprobs.append(old_lp_o)
            buffer.o_values.append(val_o)
            buffer.o_flat_np.append(fs)

            buffer.m_ctx_np.append(ctx_m[0].numpy())
            buffer.m_hs_np.append(h_m_t[0].numpy())
            buffer.m_actions.append(mach_idx)
            buffer.m_logprobs.append(old_lp_m)
            buffer.m_values.append(val_m)
            buffer.m_flat_np.append(fs)

            buffer.rewards.append(shaping_r)
            buffer.dones.append(False)

        steps += 1
        if done:
            break
        if steps > 5000:
            truncated = True
            break

    while env.event_queue:
        env.advance_to_next_event()

    terminal_reward = env._terminal_reward()
    scaled_reward   = terminal_reward / REWARD_SCALE

    if training and buffer.rewards:
        buffer.rewards[-1] += scaled_reward   # 终端奖励叠加到最后一步
        buffer.dones[-1]    = True

    buffer._truncated = truncated
    buffer._last_flat = flat_state(env._get_obs()) if truncated else None

    return terminal_reward


def train(n_episodes=N_EPISODES,
          save_path="/kaggle/working/PPO/checkpoints"):
    import os, time
    os.makedirs(save_path, exist_ok=True)

    env            = WorkshopEnv()
    order_policy   = OrderPolicyNet()
    order_value    = OrderValueNet()
    machine_policy = MachinePolicyNet()
    machine_value  = MachineValueNet()

    opt_o = optim.Adam(
        list(order_policy.parameters()) + list(order_value.parameters()), lr=LR)
    opt_m = optim.Adam(
        list(machine_policy.parameters()) + list(machine_value.parameters()), lr=LR)

    # Running reward normalization：归一化 rewards 而非 returns
    # 保证 returns 和 v_pred 的尺度一致，避免 value loss 爆炸或塌陷
    rms = RunningMeanStd()

    history = {k: [] for k in
               ["reward", "makespan", "mto_tard", "mts_tard",
                "loss_o", "loss_m", "entropy_o", "entropy_m"]}

    buf = RolloutBuffer()

    hdr = (f"{'Ep':>5} | {'Reward':>10} | {'Makespan':>9} | "
           f"{'MTO(m)':>8} | {'MTS(m)':>8} | {'LossO':>8} | {'LossM':>8} | {'s/ep':>5}")
    print(hdr)
    print("-" * len(hdr))

    for ep in range(1, n_episodes + 1):
        buf.clear()
        t0 = time.time()

        # ---- 收集 N_COLLECT 个 episode，降低梯度方差 ----
        ep_rewards = []
        ep_metrics = []
        for _ in range(N_COLLECT):
            reward = run_episode(env, order_policy, order_value,
                                 machine_policy, machine_value, buf, training=True)
            ep_rewards.append(reward)
            ep_metrics.append(env.get_metrics())

        # 取最后一个 episode 的 metrics 用于日志
        reward  = ep_rewards[-1]
        metrics = ep_metrics[-1]
        T       = len(buf)

        if T > 0:
            raw_r  = np.array(buf.rewards, dtype=np.float32)
            full_d = buf.dones.copy()

            # 用 running statistics 归一化 rewards（不动 returns 本身）
            # 保持 advantages 和 returns 在同一尺度，value 网络有稳定的学习目标
            rms.update(raw_r)
            scaled_r = rms.normalize(raw_r)

            # Bootstrap：若最后一个 episode 被截断
            if getattr(buf, '_truncated', False) and buf._last_flat is not None:
                last_fs_t = torch.FloatTensor(buf._last_flat).unsqueeze(0)
                with torch.no_grad():
                    last_val_o = order_value(last_fs_t).item()
                    last_val_m = machine_value(last_fs_t).item()
            else:
                last_val_o = last_val_m = 0.0

            adv_o, ret_o = compute_gae(scaled_r, buf.o_values, full_d, last_value=last_val_o)
            adv_m, ret_m = compute_gae(scaled_r, buf.m_values, full_d, last_value=last_val_m)

            # 只归一化 advantages，不动 returns
            adv_o = (adv_o - adv_o.mean()) / (adv_o.std() + 1e-8)
            adv_m = (adv_m - adv_m.mean()) / (adv_m.std() + 1e-8)

            flat_o_np = np.array(buf.o_flat_np, dtype=np.float32)
            flat_m_np = np.array(buf.m_flat_np, dtype=np.float32)

            pl_o, vl_o, ent_o = ppo_update_batched(
                order_policy, order_value, opt_o,
                buf.o_ctx_np, buf.o_hs_np, buf.o_actions,
                np.array(buf.o_logprobs, dtype=np.float32),
                flat_o_np, adv_o, ret_o,
                ORDER_HEURISTIC_DIM, ORDER_CONTEXT_DIM)

            pl_m, vl_m, ent_m = ppo_update_batched(
                machine_policy, machine_value, opt_m,
                buf.m_ctx_np, buf.m_hs_np, buf.m_actions,
                np.array(buf.m_logprobs, dtype=np.float32),
                flat_m_np, adv_m, ret_m,
                MACHINE_HEURISTIC_DIM, MACHINE_CONTEXT_DIM)
        else:
            pl_o = vl_o = ent_o = pl_m = vl_m = ent_m = 0.0

        elapsed = time.time() - t0
        history["reward"].append(reward)
        history["makespan"].append(metrics["makespan"])
        history["mto_tard"].append(metrics["mto_tardiness"])
        history["mts_tard"].append(metrics["mts_tardiness"])
        history["loss_o"].append(pl_o + vl_o)
        history["loss_m"].append(pl_m + vl_m)
        history["entropy_o"].append(ent_o)
        history["entropy_m"].append(ent_m)

        if ep % 50 == 0 or ep == 1:
            print(f"{ep:>5} | {reward:>10.1f} | {metrics['makespan']:>9.1f} | "
                  f"{metrics['mto_tardiness']:>8.1f} | {metrics['mts_tardiness']:>8.1f} | "
                  f"{pl_o+vl_o:>8.4f} | {pl_m+vl_m:>8.4f} | {elapsed:>5.2f}")

    torch.save({
        "order_policy":   order_policy.state_dict(),
        "order_value":    order_value.state_dict(),
        "machine_policy": machine_policy.state_dict(),
        "machine_value":  machine_value.state_dict(),
    }, f"{save_path}/ppo_final.pt")
    print(f"\nModel saved → {save_path}/ppo_final.pt")

    return history, env, order_policy, order_value, machine_policy, machine_value
