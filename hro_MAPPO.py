# -*- coding: utf-8 -*-
# @Author  : wang
# @Time    : 2025/9/8 16:03
# @File    : hro_MAPPO.py
# @Software: PyCharm
import copy
import numpy as np
import torch
import torch.nn as nn
from utils_mappo.util_mappo import get_gard_norm, huber_loss, mse_loss
from utils_mappo.valuenorm import ValueNorm
from utils_mappo.util_mappo import check


class hro_MAPPO():
    """
    MAPPO训练器类，用于更新策略。
    :param args: (argparse.Namespace) 包含相关模型、策略和环境信息的参数。
    :param policy: (R_MAPPO_Policy) 要更新的策略。
    :param device: (torch.device) 指定运行设备（cpu/gpu）。
    """

    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)  # 张量属性字典
        self.policy = policy  # 策略网络
        self.args = args
        self.global_coef = args.global_coef

        # PPO算法超参数
        self.clip_param = args.clip_param  # 剪辑参数
        self.ppo_epoch = args.ppo_epoch  # PPO迭代次数
        self.num_mini_batch = args.num_mini_batch  # 小批量数量
        self.data_chunk_length = args.data_chunk_length  # 数据块长度
        self.value_loss_coef = args.value_loss_coef  # 价值损失系数
        self.entropy_coef = args.entropy_coef  # 熵系数
        self.max_grad_norm = args.max_grad_norm  # 最大梯度范数
        self.huber_delta = args.huber_delta  # Huber损失delta参数
        self.v_global_coef = args.global_coef

        # 功能开关配置
        self._use_recurrent_policy = args.use_recurrent_policy  # 是否使用循环策略
        self._use_naive_recurrent = args.use_naive_recurrent_policy  # 是否使用朴素循环策略
        self._use_max_grad_norm = args.use_max_grad_norm  # 是否使用梯度裁剪
        self._use_clipped_value_loss = args.use_clipped_value_loss  # 是否使用价值剪辑损失
        self._use_huber_loss = args.use_huber_loss  # 是否使用Huber损失
        self._use_popart = args.use_popart  # 是否使用PopArt归一化
        self._use_valuenorm = args.use_valuenorm  # 是否使用ValueNorm归一化
        self._use_value_active_masks = args.use_value_active_masks  # 价值网络是否使用激活掩码
        self._use_policy_active_masks = args.use_policy_active_masks  # 策略网络是否使用激活掩码

        # 互斥检查
        assert (self._use_popart and self._use_valuenorm) == False, (
            "self._use_popart and self._use_valuenorm 不能同时设置为True")

        # 价值归一化器设置
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out  # 使用PopArt输出层
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)  # 创建ValueNorm实例
        else:
            self.value_normalizer = None  # 不使用价值归一化
        # ---------- 冻结旧策略（关键） ----------
        # 旧策略只需拷贝 wrapper，然后对 actor/critic 设 eval+冻结参数
        self.policy_old = copy.deepcopy(self.policy)
        if hasattr(self.policy_old, "actor") and isinstance(self.policy_old.actor, nn.Module):
            self.policy_old.actor.eval()
            for p in self.policy_old.actor.parameters():
                p.requires_grad_(False)
        if hasattr(self.policy_old, "critic") and isinstance(self.policy_old.critic, nn.Module):
            self.policy_old.critic.eval()
            for p in self.policy_old.critic.parameters():
                p.requires_grad_(False)

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        计算价值函数损失。
        :param values: (torch.Tensor) 价值函数预测值。
        :param value_preds_batch: (torch.Tensor) 数据批次中的"旧"价值预测（用于价值剪辑损失）
        :param return_batch: (torch.Tensor) 累积回报。
        :param active_masks_batch: (torch.Tensor) 表示智能体在给定时间步是否活跃。

        :return value_loss: (torch.Tensor) 价值函数损失。
        """
        # 价值预测剪辑
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)
        # 价值归一化处理
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)  # 更新归一化器
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        # 损失函数计算
        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)  # Huber损失
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)  # 均方误差损失
            value_loss_original = mse_loss(error_original)

        # 价值损失选择
        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)  # 取最大值
        else:
            value_loss = value_loss_original

        # 激活掩码处理
        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()  # 只计算活跃智能体
        else:
            value_loss = value_loss.mean()  # 平均损失

        return value_loss

    def _get_batch_size(self,graph_like):
        if graph_like is None:
            return None
        b = None
        if isinstance(graph_like, dict):
            b = graph_like.get("batch", None)
        else:
            b = getattr(graph_like, "batch", None)
        if b is not None:
            if isinstance(b, torch.Tensor):
                return int(b.max().item()) + 1
            elif isinstance(b, np.ndarray):
                return int(b.max()) + 1
        return None

    def ppo_update(self, sample, buffer,update_actor=True):
        """
        纯图版 PPO 更新。
        """
        # -------- 1) 解包（与 generator 的 yield 顺序严格一致）--------
        (critic_graph_batch,actor_graph_batch,
         rnn_states_batch, rnn_states_critic_batch,
         actions_batch,
         value_preds_batch_global, value_preds_batch_local,
         return_batch_global, return_batch_local,
         masks_batch, active_masks_batch,
         old_action_log_probs_batch, adv_targ,
         available_actions_batch) = sample

        # -------- 2) 设备/类型整理 --------
        todev = self.tpdv if hasattr(self, "tpdv") else {"device": self.device, "dtype": torch.float32}
        actions_batch = check(actions_batch).to(**todev)  # [T, m, A]  ✅
        masks_batch = check(masks_batch).to(**todev)  # [T, m, 1]  ✅
        active_masks_batch = check(active_masks_batch).to(**todev)  # [T, m, 1]  ✅
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**todev)
        adv_targ = check(adv_targ).to(**todev)
        value_preds_batch_global = check(value_preds_batch_global).to(**todev)
        value_preds_batch_local = check(value_preds_batch_local).to(**todev)
        return_batch_global = check(return_batch_global).to(**todev)
        return_batch_local = check(return_batch_local).to(**todev)
        masks_batch = check(masks_batch).to(**todev)
        active_masks_batch = check(active_masks_batch).to(**todev)
        # available_actions：允许为 None；否则必须是 [B, A]
        if available_actions_batch is not None:
            available_actions_batch = check(available_actions_batch).to(**todev)

        # -------- 3) 批维一致性防呆（关键！）--------
        # 不要用图的“节点批”去代表样本数；样本数= T * m（时间 × agent）
        Tm, m = actions_batch.shape[0], actions_batch.shape[1]  # T, m
        B_samples = Tm * m  # [T*m] ✅ 真正的样本数（展平后）

        #  用展平后的维度做一致性检查（而不是拿 available_actions 的第0维和“某个B”比）
        if available_actions_batch is not None:
            actions_flat = actions_batch.reshape(B_samples, -1)  # [T*m, A]
            avail_flat = available_actions_batch.reshape(B_samples, -1)  # [T*m, nA]
            assert avail_flat.shape[0] == actions_flat.shape[0], \
                f"[ppo_update] avail_flat rows({avail_flat.shape[0]}) != actions_flat rows({actions_flat.shape[0]}). " \
                f"请确保二者都按 [T,m,...] 展平为 [T*m,...] 再进行对齐检查。"

        # 形状防呆（避免后续广播或求和错误）
        if adv_targ.dim() == 1:
            adv_targ = adv_targ.unsqueeze(-1)  # [B,1]

        # old_action_log_probs 必须是 rollout 时冻结的旧值：不应带梯度
        if getattr(old_action_log_probs_batch, "requires_grad", False):
            old_action_log_probs_batch = old_action_log_probs_batch.detach()
        if old_action_log_probs_batch.dim() == 1:
            old_action_log_probs_batch = old_action_log_probs_batch.unsqueeze(-1)


        # -------- 4) 策略前向（图版 + 批掩码）--------
        action_log_probs, dist_entropy  = self.policy.evaluate_actions_graph(
            actor_graph_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions=available_actions_batch,
            active_masks=active_masks_batch,
        )
        # -------- 4.1) 冻结旧策略计算 old log-prob（关键！）--------
        with torch.no_grad():
            old_action_log_probs_frozen, _ = self.policy_old.evaluate_actions_graph(
                actor_graph_batch,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                available_actions=available_actions_batch,
                active_masks=active_masks_batch,
            )
        old_action_log_probs = old_action_log_probs_frozen  # 以快照为准

        # 诊断：熵必须可反传（否则 actor 只剩 0 梯度）
        # 若 evaluate_actions_graph 内把 entropy detach/no_grad 了，这里将观测到 False
        if hasattr(dist_entropy, "requires_grad") and (not dist_entropy.requires_grad):
            # 强制提醒：如果你的 evaluate_actions_graph 返回的是已 detach 的熵，
            # 请到策略里改成从 distribution.entropy() 直接求平均，不要 .detach() / no_grad。
            pass

        # -------- 5) 价值网络（全局 / 局部；均为图版）--------
        v_global_new,_ = self.policy.get_global_values_graph(
            critic_graph_batch, rnn_states_critic_batch, masks_batch
        )
        v_local_new  = self.policy.get_local_values_graph(
            actor_graph_batch, rnn_states_batch, masks_batch
        )


        # -------- 6) 优势标准化（提升稳定性）--------
        # 采用小批优势归一化；可通过 self._use_adv_norm 开关（默认 True）
        use_adv_norm = getattr(self, "_use_adv_norm", True)  # 若未在外部显式设置，则默认启用
        # use_adv_norm = False
        if use_adv_norm:
            adv_mean = adv_targ.mean()
            adv_std = adv_targ.std(unbiased=False).clamp_min(1e-8)
            adv_targ = (adv_targ - adv_mean) / adv_std

        # -------- 7) 策略目标（含剪辑）--------
        # 执行器更新
        # ratio = torch.exp(action_log_probs - old_action_log_probs_batch)  # 重要性采样权重
        ratio = torch.exp(action_log_probs - old_action_log_probs)  # 重要性采样权重
        surr1 = ratio * adv_targ  # 未剪辑的目标
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ  # 剪辑的目标

        # 策略损失计算
        if self._use_policy_active_masks:
            # 分母加 clamp，避免 0 除导致 NaN
            denom = active_masks_batch.sum().clamp(min=1.0)
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / denom
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # policy_loss = policy_action_loss
        # 熵正则（确保是均值，数值尺度稳定）
        entropy_loss = -dist_entropy.mean() * self.entropy_coef
        policy_loss = policy_action_loss + entropy_loss  # 注意：这里把“负熵”直接并入 policy_loss

        # ---------- 8) 反向传播与优化 ----------
        #  === 策略(Actor)优化 ===
        actor_grad_norm = None  #  缓存返回值：未更新时为 None
        if update_actor:
            self.policy.actor_optimizer.zero_grad()  # 清零梯度
            total_loss = policy_loss - dist_entropy * self.entropy_coef
            total_loss.backward()    # 反向传播（包含熵正则化），# 计算梯度 "告诉神经网络应该往哪个方向调整参数"
            if self._use_max_grad_norm:
                actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            else:
                actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

            self.policy.actor_optimizer.step()
        else:
            print("[ppo_update] 提示：本轮未更新 actor（update_actor=False）。")

        # === 价值损失（两路）===
        # 这里沿用你现有的 cal_value_loss（里头支持 value-clip / huber / active_masks）
        # 重要：把 old value 和 return 分别传入对应通道
        value_loss_l = self.cal_value_loss(v_local_new, value_preds_batch_local, return_batch_local,
                                           active_masks_batch)  # 计算价值损失
        value_loss_g = self.cal_value_loss(v_global_new, value_preds_batch_global, return_batch_global,
                                           active_masks_batch)  # 计算价值损失

        global_coef = self.global_coef
        value_loss = global_coef * value_loss_g + (1 - global_coef) * value_loss_l

        self.policy.critic_optimizer.zero_grad()  # 清零梯度
        (value_loss * self.value_loss_coef).backward()  # 反向传播

        #clip 所有 param_groups 的参数，避免仅裁剪 group[0]
        if self._use_max_grad_norm:
            # 收集所有 group 的参数，并且过滤掉没有梯度的
            all_params = [p for g in self.policy.critic_optimizer.param_groups
                          for p in g["params"] if p.grad is not None]
            critic_grad_norm = nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
        else:
            params = self.policy.critic_optimizer.param_groups[0]["params"]
            critic_grad_norm = get_gard_norm(params)
        self.policy.critic_optimizer.step()

        # ---- 诊断指标（供 TensorBoard 可视化）----
        # approx KL / clip fraction / ratio 统计，帮助定位“ratio≈1”的问题
        with torch.no_grad():
            approx_kl = (old_action_log_probs_batch - action_log_probs).mean() #KL散度，可用于监控
            clip_frac = ((ratio - 1.0).abs() > self.clip_param).float().mean()
            ratio_mean = ratio.mean()
            # 1) 新旧 logp 的绝对差
            # delta_lp = (action_log_probs - old_action_log_probs_batch).abs().mean()
            delta_lp = (action_log_probs - old_action_log_probs).abs().mean()
            # 2) ratio 的均值/方差
            r_mean, r_std = ratio.mean(), ratio.std()
            # 3) 参数是否真的更新了
            # 任选一个 actor 参数，更新前后做一次范数差（建议你在 train() 外层也做）
            any_param = next(self.policy.actor.parameters())
            print(f"[DBG] delta_lp={float(delta_lp):.6g}, ratio_mean={float(r_mean):.6g}, "
                  f"ratio_std={float(r_std):.6g}, actor_lr={self.policy.actor_optimizer.param_groups[0]['lr']}")
        # 你原本把 imp_weights 返回；这里沿用返回 ratio 以便画直方图
        imp_weights = ratio.detach()

        return (value_loss,value_loss_g,value_loss_l, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm,
                imp_weights,approx_kl,clip_frac)

    def train(self, buffer, update_actor=True):
        """
        使用小批量梯度下降执行训练更新。
        :param buffer: (SharedReplayBuffer) 包含训练数据的缓冲区。
        :param update_actor: (bool) 是否更新执行器网络。
        :return train_info: (dict) 包含训练更新信息的字典（如损失、梯度范数等）。
        """
        # —— 每轮 PPO 开始前，同步旧策略快照（只同步 actor/critic）——
        self.policy_old.actor.load_state_dict(self.policy.actor.state_dict())
        self.policy_old.critic.load_state_dict(self.policy.critic.state_dict())
        self.policy_old.actor.eval()
        self.policy_old.critic.eval()
        for p in self.policy_old.actor.parameters():
            p.requires_grad_(False)
        for p in self.policy_old.critic.parameters():
            p.requires_grad_(False)

        # 优势计算
        if self._use_popart or self._use_valuenorm: # 进行归一化
            adv_g = buffer.returns_global[:-1] - self.value_normalizer.denormalize(buffer.value_preds_global[:-1] )
            adv_l = buffer.returns_local[:-1] - self.value_normalizer.denormalize(buffer.value_preds_local[:-1] )
            alpha =self.global_coef
            advantages = alpha * adv_g + (1.0 - alpha) * adv_l
        else: #不进行归一化
            adv_g = buffer.returns_global[:-1] - buffer.value_preds_global[:-1]
            adv_l = buffer.returns_local[:-1] - buffer.value_preds_local[:-1]
            alpha = self.global_coef
            advantages = alpha * adv_g + (1.0 - alpha) * adv_l

        # 优势归一化
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan  # 掩码非活跃智能体
        mean_advantages = np.nanmean(advantages_copy)  # 计算均值（忽略NaN）
        std_advantages = np.nanstd(advantages_copy)  # 计算标准差（忽略NaN）
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)  # 标准化

        # 初始化训练信息字典
        train_info = {}
        train_info['value_loss'] = 0
        train_info['value_loss_local'] = 0
        train_info['value_loss_global'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['approx_kl'] = 0
        train_info['clip_frac'] = 0

        actual_update_steps = 0  # 计数器

        # 多轮PPO更新
        for epoch in range(self.ppo_epoch):
            # 选择数据生成器
            data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            epoch_approx_kl = []  # 记录当前 epoch 所有 mini-batch 的 KL 散度

            # 小批量更新
            for sample in data_generator:
                value_loss,value_loss_global,value_loss_local, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm,\
                    imp_weights ,approx_kl,clip_frac= self.ppo_update(sample, buffer, update_actor)

                epoch_approx_kl.append(approx_kl.item())

                actual_update_steps += 1  # 每次 mini-batch 更新后加 1

                # 累加训练信息
                train_info['value_loss'] += value_loss.item()
                train_info['value_loss_global'] += value_loss_global.item()
                train_info['value_loss_local'] += value_loss_local.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean().item() # torch.exp(新策略-旧策略)
                train_info['approx_kl'] += approx_kl.item()
                train_info['clip_frac'] += clip_frac.item()

            # 新增：KL 早停检查
            # 如果当前 epoch 的平均 KL 散度超过阈值的 1.5 倍，立即停止本轮数据的后续 epoch 更新
            current_epoch_kl = np.mean(epoch_approx_kl)
            if current_epoch_kl > self.args.target_kl_dis * 1.5:
                print(f"[MAPPO] 触发 KL 早停: Epoch {epoch+1}/{self.ppo_epoch}, 平均 KL={current_epoch_kl:.4f} > 容忍阈值({self.args.target_kl_dis * 1.5:.4f})")
                break

        # 计算平均训练信息
        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= actual_update_steps
        adv = advantages.mean()
        return train_info,adv

    def prep_training(self):
        """设置为训练模式"""
        self.policy.actor.train()  # 执行器训练模式
        self.policy.critic.train()  # 评论家训练模式
        self.policy.local_critic.train()  # 评论家训练模式

    def prep_rollout(self):
        """设置为评估模式"""
        self.policy.actor.eval()  # 执行器评估模式
        self.policy.critic.eval()  # 评论家评估模式
        self.policy.local_critic.eval()  # 评论家评估模式