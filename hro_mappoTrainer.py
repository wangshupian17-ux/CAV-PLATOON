# -*- coding: utf-8 -*-
# @Author  : wang
# @Time    : 2025/7/10 16:39
# @File    : mappoTrainer.py
# @Software: PyCharm

import time
from datetime import datetime

import traci
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from hro_MAPPO import hro_MAPPO
from hro_MAPPOPolicy import hro_MAPPOPolicy
from optimize_platoon_fast import optimize_platoon_fast
from shared_buffer import SharedReplayBuffer
from sumoEnv import SumoEnv
from utils import *
import os, time, atexit

class Trainer(object):
    """
        Trainer is the main controller for MAPPO-based platoon training.
    """
    def __init__(self, args, sumo_binary, sumo_config):
        self.args = args

        self.cell_length = args.cell_length
        self.max_epi_time = args.max_steps * args.control_interval
        self.max_steps = args.max_steps
        self.control_interval = args.control_interval

        self.device = args.device
        self.max_episodes = args.max_episodes
        self.episode_length = args.episode_length
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.agent_save_freq = args.agent_save_freq
        self.agent_update_freq = args.agent_update_freq

        self.action_dim_dis = args.action_dim_dis
        self.lr_actor = args.lr_actor
        self.lr_critic = args.lr_critic
        self.lr_std = args.lr_std
        self.target_kl_dis = args.target_kl_dis
        self.target_kl_con = args.target_kl_con
        self.gamma = args.gamma
        self.lam = args.gae_lambda
        self.clip_param = args.clip_param
        self.max_grad_norm = args.max_grad_norm
        self.entropy_coef = args.entropy_coef
        self.random_seed = args.random_seed

        self.use_centralized_V = self.args.use_centralized_V
        self.use_linear_lr_decay = self.args.use_linear_lr_decay

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.save_dir = f"hromappo_model/{timestamp}"

        self.file_to_save = 'data/'
        if not os.path.exists(self.file_to_save):
            os.makedirs(self.file_to_save)
        self.record_mark = args.record_mark

        self.policy_save = os.path.join(
            self.file_to_save,
            f'policy/{self.record_mark}_{timestamp}'
        )
        self.results_save = os.path.join(
            self.file_to_save,
            f'results/{self.record_mark}_{timestamp}'
        )
        self.rolling_scores_save = os.path.join(
            self.file_to_save,
            f'rolling_scores/{self.record_mark}_{timestamp}'
        )

        self.cav_num = args.cav_num

        os.makedirs(self.policy_save, exist_ok=True)
        os.makedirs(self.results_save, exist_ok=True)
        os.makedirs(self.rolling_scores_save, exist_ok=True)

        self.env = SumoEnv(
            sumo_binary,
            sumo_config,
            self.max_steps,
            self.control_interval,
            self.device,
            self.args,
            rw_override={"w_on_time": 400.0, "w_tgt_gap_pen": 6.0}
        )

        self.buffer = SharedReplayBuffer(
            self.args,
            args.cav_num,
            self.env.observation_space[0],
            self.env.share_observation_space[0],
            self.env.action_space[0]
        )

        self.policy = hro_MAPPOPolicy(
            self.args,
            self.env.observation_space[0],
            self.env.share_observation_space[0],
            self.env.action_space[0],
            device=self.device
        )

        self.trainer = hro_MAPPO(self.args, self.policy, device=self.device)

        base_dir = 'runs/04'
        run_name = time.strftime('%Y%m%d-%H%M')
        log_dir = os.path.join(base_dir, run_name)
        os.makedirs(log_dir, exist_ok=True)


        self.tb_train_step = 0
        self._action_hist = np.zeros(self.action_dim_dis, dtype=np.int64)


    def run(self):
        self.warmup()
        episodes = self.args.max_episodes
        global_step = 0
        platoon_suss_num = 0

        for episode in range(self.args.max_episodes):
            self.env.reset(episode)
            success = 0
            last_control_time = -self.control_interval
            plan = None
            target_time = -1
            rewards = []
            team_rewards = []

            decisions_this_ep = 0
            total_reward_local = 0
            total_reward_team = 0
            total_reward_local_mean = 0
            rew_comp_accum = {
                "r_lane_raw": [],
                "r_act_raw": [],
                "r_collision_raw": [],
                "r_acc": [],
                "r_uncert_raw": [],
                "r_speed_raw": [],
                "r_local_raw": [],
                "r_prog": [],
                "r_flow_raw": [],
                "r_team_done_bonus": [],
            }

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.max_steps):
                traci.simulationStep()
                current_time = traci.simulation.getTime()
                if current_time - last_control_time >= self.control_interval:
                    last_control_time = current_time
                    cav_states = self.env.get_cav_state()
                    cell_info = self.env.get_cell_info(self.cell_length)

                    if self.env.success_platoon == True:
                        break
                    else:
                        if len(cav_states) == self.cav_num:
                            arrive_erro = 20
                            first_x = cav_states[0]['x_init'][0]
                            if plan != None:
                                target_x = plan['CAV_targets'][0]['x_final']
                            else:
                                target_x = 0
                            if plan is None or first_x - target_x > arrive_erro:
                                start_time = time.time()
                                plan = optimize_platoon_fast(cav_states, cell_info, current_time, self.args)
                                end_time = time.time()
                                print(f"episode：{episode}---current_time:{current_time}，执行模块1的计算时间: {end_time - start_time} 秒")

                                if plan['target_time_step'] != None:
                                    target_cell_x = (plan['target_cell'][0] + 1) * self.cell_length
                                    self.env.read_plan(plan)
                                    target_time = plan['target_time_step']
                                    print(f"        target_time_step': {plan['target_time_step']},'need_time:'{plan['need_time']}, 'target_cell': {plan['target_cell']}")
                                else:
                                    plan = None
                                    print("⚠️ MIQP未能找到最优解或可行解。")
                                    self.env.plan = None
                                    continue
                        else:
                            print("⚠️ CAV数量不足，不能形成编队")
                            t_last = int(self.buffer.step) - 1
                            if t_last >= 0:
                                self.buffer.terminated[t_last, :, 0] = 0.0
                                self.buffer.truncated[t_last, :, 0] = 1.0
                                self.buffer.active_masks[t_last + 1, :, 0] = 0.0
                                self.buffer.masks[t_last + 1, :, 0] = 0.0
                            break

                    values_globals, values_locals, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, \
                        available_actions = self.collect()

                    stages = np.argmax(actions_env, axis=-1).astype(np.int64)
                    cav_targets = plan['CAV_targets'] if (plan is not None and 'CAV_targets' in plan) else []

                    obs_raw_next, rewards, team_rewards, info = self.env.step_discrete(stages, cav_states, cav_targets)
                    dones = np.array(info.get("dones", [False] * self.cav_num), dtype=bool).reshape(self.cav_num, 1)
                    terminated = np.array(info.get("terminated", [False] * self.cav_num), dtype=bool).reshape(self.cav_num, 1)
                    truncated = np.array(info.get("truncated", [False] * self.cav_num), dtype=bool).reshape(self.cav_num, 1)
                    self.last_graph = obs_raw_next

                    actor_graph, critic_graph = self._graphs_from_env_obs(self.last_graph)

                    masks = np.ones((self.cav_num, 1), dtype=np.float32)
                    masks[dones] = 0.0
                    active_masks_next = (1.0 - dones.astype(np.float32))

                    self.buffer.insert(
                        share_obs=critic_graph,
                        obs=actor_graph,
                        actions=actions,
                        action_log_probs=action_log_probs,
                        rewards=np.asarray(rewards, dtype=np.float32).reshape(self.cav_num, 1),
                        team_rewards=team_rewards,
                        v_global_t=values_globals,
                        v_local_t=values_locals,
                        available_actions=available_actions,
                        rnn_states_actor=rnn_states,
                        rnn_states_critic=rnn_states_critic,
                        masks=masks,
                        active_masks_next=active_masks_next,
                        terminated=terminated,
                        truncated=truncated,
                    )

                    global_step += 1

                    decisions_this_ep += 1
                    done_array = np.asarray(dones, dtype=bool)
                    if np.any(done_array, axis=(0, 1)).any():
                        break
                    if decisions_this_ep >= self.episode_length:
                        break

            if decisions_this_ep > 0:
                self.compute()


            if self.buffer.step > self.args.agent_update_minsize:
                train_infos, advantages = self.train()

                def _to_float(x):
                    try:
                        return float(x)
                    except Exception:
                        try:
                            return float(x.item())
                        except Exception:
                            return float(np.array(x).astype(np.float32))


                self.tb_train_step += 1
                self.buffer.after_update()

                self.save(episode)


    def warmup(self):
        g = self.env.reset(-1)
        self.last_graph = g
        actor_graph, critic_graph = self._graphs_from_env_obs(self.last_graph)

        self.buffer.share_obs.append(critic_graph)
        self.buffer.obs.append(actor_graph)
        self.buffer.rnn_states[0, 0] = 0.0
        self.buffer.rnn_states_critic[0, 0] = 0.0
        self.buffer.masks[0, 0] = 1.0

    @torch.no_grad()
    def collect(self):
        self.trainer.prep_rollout()

        t = int(self.buffer.step)
        share_obs_t = self.buffer.share_obs[t]
        obs_t = self.buffer.obs[t]
        rnn_t = self.buffer.rnn_states[t]
        rnnc_t = self.buffer.rnn_states_critic[t]
        masks_t = self.buffer.masks[t]

        available_actions_t = None
        A = self.env.action_space[0].n
        available_actions_np = np.ones((self.cav_num, A), dtype=np.float32)
        for a, vid in enumerate(self.env.cav_ids):
            lane_id = int(traci.vehicle.getLaneIndex(vid))
            available_actions_np[a, 1] = 1.0 if lane_id > 0 else 0.0
            available_actions_np[a, 2] = 1.0 if lane_id < (self.env.lane_num - 1) else 0.0
        available_actions_t = torch.as_tensor(available_actions_np, dtype=torch.float32, device=self.device)

        values_global, values_local, action, logp, rnn_new, rnn_state_critic = self.trainer.policy.get_actions_graph(
            share_obs_t, obs_t, rnn_t, rnnc_t, masks_t,
            available_actions=available_actions_t,
            deterministic=False
        )

        values_globals = _t2n(values_global)
        values_locals = _t2n(values_local)
        actions = _t2n(action)
        logps = _t2n(logp)
        rnn_new = _t2n(rnn_new)
        rnn_state_critic = _t2n(rnn_state_critic)

        actions_env = np.squeeze(np.eye(self.env.action_space[0].n)[actions], 1)

        return values_globals, values_locals, actions, logps, rnn_new, rnn_state_critic, actions_env, available_actions_t

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()

        rnn_states_actor_last = self.buffer.rnn_states[-1]
        rnn_states_critic_last = self.buffer.rnn_states_critic[-1]
        masks_last = self.buffer.masks[-1]

        actor_graph_last, critic_graph_last = self._graphs_from_env_obs(self.last_graph)
        next_global_values, _ = self.trainer.policy.get_global_values_graph(
            critic_graph_last, rnn_states_critic_last, masks_last
        )

        next_global_values = _t2n(next_global_values)

        next_local_values = self.trainer.policy.get_local_values_graph(
            actor_graph_last, rnn_states_actor_last, masks_last
        )
        next_local_values = _t2n(next_local_values)

        self.buffer.compute_returns(next_global_values, next_local_values, self.trainer.value_normalizer)

    def train(self):
        self.trainer.prep_training()
        train_infos, advantages = self.trainer.train(self.buffer)
        return train_infos, advantages


    def _graphs_from_env_obs(self, g, crop_radius=None):
        radius = float(crop_radius if crop_radius is not None else getattr(self, "comm_range", 50.0))

        x_full = g["node_features"]
        ei_full = g["edge_index"]
        self_index = np.asarray(g["self_index"], dtype=np.int64)
        B = int(self.cav_num)

        valid_e = (ei_full[0] >= 0) & (ei_full[1] >= 0)
        edge_index = ei_full[:, valid_e].astype(np.int64)
        max_e = int(edge_index.max()) if edge_index.size > 0 else -1
        max_si = int(np.max(self_index[:B])) if B > 0 else -1
        N = min(x_full.shape[0], max(max_e, max_si) + 1)
        X = x_full[:N]
        F = X.shape[1]

        nodes_list, edges_list, batches, self_locals = [], [], [], []
        node_offset = 0
        per_agent_node_sets = []

        for a in range(int(self.cav_num)):
            c = int(self_index[a]) if a < self_index.size else 0
            c = max(0, min(c, N - 1))

            if edge_index.size == 0:
                sel = np.array([c], dtype=np.int64)
            else:
                mask_c = (edge_index[0] == c) | (edge_index[1] == c)
                if np.any(mask_c):
                    partners = np.where(edge_index[0, mask_c] == c, edge_index[1, mask_c], edge_index[0, mask_c])
                    sel = np.unique(np.concatenate([np.array([c], dtype=np.int64), partners]))
                else:
                    sel = np.array([c], dtype=np.int64)

            per_agent_node_sets.append(sel)

            map_old2new = -np.ones(N, dtype=np.int64)
            map_old2new[sel] = np.arange(sel.shape[0], dtype=np.int64)

            Xa = X[sel]
            nodes_list.append(Xa)
            batches.append(np.full((Xa.shape[0],), a, dtype=np.int64))
            self_locals.append(int(map_old2new[c]))

            if edge_index.size > 0:
                keep = np.isin(edge_index[0], sel) & np.isin(edge_index[1], sel)
                e_local = edge_index[:, keep]
                e_local = map_old2new[e_local] + node_offset
            else:
                e_local = np.zeros((2, 0), dtype=np.int64)
            edges_list.append(e_local)

            node_offset += Xa.shape[0]

        if nodes_list:
            Xa_all = np.concatenate(nodes_list, axis=0).astype(np.float32)
            Ea_all = (
                np.concatenate(edges_list, axis=1).astype(np.int64)
                if sum(e.shape[1] for e in edges_list) > 0 else np.zeros((2, 0), dtype=np.int64)
            )
            Ba_all = np.concatenate(batches, axis=0).astype(np.int64)
            Si_local = np.asarray(self_locals, dtype=np.int64)
        else:
            Xa_all = np.zeros((0, F), dtype=np.float32)
            Ea_all = np.zeros((2, 0), dtype=np.int64)
            Ba_all = np.zeros((0,), dtype=np.int64)
            Si_local = np.zeros((B,), dtype=np.int64)

        actor_graph = {
            "node_features": Xa_all,
            "edge_index": Ea_all,
            "batch": Ba_all,
            "self_index": Si_local
        }

        if len(per_agent_node_sets) > 0:
            union_nodes = np.unique(np.concatenate(per_agent_node_sets))
        else:
            union_nodes = np.arange(min(N, B), dtype=np.int64)

        map_union = -np.ones(N, dtype=np.int64)
        map_union[union_nodes] = np.arange(union_nodes.shape[0], dtype=np.int64)
        Xc_single = X[union_nodes]

        if edge_index.size > 0:
            keep_u = np.isin(edge_index[0], union_nodes) & np.isin(edge_index[1], union_nodes)
            Eic_single = map_union[edge_index[:, keep_u]]
        else:
            Eic_single = np.zeros((2, 0), dtype=np.int64)

        Nu = Xc_single.shape[0]
        if Nu > 0:
            Xc = np.repeat(Xc_single[None, :, :], B, axis=0).reshape(B * Nu, F)
            if Eic_single.size > 0:
                offsets = (np.arange(B, dtype=np.int64) * Nu).reshape(B, 1)
                Eic = np.tile(Eic_single[None, :, :], (B, 1, 1))
                Eic[:, 0, :] += offsets
                Eic[:, 1, :] += offsets
                Eic = Eic.reshape(2, -1)
            else:
                Eic = np.zeros((2, 0), dtype=np.int64)
            batch_c = np.repeat(np.arange(B, dtype=np.int64), Nu)
        else:
            Xc = np.zeros((0, F), dtype=np.float32)
            Eic = np.zeros((2, 0), dtype=np.int64)
            batch_c = np.zeros((0,), dtype=np.int64)

        critic_graph = {
            "node_features": Xc,
            "edge_index": Eic,
            "batch": batch_c
        }

        return actor_graph, critic_graph

    def save(self, episode):
        save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)

        policy_actor = self.trainer.policy.actor
        actor_path = f"{save_dir}/actor_{episode}.pt"
        torch.save(policy_actor.state_dict(), actor_path)

        policy_critic = self.trainer.policy.critic
        critic_path = f"{save_dir}/critic_{episode}.pt"
        torch.save(policy_critic.state_dict(), critic_path)
