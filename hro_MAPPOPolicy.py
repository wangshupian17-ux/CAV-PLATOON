# -*- coding: utf-8 -*-
# @Author  : wang
# @Time    : 2025/9/8 10:52
# @File    : hro_MAPPOPolicy.py
# @Software: PyCharm

import torch
from actor_critic_hro import R_Actor, R_Critic
from utils_mappo.util_mappo import update_linear_schedule


class hro_MAPPOPolicy:
    """
    MAPPO policy class. Wraps the actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) Arguments containing relevant model and policy configuration.
    :param obs_space: (gym.Space) Observation space.
    :param share_obs_space: (gym.Space) Input space for the value function
        (centralized input for MAPPO, decentralized input for IPPO).
    :param action_space: (gym.Space) Action space.
    :param device: (torch.device) Device used for computation (cpu/gpu).
    """

    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr_actor
        self.critic_lr = args.lr_critic
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.v_global_coef = float(getattr(args, "global_coef", 0.5))
        self.v_local_coef = 1 - self.v_global_coef

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)
        self.local_critic = R_Critic(args, self.obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay
        )
        critic_params = list(self.critic.parameters()) + list(self.local_critic.parameters())
        self.critic_optimizer = torch.optim.Adam(
            critic_params,
            lr=args.lr_critic, eps=args.opti_eps, weight_decay=args.weight_decay
        )

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_goble_values(self, cent_obs, rnn_states_critic, masks):
        goble_values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return goble_values, rnn_states_critic

    def get_local_values(self, obs_tensor, rnn_states_actor, masks):
        local_values, rnn_states_critic = self.local_critic(obs_tensor, rnn_states_actor, masks)
        return local_values, rnn_states_critic

    def evaluate_actions(self, share_obs_batch, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions, active_masks=None):
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs,
            rnn_states_actor,
            action,
            masks,
            available_actions,
            active_masks
        )

        values_global, _ = self.critic(share_obs_batch, rnn_states_critic, masks)
        values_local, _ = self.local_critic(obs, rnn_states_actor, masks)
        return action_log_probs, dist_entropy, values_global, values_local

    def evaluate_actions_graph(self, actor_graph: dict, rnn_states_actor, action, masks,
                               available_actions=None, active_masks=None):
        """
        Reuse the actor's evaluate_actions directly with local graph input.
        Returns values consistent with the current implementation.
        """
        return self.actor.evaluate_actions(
            actor_graph, rnn_states_actor, action, masks,
            available_actions=available_actions, active_masks=active_masks
        )

    def get_actions_graph(self, critic_graph: dict, actor_graph: dict,
                          rnn_states_actor, rnn_states_critic, masks,
                          available_actions=None, deterministic=False):
        """
        critic_graph: {'node_features','edge_index','batch'}
        actor_graph : {'node_features','edge_index','batch','self_index'}
        Returns: values_global, values_local, actions, logp, rnn_states_actor_new, rnn_states_critic_new
        """
        actions, action_log_probs, rnn_states_actor_new = self.actor(
            actor_graph, rnn_states_actor, masks,
            available_actions=available_actions, deterministic=deterministic
        )

        values_global, rnn_states_critic_new = self.critic(critic_graph, rnn_states_critic, masks)
        values_local, _ = self.local_critic(actor_graph, rnn_states_actor, masks)

        return values_global, values_local, actions, action_log_probs, rnn_states_actor_new, rnn_states_critic_new

    def get_global_values_graph(self, critic_graph: dict, rnn_states_critic, masks):
        values_global, rnn_states_critic_new = self.critic(critic_graph, rnn_states_critic, masks)
        return values_global, rnn_states_critic_new

    def get_local_values_graph(self, actor_graph: dict, rnn_states_actor, masks):
        values_local, _ = self.local_critic(actor_graph, rnn_states_actor, masks)
        return values_local