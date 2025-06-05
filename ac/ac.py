import hydra
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim, std=0.1):
        super().__init__()

        self.std = std
        self.policy = nn.Sequential(nn.Linear(obs_shape[0], hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs):
        mu = self.policy(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * self.std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_shape, action_shape, num_critics,
                 hidden_dim):
        super().__init__()

        self.critics = nn.ModuleList([nn.Sequential(
            nn.Linear(obs_shape[0] + action_shape[0], hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
            for _ in range(num_critics)])

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h_action = torch.cat([obs, action], dim=-1)
        return [critic(h_action) for critic in self.critics]


class ACAgent:
    def __init__(self,
                 obs_shape, action_shape, device, lr, hidden_dim,
                 num_critics,               # ← as before
                 critic_target_tau,
                 stddev_clip,
                 use_tb,
                 target_mode='double',      # 'double' | 'min_k' | 'min_all'
                 target_k=4):               # k for 'min_k'
        
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.stddev_clip = stddev_clip
        self.target_mode = target_mode
        self.target_k = target_k
        self.num_critics = num_critics

        # models
        self.actor = Actor(obs_shape, action_shape,
                           hidden_dim).to(device)

        self.critic = Critic(obs_shape, action_shape,
                             num_critics, hidden_dim).to(device)
        self.critic_target = Critic(obs_shape, action_shape,
                                    num_critics, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        dist = self.actor(obs.unsqueeze(0))
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
        return action.cpu().numpy()[0]

    def update_critic(self, replay_iter):
        '''
        This function updates the critic and target critic parameters.

        Args:

        replay_iter:
            An iterable that produces batches of tuples
            (observation, action, reward, discount, next_observation),
            where:
            observation: array of shape [batch, D] of states
            action: array of shape [batch, action_dim]
            reward: array of shape [batch,]
            discount: array of shape [batch,]
            next_observation: array of shape [batch, D] of states

        Returns:

        metrics: dictionary of relevant metrics to be logged. Add any metrics
                 that you find helpful to log for debugging, such as the critic
                 loss, or the mean Bellman targets.
        '''

        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        ### YOUR CODE HERE ###
        # ── compute Bellman targets  Q̂  = r + γ·min_i Q_target_i(s′,a′) ──
        with torch.no_grad():
            # 1) sample next action
            next_action = self.actor(next_obs).sample(clip=self.stddev_clip)
            h_next      = torch.cat([next_obs, next_action], dim=-1)

            # 2) gather Q-values from all heads once  (shape [B, N])
            q_targets = torch.cat(
                [c(h_next) for c in self.critic_target.critics], dim=-1
            )
            
            # 3) choose how to aggregate
            if self.target_mode == 'double':
                # min over two random heads  (classic TD3 / double-Q)
                i, j = random.sample(range(q_targets.shape[-1]), k=2)
                target_q = torch.min(q_targets[:, i:i+1], q_targets[:, j:j+1])

            elif self.target_mode == 'min_k':
                k = min(self.target_k, q_targets.shape[-1])
                idx      = torch.randperm(q_targets.shape[-1])[:k]
                target_q = q_targets[:, idx].min(dim=-1, keepdim=True)[0]

            else:          # 'min_all'
                target_q = q_targets.min(dim=-1, keepdim=True)[0]

            # 4) Bellman target
            target = reward + discount * target_q

        
        # ── critic loss  (1/N_critics) Σ‖Q_i(s,a) − Q̂‖² ─────────────────
        current_Qs  = self.critic(obs, action)
        critic_loss = sum(F.mse_loss(q, target) for q in current_Qs) / len(current_Qs)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── soft-update target network  θ′ ← τθ + (1−τ)θ′ ────────────────
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        metrics["critic_loss"]   = critic_loss.item()
        metrics["target_Q_mean"] = target.mean().item()
        #####################
        return metrics

    def update_actor(self, replay_iter):
        '''
        This function updates the policy parameters.

        Args:

        replay_iter:
            An iterable that produces batches of tuples
            (observation, action, reward, discount, next_observation),
            where:
            observation: array of shape [batch, D] of states
            action: array of shape [batch, action_dim]
            reward: array of shape [batch,]
            discount: array of shape [batch,]
            next_observation: array of shape [batch, D] of states

        Returns:

        metrics: dictionary of relevant metrics to be logged. Add any metrics
                 that you find helpful to log for debugging, such as the actor
                 loss.
        '''
        metrics = dict()

        batch = next(replay_iter)
        obs, _, _, _, _ = utils.to_torch(
            batch, self.device)

        ### YOUR CODE HERE ###
        dist   = self.actor(obs)
        action = dist.sample(clip=None)                # differentiable through μ
        q_vals = self.critic(obs, action)
        q_min  = torch.min(torch.cat(q_vals, dim=-1), dim=-1)[0]

        actor_loss = (-q_min).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        metrics["actor_loss"] = actor_loss.item()
        return metrics

    def bc(self, replay_iter):
        '''
        This function updates the policy with end-to-end
        behavior cloning

        Args:

        replay_iter:
            An iterable that produces batches of tuples
            (observation, action, reward, discount, next_observation),
            where:
            observation: array of shape [batch, D] of states
            action: array of shape [batch, action_dim]
            reward: array of shape [batch,]
            discount: array of shape [batch,]
            next_observation: array of shape [batch, D] of states

        Returns:

        metrics: dictionary of relevant metrics to be logged. Add any metrics
                 that you find helpful to log for debugging, such as the loss.
        '''

        metrics = dict()

        batch = next(replay_iter)
        obs, action, _, _, _ = utils.to_torch(batch, self.device)

        ### YOUR CODE HERE ###
        dist     = self.actor(obs)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)  # per-sample LL
        bc_loss  = (-log_prob).mean()                           # maximise LL

        self.actor_opt.zero_grad()
        bc_loss.backward()
        self.actor_opt.step()

        metrics["bc_loss"] = bc_loss.item()
        return metrics
