import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call

from policy.layers.base import Base
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from utils.rl import (
    compute_kl,
    conjugate_gradients,
    estimate_advantages,
    flat_params,
    hessian_vector_product,
    set_flat_params,
)


class PSNE_Learner(Base):
    def __init__(
        self,
        actor: PPO_Actor,
        critic: PPO_Critic,
        states: np.ndarray,
        nupdates: int,
        critic_lr: float = 5e-4,
        entropy_scaler: float = 1e-3,
        batch_size: int = 8,
        l2_reg: float = 1e-8,
        target_kl: float = 0.03,
        damping: float = 1e-1,
        backtrack_iters: int = 10,
        backtrack_coeff: float = 0.8,
        gamma: float = 0.99,
        gae: float = 0.9,
        device: str = "cpu",
    ):
        super().__init__(device=device)

        # constants
        self.name = "PSNE"
        self.device = device

        self.states = self.preprocess_state(states)
        self.state_dim = actor.state_dim
        self.action_dim = actor.action_dim

        self.entropy_scaler = entropy_scaler
        self.batch_size = batch_size
        self.damping = damping
        self.gamma = gamma
        self.gae = gae
        self.l2_reg = l2_reg
        self.init_target_kl = target_kl
        self.target_kl = target_kl
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.nupdates = nupdates

        # trainable networks
        self.actor = actor
        self.sampled_actor = deepcopy(self.actor)

        self.critic = critic
        self.optimizer = torch.optim.Adam(params=self.critic.parameters(), lr=critic_lr)

        #
        self.steps = 0
        self.to(self.dtype).to(self.device)
        self.sample_policy()

    def lr_scheduler(self):
        self.target_kl = self.init_target_kl * (1 - self.steps / self.nupdates)
        self.steps += 1

    def sample_policy(self):
        # Backtracking line search
        with torch.no_grad():
            old_params = flat_params(self.actor)
            epsilon = torch.randn_like(old_params)

            # Backtracking line search
            success = False
            for i in range(self.backtrack_iters):
                # we use 0.1 since epsilon is quite large
                step_frac = 0.1**i
                new_params = old_params - step_frac * epsilon
                set_flat_params(self.sampled_actor, new_params)
                kl = compute_kl(self.actor, self.sampled_actor, self.states)

                if kl <= self.target_kl:
                    success = True
                    break

            if not success:
                set_flat_params(self.sampled_actor, old_params)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        state = self.preprocess_state(state)
        a, metaData = self.sampled_actor(state, deterministic=deterministic)

        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
            "dist": metaData["dist"],
        }

    def learn(self, batch):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        states = self.preprocess_state(batch["states"])
        actions = self.preprocess_state(batch["actions"])
        rewards = self.preprocess_state(batch["rewards"])
        terminals = self.preprocess_state(batch["terminals"])
        old_logprobs = self.preprocess_state(batch["logprobs"])

        self.record_state_visitations(states)

        # Compute advantages and returns
        with torch.no_grad():
            values = self.critic(states)
            advantages, returns = estimate_advantages(
                rewards,
                terminals,
                values,
                gamma=self.gamma,
                gae=self.gae,
            )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_gradients, actor_loss, entropy_loss = self.actor_loss(
            states, actions, old_logprobs, advantages
        )

        # === actor trpo update === #
        old_actor = deepcopy(self.actor)
        grad_flat = torch.cat([g.view(-1) for g in actor_gradients]).detach()

        # KL function (closure)
        def kl_fn():
            return compute_kl(old_actor, self.actor, states)

        # Define HVP function
        Hv = lambda v: hessian_vector_product(kl_fn, self.actor, self.damping, v)

        # Compute step direction with CG
        step_dir = conjugate_gradients(Hv, grad_flat, nsteps=10)

        # Compute step size to satisfy KL constraint
        sAs = 0.5 * torch.dot(step_dir, Hv(step_dir))
        lm = torch.sqrt(sAs / self.target_kl)
        full_step = step_dir / (lm + 1e-8)

        # Apply update
        with torch.no_grad():
            old_params = flat_params(self.actor)

            # Backtracking line search
            success = False
            for i in range(self.backtrack_iters):
                step_frac = self.backtrack_coeff**i
                new_params = old_params - step_frac * full_step
                set_flat_params(self.actor, new_params)
                kl = compute_kl(old_actor, self.actor, states)

                if kl <= self.target_kl:
                    success = True
                    break

            if not success:
                set_flat_params(self.actor, old_params)

        # self.lr_scheduler()
        # given the update of new actor, sample a actor for exploration
        self.sample_policy()
        self.states = states

        # === critic update === #
        value_loss, l2_loss = self.critic_loss(states, returns)
        loss = (
            value_loss + l2_loss + actor_loss + entropy_loss
        )  # actor_loss is already detached
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.optimizer.step()

        # Logging
        loss_dict = {
            f"{self.name}/loss/loss": loss.item(),
            f"{self.name}/loss/actor_loss": actor_loss.item(),
            f"{self.name}/loss/entropy_loss": entropy_loss.item(),
            f"{self.name}/loss/value_loss": value_loss.item(),
            f"{self.name}/loss/l2_loss": l2_loss.item(),
            f"{self.name}/analytics/grad_norm (actor)": torch.linalg.norm(
                grad_flat
            ).item(),
            f"{self.name}/analytics/backtrack_iter": i,
            f"{self.name}/analytics/backtrack_success": int(success),
            f"{self.name}/analytics/klDivergence": kl.item(),
            f"{self.name}/analytics/avg_rewards": torch.mean(rewards).item(),
            f"{self.name}/analytics/target_kl": self.target_kl,
            f"{self.name}/analytics/critic_lr": self.optimizer.param_groups[0]["lr"],
        }
        norm_dict = self.compute_weight_norm(
            [self.sampled_actor, self.critic],
            ["actor", "critic"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(norm_dict)

        # Cleanup
        del states, actions, rewards, terminals, old_logprobs
        self.eval()

        timesteps = self.batch_size
        update_time = time.time() - t0

        return loss_dict, timesteps, update_time

    def actor_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
    ):
        _, metaData = self.actor(states)
        logprobs = self.actor.log_prob(metaData["dist"], actions)
        entropy = self.actor.entropy(metaData["dist"])
        ratios = torch.exp(logprobs - old_logprobs)

        surr = ratios * advantages
        actor_loss = -surr.mean()
        entropy_loss = self.entropy_scaler * entropy.mean()

        loss = actor_loss + entropy_loss

        # Compute gradients wrt mean and logstd predictors
        actor_gradients = torch.autograd.grad(
            loss, self.actor.parameters(), retain_graph=True
        )

        return actor_gradients, actor_loss.detach(), entropy_loss.detach()

    def critic_loss(self, states: torch.Tensor, returns: torch.Tensor):
        mb_values = self.critic(states)
        value_loss = self.mse_loss(mb_values, returns)
        l2_loss = (
            sum(param.pow(2).sum() for param in self.critic.parameters()) * self.l2_reg
        )

        return value_loss, l2_loss
