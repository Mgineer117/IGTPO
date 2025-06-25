import os
import pickle
import time
from copy import deepcopy
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch import inverse, matmul, transpose
from torch.autograd import grad

from policy.layers.base import Base
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from utils.intrinsic_rewards import IntrinsicRewardFunctions
from utils.rl import (
    compute_kl,
    conjugate_gradients,
    estimate_advantages,
    flat_params,
    hessian_vector_product,
    set_flat_params,
)
from utils.sampler import OnlineSampler


class IGTPO_Learner(Base):
    def __init__(
        self,
        actor: PPO_Actor,
        critic: PPO_Critic,
        intrinsic_reward_fn: IntrinsicRewardFunctions,
        nupdates: int,
        num_inner_updates: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        eps_clip: float = 0.2,
        entropy_scaler: float = 1e-3,
        target_kl: float = 0.03,
        l2_reg: float = 1e-8,
        gamma: float = 0.99,
        gae: float = 0.9,
        device: str = "cpu",
    ):
        super(IGTPO_Learner, self).__init__()

        # constants
        self.name = "IGTPO"
        self.device = device

        self.state_dim = actor.state_dim
        self.action_dim = actor.action_dim
        self.num_inner_updates = num_inner_updates

        self.init_actor_lr = actor_lr
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.nupdates = nupdates
        self.entropy_scaler = entropy_scaler
        self.gamma = gamma
        self.gae = gae
        self.eps_clip = eps_clip
        self.init_target_kl = target_kl
        self.target_kl = target_kl
        self.l2_reg = l2_reg

        # define nn.Module
        self.actor = actor
        self.critic = critic
        self.intrinsic_reward_fn = intrinsic_reward_fn

        self.extrinsic_critics = nn.ModuleList(
            [deepcopy(self.critic) for _ in range(self.intrinsic_reward_fn.num_rewards)]
        )
        self.intrinsic_critics = nn.ModuleList(
            [deepcopy(self.critic) for _ in range(self.intrinsic_reward_fn.num_rewards)]
        )

        self.extrinsic_critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=self.critic_lr)
            for critic in self.extrinsic_critics
        ]
        self.intrinsic_critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=self.critic_lr)
            for critic in self.intrinsic_critics
        ]

        #
        self.steps = 0
        self.epsilon = 0.3
        self.contributing_indices = [
            str(i) for i in range(self.intrinsic_reward_fn.num_rewards)
        ]
        self.init_avg_intrinsic_rewards = {}
        self.final_avg_intrinsic_rewards = {}
        self.probability_history = np.array(
            [0.0 for _ in range(self.intrinsic_reward_fn.num_rewards)]
        )
        self.to(self.dtype).to(self.device)

    def lr_scheduler(self, fraction: float):
        self.actor_lr = (self.init_actor_lr - 1e-4) * (1.0 - fraction) + 1e-4
        self.steps += 1

    def prune(self):
        # np.all(self.probability_history > 0.0) and len(self.probability_history) > 1
        if len(self.probability_history) > 1:  #   and len(tied_indices) == 1:
            least_contributing_index = np.argmin(self.probability_history)

            # === PRUNE CRITICS === #
            del self.extrinsic_critics[least_contributing_index]
            del self.intrinsic_critics[least_contributing_index]

            del self.extrinsic_critic_optimizers[least_contributing_index]
            del self.intrinsic_critic_optimizers[least_contributing_index]

            # === PRUNE ETC. === #
            del self.contributing_indices[least_contributing_index]

            self.probability_history = np.delete(
                self.probability_history, least_contributing_index
            )

            # === SEND PRUNE SIGNAL TO INTRINSIC REWARD CLASS === #
            self.intrinsic_reward_fn.prune(least_contributing_index)

    def trim(self):
        if self.num_inner_updates > 2:
            self.num_inner_updates -= 1

    def forward(self, state: np.ndarray, deterministic: bool = False):
        state = self.preprocess_state(state)
        a, metaData = self.actor(state, deterministic=deterministic)

        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
            "dist": metaData["dist"],
        }

    def learn(
        self,
        env: gym.Env,
        outer_sampler: OnlineSampler,
        inner_sampler: OnlineSampler,
        seed: int,
        fraction: float,
    ):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()

        total_timesteps, total_sample_time, total_update_time = 0, 0, 0
        policy_dict, gradient_dict = {}, {}

        # === initialize the policy dict with outer-level policy === #
        for i in range(self.intrinsic_reward_fn.num_rewards):
            actor_idx = f"{i}_{0}"
            policy_dict[actor_idx] = deepcopy(self.actor)

        # === SAMPLE FOR INITIAL UPDATE === #
        init_batch, sample_time = inner_sampler.collect_samples(env, self.actor, seed)
        self.actor.record_state_visitations(init_batch["states"], alpha=1.0)
        total_timesteps += init_batch["states"].shape[0]

        # === UPDATE VIA GRADIENT CHAIN === #
        loss_dict_list = []
        for i in range(self.intrinsic_reward_fn.num_rewards):
            for j in range(self.num_inner_updates):
                # decide update
                prefix = "outer" if j == self.num_inner_updates - 1 else "inner"

                # choose actor
                actor_idx = f"{i}_{j}"
                future_actor_idx = f"{i}_{j+1}"
                actor = policy_dict[actor_idx]

                if prefix == "inner":
                    if j == 0:
                        batch = deepcopy(init_batch)
                        sample_time = 0
                        timesteps = 0
                    else:
                        batch, sample_time = inner_sampler.collect_samples(
                            env, actor, seed
                        )
                        timesteps = batch["states"].shape[0]
                else:  # prefix == "outer"
                    batch, sample_time = outer_sampler.collect_samples(env, actor, seed)
                    actor.record_state_visitations(batch["states"], alpha=1.0)
                    timesteps = batch["states"].shape[0]

                beta = 0.95
                self.probability_history[i] = (
                    beta * self.probability_history[i]
                    + (1 - beta) * batch["rewards"].mean()
                )

                (
                    loss_dict,
                    update_time,
                    actor_clone,
                    gradients,
                    avg_intrinsic_rewards,
                ) = self.learn_model(actor, batch, i, fraction, prefix)

                # logging
                loss_dict_list.append(loss_dict)

                gradient_dict[actor_idx] = gradients
                policy_dict[future_actor_idx] = actor_clone

                if j == 0:
                    self.init_avg_intrinsic_rewards[str(i)] = avg_intrinsic_rewards
                elif j == self.num_inner_updates - 1:
                    self.final_avg_intrinsic_rewards[str(i)] = avg_intrinsic_rewards

                total_timesteps += timesteps
                total_sample_time += sample_time
                total_update_time += update_time

        # === Meta-gradient computation === #
        outer_gradients = []
        for i in range(self.intrinsic_reward_fn.num_rewards):
            gradients = gradient_dict[f"{i}_{self.num_inner_updates - 1}"]
            for j in reversed(range(self.num_inner_updates - 1)):
                iter_idx = f"{i}_{j}"
                Hv = grad(
                    gradient_dict[iter_idx],
                    policy_dict[iter_idx].parameters(),
                    grad_outputs=gradients,
                )
                gradients = tuple(g - self.actor_lr * h for g, h in zip(gradients, Hv))
            outer_gradients.append(gradients)

        # Average across vectors
        most_contributing_index = np.argmax(self.probability_history)
        outer_gradients_transposed = list(zip(*outer_gradients))  # Group by parameter
        # gradients = outer_gradients[most_contributing_index]

        prob = np.random.rand()
        epsilon = self.epsilon * (1 - fraction)

        if prob < epsilon:
            # Uniform exploration
            # weights = np.ones_like(self.probability_history)
            # Random exploration
            weights = np.random.rand(len(self.probability_history))
            weights = weights / (weights.sum() + 1e-8)
        else:
            # Exploitation: use normalized probability_history
            weights = self.probability_history
            weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
            weights = weights / (weights.sum() + 1e-8)

        gradients = tuple(
            sum(w * g for w, g in zip(weights, grads_per_param))
            for grads_per_param in outer_gradients_transposed
        )

        # === TRPO update === #
        backtrack_iter, backtrack_success = self.learn_outer_level_polocy(
            states=init_batch["states"],
            grads=gradients,
        )

        # === COLLECT VISITATION MAPS === #
        visitation_dict = {}
        visitation_dict["visitation map (outer)"] = self.actor.state_visitation
        for i in range(self.intrinsic_reward_fn.num_rewards):
            idx = f"{i}_{self.num_inner_updates}"
            name = f"visitation map ({self.contributing_indices[i]})"
            visitation_dict[name] = policy_dict[idx].state_visitation

        # === Logging === #
        loss_dict = self.average_dict_values(loss_dict_list)
        loss_dict[f"{self.name}/analytics/avg_extrinsic_rewards"] = init_batch[
            "rewards"
        ].mean()
        loss_dict[f"{self.name}/analytics/sample_time"] = total_sample_time
        loss_dict[f"{self.name}/analytics/update_time"] = total_update_time
        loss_dict[f"{self.name}/parameters/num vectors"] = (
            self.intrinsic_reward_fn.num_rewards
        )
        loss_dict[f"{self.name}/parameters/num_inner_updates"] = self.num_inner_updates
        loss_dict[f"{self.name}/analytics/Contributing Option"] = int(
            self.contributing_indices[most_contributing_index]
        )
        loss_dict[f"{self.name}/analytics/Backtrack_iter"] = backtrack_iter
        loss_dict[f"{self.name}/analytics/Backtrack_success"] = backtrack_success
        loss_dict[f"{self.name}/analytics/target_kl"] = self.target_kl
        loss_dict[f"{self.name}/analytics/igtpo_actor_lr"] = self.actor_lr

        loss_dict.update(self.intrinsic_reward_fn.loss_dict)

        for i in range(self.intrinsic_reward_fn.num_rewards):
            intrinsic_avg_rewards_improvement = (
                self.final_avg_intrinsic_rewards[str(i)]
                - self.init_avg_intrinsic_rewards[str(i)]
            )
            loss_dict[f"{self.name}/analytics/R_int_improvement ({i})"] = (
                intrinsic_avg_rewards_improvement
            )

        return loss_dict, total_timesteps, visitation_dict

    def learn_model(
        self, actor: nn.Module, batch: dict, i: int, fraction: float, prefix: str
    ):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # 0. Prepare ingredients
        states = self.preprocess_state(batch["states"])
        next_states = self.preprocess_state(batch["next_states"])
        actions = self.preprocess_state(batch["actions"])
        extrinsic_rewards = self.preprocess_state(batch["rewards"])
        intrinsic_rewards, source = self.intrinsic_reward_fn(states, next_states, i)
        terminals = self.preprocess_state(batch["terminals"])
        old_logprobs = self.preprocess_state(batch["logprobs"])

        # 1. Compute advantages and returns
        with torch.no_grad():
            extrinsic_values = self.extrinsic_critics[i](states)
            extrinsic_advantages, extrinsic_returns = estimate_advantages(
                extrinsic_rewards,
                terminals,
                extrinsic_values,
                gamma=self.gamma,
                gae=self.gae,
            )
        with torch.no_grad():
            intrinsic_values = self.intrinsic_critics[i](states)
            if source == "drnd":
                intrinsic_advantages, intrinsic_returns = estimate_advantages(
                    intrinsic_rewards,
                    torch.zeros_like(terminals),
                    intrinsic_values,
                    gamma=self.gamma,
                    gae=self.gae,
                )
            else:
                intrinsic_advantages, intrinsic_returns = estimate_advantages(
                    intrinsic_rewards,
                    terminals,
                    intrinsic_values,
                    gamma=self.gamma,
                    gae=self.gae,
                )

        # 1. Learn critic
        critics = [self.extrinsic_critics[i], self.intrinsic_critics[i]]
        critic_optims = [
            self.extrinsic_critic_optimizers[i],
            self.intrinsic_critic_optimizers[i],
        ]
        critic_targets = [extrinsic_returns, intrinsic_returns]
        critic_iteration = 5
        extrinsic_critic_loss = None
        intrinsic_critic_loss = None
        batch_size = states.shape[0]
        for critic, optim, returns in zip(critics, critic_optims, critic_targets):
            losses = []
            perm = torch.randperm(batch_size)
            mb_size = batch_size // critic_iteration
            for j in range(critic_iteration):
                indices = perm[j * mb_size : (j + 1) * mb_size]
                critic_loss = self.critic_loss(
                    critic, states[indices], returns[indices]
                )
                optim.zero_grad()
                critic_loss.backward()
                optim.step()
                losses.append(critic_loss.item())

            avg_loss = sum(losses) / len(losses)
            if extrinsic_critic_loss is None:
                extrinsic_critic_loss = avg_loss
            else:
                intrinsic_critic_loss = avg_loss

        # 2. Learn actor
        actor_clone = deepcopy(actor)  # self.clone_actor()  # clone for future update

        if prefix == "outer":  #  and np.any(self.probability_history > 1e-6):
            # advantages = (
            #     fraction * extrinsic_advantages + (1 - fraction) * intrinsic_advantages
            # )
            # advantages = (1 - fraction) * intrinsic_advantages + extrinsic_advantages
            advantages = extrinsic_advantages
        else:
            advantages = intrinsic_advantages

        # advantages = extrinsic_advantages if prefix == "outer" else intrinsic_advantages
        normalized_advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        # 3. actor Loss
        actor_loss, entropy_loss = self.actor_loss(
            actor, states, actions, old_logprobs, normalized_advantages
        )

        # 4. Total loss
        # if prefix == "outer":
        # entropy_loss *= 0.0

        loss = actor_loss - entropy_loss

        # 5. Compute gradients (example)
        gradients = torch.autograd.grad(loss, actor.parameters(), create_graph=True)
        gradients = self.clip_grad_norm(gradients, max_norm=0.5)

        # 6. Manual SGD update (structured, not flat)
        with torch.no_grad():
            for p, g in zip(actor_clone.parameters(), gradients):
                p -= self.actor_lr * g

        # 8. Logging
        actor_grad_norm = torch.sqrt(
            sum(g.pow(2).sum() for g in gradients if g is not None)
        )

        # 9. do update of intrinsic reward if necessary
        self.intrinsic_reward_fn.learn(states, next_states, i, source)

        loss_dict = {
            f"{self.name}/loss/loss": loss.item(),
            f"{self.name}/loss/actor_loss": actor_loss.item(),
            f"{self.name}/loss/extrinsic_critic_loss": extrinsic_critic_loss,
            f"{self.name}/loss/intrinsic_critic_loss": intrinsic_critic_loss,
            f"{self.name}/loss/entropy_loss": entropy_loss.item(),
            f"{self.name}/grad/actor": actor_grad_norm.item(),
            f"{self.name}/analytics/avg_intrinsic_rewards ({self.contributing_indices[i]})": torch.mean(
                intrinsic_rewards
            ).item(),
        }
        norm_dict = self.compute_weight_norm(
            [actor],
            ["actor"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(norm_dict)

        self.eval()
        update_time = time.time() - t0

        return (
            loss_dict,
            update_time,
            actor_clone,
            gradients,
            torch.mean(intrinsic_rewards).item(),
        )

    def learn_outer_level_polocy(
        self,
        states: np.ndarray,
        grads: tuple[torch.Tensor],
        damping: float = 1e-1,
        backtrack_iters: int = 15,
        backtrack_coeff: float = 0.7,
    ):
        states = self.preprocess_state(states)

        old_actor = deepcopy(self.actor)

        # Flatten meta-gradients
        meta_grad_flat = torch.cat([g.view(-1) for g in grads]).detach()

        # KL function (closure)
        def kl_fn():
            return compute_kl(old_actor, self.actor, states)

        # Define HVP function
        Hv = lambda v: hessian_vector_product(kl_fn, self.actor, damping, v)

        # Compute step direction with CG
        step_dir = conjugate_gradients(Hv, meta_grad_flat, nsteps=10)

        # Compute step size to satisfy KL constraint
        sAs = 0.5 * torch.dot(step_dir, Hv(step_dir))
        lm = torch.sqrt(sAs / self.target_kl)
        full_step = step_dir / (lm + 1e-8)

        # Apply update
        with torch.no_grad():
            old_params = flat_params(self.actor)

            # Backtracking line search
            success = False
            for i in range(backtrack_iters):
                step_frac = backtrack_coeff**i
                new_params = old_params - step_frac * full_step
                set_flat_params(self.actor, new_params)
                kl = compute_kl(old_actor, self.actor, states)

                if kl <= self.target_kl:
                    success = True
                    break

            if not success:
                set_flat_params(self.actor, old_params)

        return i, success

    def clone_actor(self, actor: nn.Module):
        """
        Safe cloning of actor model for multiprocessing (CUDA-safe if run before CUDA init):
        - Avoids in-place .to("cpu") on original model.
        - Does not require original model to move off-device.
        """
        # Create a new clone on CPU
        actor_clone = PPO_Actor(
            input_dim=actor.state_dim,
            hidden_dim=actor.hidden_dim,
            action_dim=actor.action_dim,
            is_discrete=actor.is_discrete,
            activation=nn.Tanh(),
            device=torch.device("cpu"),  # <- important
        )
        # Copy weights from actor (which may be on GPU)
        actor_clone.load_state_dict({k: v.cpu() for k, v in actor.state_dict().items()})
        actor_clone.device = self.device

        # Move clone to target device (if needed)
        return actor_clone.to(self.device)

    def actor_loss(
        self,
        actor: nn.Module,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
    ):
        # === PPO UPDATE STYLE === #
        # off-policy correction term should be 1
        _, metaData = actor(states)
        logprobs = actor.log_prob(metaData["dist"], actions)
        entropy = actor.entropy(metaData["dist"])
        ratios = torch.exp(logprobs - old_logprobs)

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        actor_loss = -(logprobs * advantages).mean()
        entropy_loss = self.entropy_scaler * entropy.mean()

        return actor_loss, entropy_loss

    def critic_loss(
        self, critic: nn.Module, states: torch.Tensor, returns: torch.Tensor
    ):
        values = critic(states)
        value_loss = self.mse_loss(values, returns)

        l2_loss = sum(param.pow(2).sum() for param in critic.parameters()) * self.l2_reg

        value_loss += l2_loss

        return value_loss

    def clip_grad_norm(self, grads, max_norm, eps=1e-6):
        # Compute total norm
        total_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
        clip_coef = max_norm / (total_norm + eps)

        if clip_coef < 1:
            grads = tuple(g * clip_coef for g in grads)

        return grads
