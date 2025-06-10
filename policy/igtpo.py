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
from utils.rl import (
    compute_kl,
    conjugate_gradients,
    estimate_advantages,
    flat_params,
    hessian_vector_product,
    set_flat_params,
)
from utils.sampler import OnlineSampler


def check_model_params_equal(model1, model2) -> bool:
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not torch.equal(p1, p2):
            return False
    return True


class IGTPO_Learner(Base):
    def __init__(
        self,
        extractor: nn.Module,
        eigenvectors: np.ndarray,
        actor: PPO_Actor,
        extrinsic_critics: list[PPO_Critic],
        intrinsic_critics: list[PPO_Critic],
        nupdates: int,
        num_vectors: int,
        num_inner_updates: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        batch_size: int = 256,
        eps_clip: float = 0.2,
        entropy_scaler: float = 1e-3,
        target_kl: float = 0.03,
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

        self.eigenvectors = torch.from_numpy(eigenvectors).to(self.device)
        self.num_vectors = num_vectors
        self.num_inner_updates = num_inner_updates

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.nupdates = nupdates
        self.batch_size = batch_size
        self.entropy_scaler = entropy_scaler
        self.gamma = gamma
        self.gae = gae
        self.eps_clip = eps_clip
        self.init_target_kl = target_kl
        self.target_kl = target_kl

        # define nn.Module
        self.extractor = extractor
        self.actor = actor

        self.extrinsic_critics = extrinsic_critics
        self.intrinsic_critics = intrinsic_critics

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
        self.probabilities = np.array([0.0 for _ in range(self.num_vectors)])
        self.contributing_indices = [str(i) for i in range(self.num_vectors)]
        self.to(self.dtype).to(self.device)

    def lr_scheduler(self, fraction: float):
        self.target_kl = self.init_target_kl * (1 - fraction)
        self.steps += 1

    def prune(self):
        if len(self.probabilities) > 1:
            least_contributing_index = np.argmin(self.probabilities)
            del self.extrinsic_critics[least_contributing_index]
            del self.intrinsic_critics[least_contributing_index]

            del self.extrinsic_critic_optimizers[least_contributing_index]
            del self.intrinsic_critic_optimizers[least_contributing_index]

            self.probabilities = np.delete(self.probabilities, least_contributing_index)
            self.eigenvectors = torch.cat(
                (
                    self.eigenvectors[:least_contributing_index],  # rows before index 4
                    self.eigenvectors[
                        least_contributing_index + 1 :
                    ],  # rows after index 4
                ),
                dim=0,
            )

            self.num_vectors -= 1

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

    def learn(self, env: gym.Env, sampler: OnlineSampler, seed: int):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()

        total_timesteps, total_sample_time, total_update_time = 0, 0, 0
        policy_dict, gradient_dict = {}, {}

        # === initialize the policy dict with outer-level policy === #
        for i in range(self.num_vectors):
            actor_idx = f"{i}_{0}"
            policy_dict[actor_idx] = self.clone_actor(self.actor)

        # === first iteration === #
        loss_dict_list = []
        for i in range(self.num_vectors):
            for j in range(self.num_inner_updates):
                # decide update
                prefix = "outer" if j == self.num_inner_updates - 1 else "inner"

                # choose actor
                actor_idx = f"{i}_{j}"
                future_actor_idx = f"{i}_{j+1}"

                actor = policy_dict[actor_idx]
                batch, sample_time = sampler.collect_samples(env, actor, seed)

                # save reward probability
                self.probabilities[i] += batch["rewards"].mean()
                (
                    loss_dict,
                    timesteps,
                    update_time,
                    actor_clone,
                    gradients,
                ) = self.learn_model(actor, batch, i, prefix)

                # logging
                loss_dict_list.append(loss_dict)

                gradient_dict[actor_idx] = gradients
                policy_dict[future_actor_idx] = actor_clone

                total_timesteps += timesteps
                total_sample_time += sample_time
                total_update_time += update_time

        # === Meta-gradient computation === #
        argmax_idx = np.argmax(self.probabilities)
        most_contributing_index = self.contributing_indices[argmax_idx]

        outer_gradients = []
        for i in range(self.num_vectors):
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
        outer_gradients_transposed = list(zip(*outer_gradients))  # Group by parameter
        gradients = tuple(
            torch.mean(torch.stack(grads_per_param), dim=0)
            for grads_per_param in outer_gradients_transposed
        )

        # gradients = outer_gradients[argmax_idx]

        # === TRPO update === #
        batch, sample_time = sampler.collect_samples(env, self.actor, seed)
        backtrack_iter, backtrack_success = self.learn_outer_level_polocy(
            states=batch["states"],
            grads=gradients,
        )

        loss_dict = self.average_dict_values(loss_dict_list)
        loss_dict[f"{self.name}/analytics/sample_time"] = total_sample_time
        loss_dict[f"{self.name}/analytics/update_time"] = total_update_time
        loss_dict[f"{self.name}/parameters/num vectors"] = self.num_vectors
        loss_dict[f"{self.name}/parameters/num_local_updates"] = self.num_inner_updates
        loss_dict[f"{self.name}/analytics/Contributing Option"] = int(
            most_contributing_index
        )
        loss_dict[f"{self.name}/analytics/Backtrack_iter"] = backtrack_iter
        loss_dict[f"{self.name}/analytics/Backtrack_success"] = backtrack_success
        loss_dict[f"{self.name}/analytics/target_kl"] = self.target_kl

        return loss_dict, total_timesteps

    def learn_model(self, actor: nn.Module, batch: dict, i: int, prefix: str):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # 0. Prepare ingredients
        states = self.preprocess_state(batch["states"])
        next_states = self.preprocess_state(batch["next_states"])
        actions = self.preprocess_state(batch["actions"])
        extrinsic_rewards = self.preprocess_state(batch["rewards"])
        intrinsic_rewards = self.intrinsic_rewards(
            states, next_states, self.eigenvectors[i]
        )
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

        for critic, optim, returns in zip(critics, critic_optims, critic_targets):
            losses = []
            perm = torch.randperm(self.batch_size)
            mb_size = self.batch_size // critic_iteration
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
        actor_clone = self.clone_actor(actor)  # clone for future update

        advantages = extrinsic_advantages if prefix == "outer" else intrinsic_advantages
        normalized_advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        # 3. actor Loss
        actor_loss, entropy_loss = self.actor_loss(
            actor, states, actions, old_logprobs, normalized_advantages
        )

        # 4. Total loss
        if prefix == "outer":
            entropy_loss *= 0.0  # No entropy loss for meta-IGTPO

        loss = actor_loss - entropy_loss

        # 5. Compute gradients (example)
        gradients = torch.autograd.grad(loss, actor.parameters(), create_graph=True)
        if prefix == "inner":
            gradients = self.clip_grad_norm(gradients, max_norm=0.5)

        # 6. Manual SGD update (structured, not flat)
        with torch.no_grad():
            for p, g in zip(actor_clone.parameters(), gradients):
                p -= self.actor_lr * g

        # 8. Logging
        actor_grad_norm = torch.sqrt(
            sum(g.pow(2).sum() for g in gradients if g is not None)
        )

        loss_dict = {
            f"{self.name}/loss/loss": loss.item(),
            f"{self.name}/loss/actor_loss": actor_loss.item(),
            f"{self.name}/loss/extrinsic_critic_loss": extrinsic_critic_loss,
            f"{self.name}/loss/intrinsic_critic_loss": intrinsic_critic_loss,
            f"{self.name}/loss/entropy_loss": entropy_loss.item(),
            f"{self.name}/grad/actor": actor_grad_norm.item(),
            f"{self.name}/analytics/avg_extrinsic_rewards": torch.mean(
                extrinsic_rewards
            ).item(),
            f"{self.name}/analytics/avg_intrinsic_rewards": torch.mean(
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

        timesteps = states.shape[0]
        update_time = time.time() - t0

        return (
            loss_dict,
            timesteps,
            update_time,
            actor_clone,
            gradients,
        )

    def learn_outer_level_polocy(
        self,
        states: np.ndarray,
        grads: tuple[torch.Tensor],
        damping: float = 1e-1,
        backtrack_iters: int = 10,
        backtrack_coeff: float = 0.8,
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

    def intrinsic_rewards(self, states, next_states, eigenvector):
        # get features
        with torch.no_grad():
            feature, _ = self.extractor(states)
            next_feature, _ = self.extractor(next_states)

            difference = next_feature - feature

        # Calculate the intrinsic reward using the eigenvector
        intrinsic_rewards = torch.matmul(difference, eigenvector.unsqueeze(-1))

        return intrinsic_rewards.to(self.device)

    def actor_loss(
        self,
        actor: nn.Module,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
    ):
        _, metaData = actor(states)
        logprobs = actor.log_prob(metaData["dist"], actions)
        entropy = actor.entropy(metaData["dist"])
        ratios = torch.exp(logprobs - old_logprobs)

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = self.entropy_scaler * entropy.mean()

        return actor_loss, entropy_loss

    def critic_loss(
        self, critic: nn.Module, states: torch.Tensor, returns: torch.Tensor
    ):
        values = critic(states)
        value_loss = self.mse_loss(values, returns)

        return value_loss

    def clip_grad_norm(self, grads, max_norm, eps=1e-6):
        # Compute total norm
        total_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
        clip_coef = max_norm / (total_norm + eps)

        if clip_coef < 1:
            grads = tuple(g * clip_coef for g in grads)

        return grads
