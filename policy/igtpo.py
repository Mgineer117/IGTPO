import os
import pickle
import time
from copy import deepcopy
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import inverse, matmul, transpose
from torch.autograd import grad

from policy.base import Base
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from utils.rl import estimate_advantages


def flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model, flat_params):
    pointer = 0
    for p in model.parameters():
        num_param = p.numel()
        p.data.copy_(flat_params[pointer : pointer + num_param].view_as(p))
        pointer += num_param


def compute_kl(old_actor, new_policy, obs):
    _, old_infos = old_actor(obs)
    _, new_infos = new_policy(obs)

    kl = torch.distributions.kl_divergence(old_infos["dist"], new_infos["dist"])
    return kl.mean()


def hessian_vector_product(kl_fn, model, damping, v):
    kl = kl_fn()
    grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
    flat_grads = torch.cat([g.view(-1) for g in grads])
    g_v = (flat_grads * v).sum()
    hv = torch.autograd.grad(g_v, model.parameters())
    flat_hv = torch.cat([h.contiguous().view(-1) for h in hv])
    return flat_hv + damping * v


def conjugate_gradients(Av_func, b, nsteps=10, tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(nsteps):
        Avp = Av_func(p)
        alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        if new_rdotr < tol:
            break
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr
    return x


class IGTPO_Learner(Base):
    def __init__(
        self,
        actor: PPO_Actor,
        nupdates: int,
        igtpo_actor_lr: float = 3e-4,
        batch_size: int = 256,
        eps_clip: float = 0.2,
        entropy_scaler: float = 1e-3,
        l2_reg: float = 1e-8,
        target_kl: float = 0.03,
        gamma: float = 0.99,
        gae: float = 0.9,
        K: int = 5,
        device: str = "cpu",
    ):
        super(IGTPO_Learner, self).__init__()

        # constants
        self.name = "IGTPO"
        self.device = device

        self.state_dim = actor.state_dim
        self.action_dim = actor.action_dim

        self.nupdates = nupdates
        self.batch_size = batch_size
        self.entropy_scaler = entropy_scaler
        self.gamma = gamma
        self.gae = gae
        self.K = K
        self.l2_reg = l2_reg
        self.eps_clip = eps_clip
        self.init_target_kl = target_kl
        self.target_kl = target_kl

        # trainable networks
        self.actor = actor
        self.igtpo_actor_lr = igtpo_actor_lr

        #
        self.steps = 0
        self.to(self.dtype).to(self.device)

    def lr_scheduler(self, fraction: float):
        self.target_kl = self.init_target_kl * (1 - fraction)
        self.steps += 1

    def forward(self, state: np.ndarray, deterministic: bool = False):
        state = self.preprocess_state(state)
        a, metaData = self.actor(state, deterministic=deterministic)

        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
            "dist": metaData["dist"],
        }

    def trpo_learn(
        self,
        states: np.ndarray,
        grads: torch.Tensor,
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

        # check if step_dir and lm contain nan or inf values
        if torch.isnan(step_dir).any() or torch.isinf(step_dir).any():
            print("Warning: step_dir contains NaN or Inf values.")
            return 0, False
        if torch.isnan(lm).any() or torch.isinf(lm).any():
            print("Warning: lm contains NaN or Inf values.")
            return 0, False

        full_step = step_dir / (lm + 1e-8)

        # check if full_step contains nan or inf values
        if torch.isnan(full_step).any() or torch.isinf(full_step).any():
            print("Warning: full_step contains NaN or Inf values.")
            return 0, False

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

    def learn(self, critic: nn.Module, batch: dict, prefix: str):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # 0. Prepare ingredients
        states = self.preprocess_state(batch["states"])
        actions = self.preprocess_state(batch["actions"])
        rewards = self.preprocess_state(batch["rewards"])
        terminals = self.preprocess_state(batch["terminals"])
        old_logprobs = self.preprocess_state(batch["logprobs"])

        # 1. Compute advantages and returns
        with torch.no_grad():
            values = critic(states)
            advantages, _ = estimate_advantages(
                rewards,
                terminals,
                values,
                gamma=self.gamma,
                gae=self.gae,
            )

        normalized_advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        # 3. actor Loss
        actor_loss, entropy_loss, clip_fraction, kl_div = self.actor_loss(
            states, actions, old_logprobs, normalized_advantages
        )

        # 4. Total loss
        loss = actor_loss - entropy_loss

        # 5. Compute gradients (example)
        gradients = torch.autograd.grad(loss, self.parameters(), create_graph=True)
        gradients = self.clip_grad_norm(gradients, max_norm=1.0)

        # 6. Manual SGD update (structured, not flat)
        actor_clone = deepcopy(self.actor)
        with torch.no_grad():
            for p, g in zip(actor_clone.parameters(), gradients):
                p -= self.igtpo_actor_lr * g

        # 7. create a new policy
        new_policy = deepcopy(self)
        new_policy.actor = actor_clone

        # 8. Logging
        actor_grad_norm = torch.sqrt(
            sum(g.pow(2).sum() for g in gradients if g is not None)
        )

        loss_dict = {
            f"{self.name}-{prefix}/loss/loss": loss.item(),
            f"{self.name}-{prefix}/loss/actor_loss": actor_loss.item(),
            f"{self.name}-{prefix}/loss/entropy_loss": entropy_loss.item(),
            f"{self.name}-{prefix}/grad/actor": actor_grad_norm.item(),
            f"{self.name}-{prefix}/analytics/avg_rewards": torch.mean(rewards).item(),
        }
        norm_dict = self.compute_weight_norm(
            [self.actor],
            ["actor"],
            dir=f"{self.name}-{prefix}",
            device=self.device,
        )
        loss_dict.update(norm_dict)

        self.eval()

        timesteps = self.batch_size
        update_time = time.time() - t0

        return (
            loss_dict,
            timesteps,
            update_time,
            new_policy,
            gradients,
            values.mean().cpu().numpy(),
        )

    def actor_loss(
        self,
        mb_states: torch.Tensor,
        mb_actions: torch.Tensor,
        mb_old_logprobs: torch.Tensor,
        mb_advantages: torch.Tensor,
    ):
        _, metaData = self.actor(mb_states)
        logprobs = self.actor.log_prob(metaData["dist"], mb_actions)
        entropy = self.actor.entropy(metaData["dist"])
        ratios = torch.exp(logprobs - mb_old_logprobs)

        surr1 = ratios * mb_advantages
        surr2 = (
            torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages
        )

        actor_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = self.entropy_scaler * entropy.mean()

        # Compute clip fraction (for logging)
        clip_fraction = torch.mean(
            (torch.abs(ratios - 1) > self.eps_clip).float()
        ).item()

        # Check if KL divergence exceeds target KL for early stopping
        kl_div = torch.mean(mb_old_logprobs - logprobs)

        return actor_loss, entropy_loss, clip_fraction, kl_div

    def clip_grad_norm(self, grads, max_norm, eps=1e-6):
        # Compute total norm
        total_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
        clip_coef = max_norm / (total_norm + eps)

        if clip_coef < 1:
            grads = tuple(g * clip_coef for g in grads)

        return grads
