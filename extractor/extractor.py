import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from policy.layers.base import Base


class DummyExtractor(Base):
    def __init__(self, indices: list):
        super(DummyExtractor, self).__init__()

        ### constants
        self.indices = indices
        self.name = "DummyExtractor"

    def to_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, states, deterministic: bool = False):
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).to(self.dtype).to(self.device)
        if len(states.shape) == 1 or len(states.shape) == 3:
            states = states.unsqueeze(0)
        if self.indices is not None:
            return states[:, self.indices], {}
        else:
            return states, {}

    def decode(self, features: torch.Tensor, actions: torch.Tensor):
        pass

    def learn(
        self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor
    ):
        pass


class Extractor(Base):
    def __init__(
        self,
        network: nn.Module,
        extractor_lr: float,
        epochs: int,
        batch_size: int,
        device: str = "cpu",
    ):
        super(Extractor, self).__init__()

        ### constants
        self.name = "EigenOption"
        self.epochs = epochs

        ### trainable parameters
        self.network = network
        self.optimizer = torch.optim.Adam(
            [{"params": self.network.parameters(), "lr": extractor_lr}],
        )
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)
        self.batch_size = batch_size

        #
        self.dummy = torch.tensor(1e-5)
        self.device = device
        self.to(self.device)

    def lr_lambda(self, step: int):
        return 1.0 - float(step) / float(self.epochs)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, states, deterministic: bool = False):
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).to(self.dtype).to(self.device)
        if len(states.shape) == 1 or len(states.shape) == 3:
            states = states.unsqueeze(0)

        features, infos = self.network(states, deterministic=deterministic)
        return features, infos

    def decode(self, features: torch.Tensor, actions: torch.Tensor):
        # match the data types
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).to(self.device)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)
        # match the dimensions
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)

        reconstructed_state = self.network.decode(features, actions)
        return reconstructed_state

    def learn(self, batch: dict):
        self.train()
        t0 = time.time()

        ### Pull data from the batch
        num_samples = batch["states"].shape[0]
        indices = torch.randperm(num_samples)[: self.batch_size]

        states = torch.from_numpy(batch["states"][indices]).to(self.device)
        actions = torch.from_numpy(batch["actions"][indices]).to(self.device)
        next_states = torch.from_numpy(batch["next_states"][indices]).to(self.device)

        timesteps = states.shape[0]

        ### Update
        features, infos = self(states)

        # if kl is too strong, it decoder is not converging
        encoder_loss = infos["loss"]

        reconstructed_states = self.decode(features, actions)
        decoder_loss = self.mse_loss(reconstructed_states, next_states)
        comparing_img = self.get_comparison_img(next_states[0], reconstructed_states[0])

        weight_loss = 0
        for param in self.network.parameters():
            if param.requires_grad:  # Only include parameters that require gradients
                weight_loss += torch.norm(param, p=2)  # L

        loss = encoder_loss + decoder_loss + 1e-6 * weight_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        grad_dict, norm_dict = self.get_grad_weight_norm()
        self.optimizer.step()

        self.lr_scheduler.step()

        ### Logging

        loss_dict = {
            f"{self.name}/loss": loss.item(),
            f"{self.name}/encoder_loss": encoder_loss.item(),
            f"{self.name}/decoder_loss": decoder_loss.item(),
            f"{self.name}/weight_loss": weight_loss.item(),
            f"{self.name}/extractor_lr": self.optimizer.param_groups[0]["lr"],
        }
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        del states, actions, next_states
        torch.cuda.empty_cache()

        t1 = time.time()
        self.eval()
        return loss_dict, timesteps, comparing_img, t1 - t0

    def get_comparison_img(self, x: torch.Tensor, y: torch.Tensor):
        if len(x.shape) == 1 and len(y.shape):
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        with torch.no_grad():
            comparing_img = torch.concatenate((x, y), dim=1)
            comparing_img = (comparing_img - comparing_img.min()) / (
                comparing_img.max() - comparing_img.min() + 1e-6
            )
        return comparing_img

    def get_grad_weight_norm(self):
        grad_dict = self.compute_gradient_norm(
            [self.network],
            ["extractor"],
            dir=f"{self.name}",
            device=self.device,
        )
        norm_dict = self.compute_weight_norm(
            [self.network],
            ["extractor"],
            dir=f"{self.name}",
            device=self.device,
        )
        return grad_dict, norm_dict


class ALLO(Extractor):
    def __init__(self, d, orth_lambda=1.0, graph_lambda=1.0, **kwargs):
        super(ALLO, self).__init__(**kwargs)
        self.name = "ALLO"
        self.d = d
        self.orth_lambda = orth_lambda
        self.graph_lambda = graph_lambda

        self.lr_duals = 1e-4
        self.lr_dual_velocities = 0.1
        self.lr_barrier_coeff = 1e-2
        self.use_barrier_for_duals = 0
        self.min_duals = 0.0
        self.max_duals = 100.0
        self.barrier_increase_rate = 0.1
        self.min_barrier_coefs = 0
        self.max_barrier_coefs = 10000

        self.permutation_array = np.arange(self.d)

        # Assumes self.barrier_initial_val is already defined as a float or torch scalar
        self.dual_variables = torch.zeros(
            (self.d, self.d), device=self.device, requires_grad=False
        )

        self.barrier_coeffs = torch.tril(
            2 * torch.ones((self.d, self.d), device=self.device, requires_grad=False),
            diagonal=0,
        )

        self.dual_velocities = torch.zeros(
            (self.d, self.d), device=self.device, requires_grad=False
        )

        self.errors = torch.zeros(
            (self.d, self.d), device=self.device, requires_grad=False
        )

        self.quadratic_errors = torch.zeros(
            (1, 1), device=self.device, requires_grad=False
        )

        self.nupdates = 0

    def learn(self, batch: dict):
        self.train()
        t0 = time.time()
        timesteps = self.batch_size

        # Sample discounted state pairs
        s1, s2 = self.sample_discounted_pairs_from_batch(
            batch, batch_size=self.batch_size, discount=0.9, device=self.device
        )

        phi1, _ = self(s1)  # [B, d]
        phi2, _ = self(s2)

        if self.nupdates % 5000 == 0:
            self.permutation_array = np.random.permutation(self.permutation_array)

        phi1 = phi1[:, self.permutation_array]
        phi2 = phi2[:, self.permutation_array]

        n, d = phi1.size()

        # Graph drawing loss (temporal smoothness)
        graph_loss_vec = ((phi1 - phi2) ** 2).mean(dim=0)  # [d]
        graph_loss = graph_loss_vec.sum()  # scalar

        # Orthogonality loss
        uncorrelated_s1 = self.sample_steps_from_batch(
            batch, batch_size=self.batch_size, device=self.device
        )
        uncorrelated_s2 = self.sample_steps_from_batch(
            batch, batch_size=self.batch_size, device=self.device
        )

        uncorrelated_phi1, _ = self(uncorrelated_s1)
        uncorrelated_phi2, _ = self(uncorrelated_s2)

        inner_product_matrix_1 = (
            uncorrelated_phi1.T @ uncorrelated_phi1.detach()
        ) / n  # [d,d]
        inner_product_matrix_2 = (
            uncorrelated_phi2.T @ uncorrelated_phi2.detach()
        ) / n  # [d,d]

        identity = torch.eye(d, device=self.device)
        error_matrix_1 = torch.tril(inner_product_matrix_1 - identity)
        error_matrix_2 = torch.tril(inner_product_matrix_2 - identity)
        error_matrix = 0.5 * (
            error_matrix_1 + error_matrix_2
        )  # This is your (⟨uj, Juk⟩ - δjk)

        # === CORRECTED LINE FOR QUADRATIC ERROR ===
        quadratic_error_matrix = error_matrix_1 * error_matrix_2
        quadratic_error_matrix = error_matrix**2  # elementwise square

        # Orthogonality dual loss
        dual_loss = (self.dual_variables.detach() * error_matrix).sum()

        # Barrier loss penalizing squared errors weighted by barrier coefficients
        # Use the correctly calculated quadratic_error_for_barrier
        barrier_loss = (
            self.barrier_coeffs[0, 0].detach() * quadratic_error_matrix
        ).sum()

        # Total loss
        loss = self.graph_lambda * graph_loss + self.orth_lambda * (
            dual_loss + barrier_loss
        )
        # print(barrier_loss)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10.0)
        grad_dict, norm_dict = self.get_grad_weight_norm()
        self.optimizer.step()

        self.lr_scheduler.step()

        # === Dual Update === #
        with torch.no_grad():
            barrier_coeff_val = self.barrier_coeffs[0, 0].item()
            scaled_barrier_coeff = 1 + self.use_barrier_for_duals * (
                barrier_coeff_val - 1
            )
            effective_lr = self.lr_duals * scaled_barrier_coeff

            updates = torch.tril(error_matrix)
            updated_duals = self.dual_variables + effective_lr * updates
            updated_duals = torch.clamp(
                updated_duals, min=self.min_duals, max=self.max_duals
            )

            delta = updated_duals - self.dual_variables
            norm_vel = torch.norm(self.dual_velocities)
            init_coeff = float(
                torch.isclose(
                    norm_vel,
                    torch.tensor(0.0, device=norm_vel.device),
                    rtol=1e-10,
                    atol=1e-13,
                )
            )
            update_rate = init_coeff + (1 - init_coeff) * self.lr_dual_velocities
            self.dual_velocities += update_rate * (delta - self.dual_velocities)
            self.dual_variables.copy_(torch.tril(updated_duals))

        # === Update barrier coefficients (matrix) ===
        with torch.no_grad():
            clipped_quad_error = torch.clamp(quadratic_error_matrix, min=0.0)
            error_update = clipped_quad_error.mean()  # scalar, like in JAX
            updated_barrier_coeffs = (
                self.barrier_coeffs + self.lr_barrier_coeff * error_update
            )
            self.barrier_coeffs.copy_(
                torch.clamp(
                    updated_barrier_coeffs,
                    min=self.min_barrier_coefs,
                    max=self.max_barrier_coefs,
                )
            )

        # Cleanup
        del phi1, phi2, uncorrelated_phi1, uncorrelated_phi2
        torch.cuda.empty_cache()

        t1 = time.time()
        self.eval()
        loss_dict = {
            f"{self.name}/loss": loss.item(),
            f"{self.name}/graph_loss": graph_loss.item(),
            f"{self.name}/dual_loss": dual_loss.item(),
            f"{self.name}/barrier_loss": barrier_loss.item(),
            f"{self.name}/extractor_lr": self.optimizer.param_groups[0]["lr"],
            f"{self.name}/duals": torch.linalg.norm(self.dual_variables).item(),
            f"{self.name}/b": self.barrier_coeffs[0, 0].item(),
        }
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        self.nupdates += 1

        return loss_dict, timesteps, None, t1 - t0

    def discounted_sampling(
        self, ranges: torch.Tensor, discount: float
    ) -> torch.Tensor:
        """Inverse transform sampling from truncated geometric distribution."""
        seeds = torch.rand_like(ranges, dtype=torch.float)
        if discount == 0.0:
            return torch.zeros_like(ranges, dtype=torch.long)
        elif discount == 1.0:
            return (seeds * ranges).floor().long()
        else:
            discount_pow = discount**ranges
            samples = torch.log1p(-(1 - discount_pow) * seeds) / torch.log(
                torch.tensor(discount)
            )
            return samples.floor().long()

    def sample_discounted_pairs_from_batch(
        self, batch: dict, batch_size: int, discount: float, device
    ):
        states_np = batch["states"]
        terminals_np = batch["terminals"]

        # Convert to CPU tensors
        states = torch.from_numpy(states_np).float()
        terminals = torch.from_numpy(terminals_np).float()

        # Identify episode start/end indices
        done_idxs = (terminals == 1).nonzero(as_tuple=True)[0]
        start_idxs = torch.cat([torch.tensor([0]), done_idxs[:-1] + 1])
        end_idxs = done_idxs + 1

        episodes = [(int(start), int(end)) for start, end in zip(start_idxs, end_idxs)]
        valid_episodes = [ep for ep in episodes if ep[1] - ep[0] >= 2]

        assert len(valid_episodes) > 0, "No valid episodes found"

        sampled = torch.randint(0, len(valid_episodes), (batch_size,))
        s1_list, s2_list = [], []

        for i in sampled:
            start, end = valid_episodes[i]
            ep_len = end - start

            t1_local = torch.randint(0, ep_len - 1, (1,)).item()
            max_delta = ep_len - t1_local - 1
            delta = (
                self.discounted_sampling(torch.tensor([max_delta]), discount=discount)[
                    0
                ].item()
                + 1
            )

            t1 = start + t1_local
            t2 = t1 + delta

            s1_list.append(states[t1])
            s2_list.append(states[t2])

        s1 = torch.stack(s1_list).to(device)
        s2 = torch.stack(s2_list).to(device)
        return s1, s2

    def sample_steps_from_batch(
        self, batch: dict, batch_size: int, device: torch.device
    ):
        states_np = batch["states"]  # shape: [total_steps, state_dim]
        terminals_np = batch["terminals"]  # shape: [total_steps]

        # Identify episode boundaries (start and end indices)
        done_idxs = np.where(terminals_np == 1)[0]
        start_idxs = np.concatenate(([0], done_idxs[:-1] + 1))
        end_idxs = done_idxs + 1

        episodes = list(zip(start_idxs, end_idxs))
        valid_episodes = [ep for ep in episodes if ep[1] - ep[0] > 0]

        assert len(valid_episodes) > 0, "No valid episodes found in batch."

        # Sample episode indices with replacement (batch_size episodes)
        episode_indices = np.random.choice(
            len(valid_episodes), size=batch_size, replace=True
        )

        # Sample steps uniformly within each sampled episode
        sampled_states = []
        for epi_idx in episode_indices:
            start, end = valid_episodes[epi_idx]
            step_idx = np.random.randint(
                start, end
            )  # uniform sampling of step within episode
            sampled_states.append(states_np[step_idx])

        # Convert to torch tensor and send to device
        sampled_states_tensor = torch.tensor(
            np.array(sampled_states), dtype=torch.float32, device=device
        )

        return sampled_states_tensor
