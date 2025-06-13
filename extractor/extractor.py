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
        minibatch_size: int,
        device: str = "cpu",
    ):
        super(Extractor, self).__init__()

        ### constants
        self.name = "Extractor"
        self.epochs = epochs

        ### trainable parameters
        self.network = network
        self.optimizer = torch.optim.Adam(
            [{"params": self.network.parameters(), "lr": extractor_lr}],
        )
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

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

    def learn(
        self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor
    ):
        self.train()
        t0 = time.time()

        ### Pull data from the batch
        timesteps = states.shape[0]
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)

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
        self.d = d
        self.orth_lambda = orth_lambda
        self.graph_lambda = graph_lambda

        self.lr_duals = 1e-4
        self.lr_dual_velocities = 0.1
        self.lr_barrier_coeff = 1.0
        self.use_barrier_for_duals = 0
        self.min_duals = 0.0
        self.max_duals = 100.0
        self.barrier_increase_rate = 0.1
        self.min_barrier_coefs = 0
        self.max_barrier_coefs = 10000

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

    def learn(
        self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor
    ):
        self.train()
        t0 = time.time()

        timesteps = states.shape[0]
        states = torch.from_numpy(states).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)

        phi, _ = self(states)  # [B, d]
        next_phi, _ = self(next_states)

        n, d = phi.size()

        # Graph drawing loss (temporal smoothness)
        graph_loss_vec = ((phi - next_phi) ** 2).mean(dim=0)  # [d]
        graph_loss = graph_loss_vec.sum()  # scalar

        # Orthogonality loss
        permuted_indices = torch.randperm(n)
        next_phi_permuted = next_phi[permuted_indices]
        phi_permuted = phi[permuted_indices]

        inner_product_matrix_1 = (phi_permuted.T @ phi_permuted) / n  # [d,d]
        inner_product_matrix_2 = (next_phi_permuted.T @ next_phi_permuted) / n  # [d,d]

        identity = torch.eye(d, device=phi.device)
        error_matrix_1 = torch.tril(inner_product_matrix_1 - identity)
        error_matrix_2 = torch.tril(inner_product_matrix_2 - identity)
        error_matrix = 0.5 * (error_matrix_1 + error_matrix_2)
        quadratic_error_matrix = error_matrix_1 * error_matrix_2

        # Orthogonality dual loss
        dual_loss = (self.dual_variables.detach() * error_matrix).sum()

        # Barrier loss penalizing squared errors weighted by barrier coefficients
        barrier_loss = (
            self.barrier_coeffs[0, 0].detach() * quadratic_error_matrix
        ).sum()

        # Total loss
        loss = self.graph_lambda * graph_loss + self.orth_lambda * (
            dual_loss + barrier_loss
        )

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10.0)
        grad_dict, norm_dict = self.get_grad_weight_norm()
        self.optimizer.step()

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
            updated_duals = torch.tril(updated_duals)

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
            self.dual_variables.copy_(updated_duals)

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
        del states, next_states, phi, next_phi
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

        return loss_dict, timesteps, None, t1 - t0
