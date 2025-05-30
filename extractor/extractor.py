import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from policy.base import Base


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
        self.optimizer.step()

        self.lr_scheduler.step()

        ### Logging
        grad_dict, norm_dict = self.get_grad_weight_norm()
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
