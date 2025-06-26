import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal, Normal

from policy.layers.base import Base
from policy.layers.building_blocks import MLP


class SAC_Actor(Base):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        action_dim: int,
        action_scale: tuple,
        activation: nn.Module = nn.ReLU(),
        device=torch.device("cpu"),
    ):
        super().__init__(device=device)  # modify based on Base's signature

        self.state_dim = np.prod(input_dim)
        self.hidden_dim = hidden_dim
        self.action_dim = np.prod(action_dim)

        self.low_action_scale = torch.from_numpy(action_scale[0]).to(self.device)
        self.high_action_scale = torch.from_numpy(action_scale[1]).to(self.device)

        self.action_max = torch.max(
            torch.stack(
                [
                    torch.abs(self.low_action_scale).max(),
                    torch.abs(self.high_action_scale).max(),
                ]
            )
        )

        self.is_discrete = False

        self.model = MLP(
            self.state_dim,
            hidden_dim,
            self.action_dim,
            activation=activation,
            initialization="actor",
        )

        self.device = device
        self._dummy = torch.tensor(1e-8)
        self.to(self.device).to(self.dtype)

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool,
    ):
        logits = self.model(state)
        action = self.action_max * F.tanh(logits)

        if not deterministic:
            # Add small exploration noise for action selection (not training!)
            mean = torch.zeros(action.size()).to(self.device)
            var = 0.1 * torch.ones(action.size()).to(self.device)
            noise = torch.normal(mean, var)
            action += noise

        action = torch.min(action, self.high_action_scale)
        action = torch.max(action, self.low_action_scale)

        return action, {
            "dist": self._dummy,
            "probs": self._dummy,
            "logprobs": self._dummy,
            "entropy": self._dummy,
        }


class SAC_Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: list,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        self.input_dim = np.prod(state_dim) + np.prod(action_dim)

        self.model = MLP(
            self.input_dim,
            hidden_dim,
            1,
            activation=activation,
            initialization="critic",
        )

    def forward(self, x: torch.Tensor):
        value = self.model(x)
        return value
