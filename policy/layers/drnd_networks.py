import numpy as np
import torch
import torch.nn as nn

from policy.layers.base import Base
from policy.layers.building_blocks import MLP


class DRNDModel(Base):
    """
    DRND Model for predicting the next state and target features.
    Adapted from: https://github.com/yk7333/DRND/blob/main/online/model.py
    """

    def __init__(
        self, input_dim: int, output_dim: int, num: int, device=torch.device("cpu")
    ):
        super().__init__(device=device)

        self.input_dim = np.prod(input_dim)
        self.output_dim = np.prod(output_dim)
        self.num_target = num
        self.device = device

        self.predictor = MLP(
            self.input_dim,
            [512, 512, 512],
            self.output_dim,
            activation=nn.ReLU(),
            initialization="critic",
        )

        self.target = nn.ModuleList(
            [
                MLP(
                    self.input_dim,
                    [128, 128],
                    self.output_dim,
                    activation=nn.Tanh(),
                    initialization="critic",
                )
                for _ in range(num)
            ]
        )

        # detach the gradients of target networks
        for t_net in self.target:
            for param in t_net.parameters():
                param.requires_grad = False

        self.to(device)

    def forward(self, next_obs: torch.Tensor):
        # Predict the next state features and target features
        predict_feature = self.predictor(next_obs)

        # Collect target features from each target network
        target_features = []
        for t_net in self.target:
            target_features.append(t_net(next_obs))
        target_feature = torch.stack(target_features, dim=0)

        return predict_feature, target_feature
