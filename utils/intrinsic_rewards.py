import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from utils.sampler import OnlineSampler


class IntrinsicRewardFunctions(nn.Module):
    def __init__(self, env, logger, writer, args):
        super(IntrinsicRewardFunctions, self).__init__()

        from utils.functions import call_env

        # === Parameter saving === #
        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args

        self.current_timesteps = 0
        self.loss_dict = {}

        # === MAKE ENV === #
        self.env_name, version = args.env_name.split("-")

        if self.args.intrinsic_reward_mode == "eigenpurpose":
            self.num_rewards = self.args.num_options
            self.extractor_mode = "eigenpurpose"
            self.define_extractor()
            self.define_eigenvectors()
            self.define_intrinsic_reward_normalizer()

            self.sources = ["eigenpurpose" for _ in range(self.num_rewards)]
        elif self.args.intrinsic_reward_mode == "allo":
            self.num_rewards = self.args.num_options
            self.extractor_mode = "allo"
            self.define_extractor()
            self.define_eigenvectors()
            self.define_intrinsic_reward_normalizer()

            self.sources = ["allo" for _ in range(self.num_rewards)]
        elif self.args.intrinsic_reward_mode == "drnd":
            self.num_rewards = 1
            self.extractor_mode = None
            self.define_drnd_policy()

            self.sources = ["drnd"]
        elif self.args.intrinsic_reward_mode == "eig-drnd":
            # eigenpurpose + drnd
            self.num_rewards = self.args.num_options + 1
            self.extractor_mode = "eigenpurpose"

            self.define_extractor()
            self.define_eigenvectors()
            self.define_intrinsic_reward_normalizer()
            self.define_drnd_policy()

            self.sources = ["eigenpurpose" for _ in range(self.args.num_options)]
            self.sources.append("drnd")
        elif self.args.intrinsic_reward_mode == "allo-drnd":
            # allo + drnd
            self.num_rewards = self.args.num_options + 1
            self.extractor_mode = "allo"

            self.define_extractor()
            self.define_eigenvectors()
            self.define_intrinsic_reward_normalizer()
            self.define_drnd_policy()

            self.sources = ["allo" for _ in range(self.args.num_options)]
            self.sources.append("drnd")
        else:
            raise NotImplementedError(
                f"The intrinsic reward mode {self.args.intrinsic_reward_mode} not implemented or unknown."
            )

    def prune(self, i: int):
        if self.num_rewards > 1:
            source = self.sources[i]

            # === REMOVE EIGENVECTORS === #
            if self.extractor_mode == "allo":
                del self.eigenvectors[i]
            elif self.extractor_mode == "allo":
                self.eigenvectors = torch.cat(
                    (
                        self.eigenvectors[:i],
                        self.eigenvectors[i + 1 :],
                    ),
                    dim=0,
                )
            else:
                raise ValueError(f"Unknown extractor mode.")

            # === REMOVE NORMALIZER === #
            if source in ("eigenpurpose", "allo"):
                del self.reward_rms[i]
                del self.sources[i]
            elif source == "drnd":
                del self.sources[i]

            self.num_rewards = len(self.sources)
        else:
            print(
                f"Nothing to prune. Current number of intrinsic rewards: {self.num_rewards}"
            )

    def forward(self, states: torch.Tensor, next_states: torch.Tensor, i: int):
        if self.sources[i] == "eigenpurpose":
            # get features
            with torch.no_grad():
                feature, _ = self.extractor(states)
                next_feature, _ = self.extractor(next_states)
                difference = next_feature - feature

            # Calculate the intrinsic reward using the eigenvector
            intrinsic_rewards = torch.matmul(
                difference, self.eigenvectors[i].unsqueeze(-1)
            )
            # .to(self.args.device)
        elif self.sources[i] == "allo":
            # if self.env_name in ("fourrooms", "ninerooms", "maze"):
            # states (B, width, height, channel)
            # to states (B, 2) where 2 is agent position
            # agent_idx = 10
            # states = self.extract_agent_positions(states, agent_idx=10)
            # next_states = self.extract_agent_positions(next_states, agent_idx=10)

            with torch.no_grad():
                feature, _ = self.extractor(states)
                next_feature, _ = self.extractor(next_states)
                difference = next_feature - feature

                eigenvector_idx, eigenvector_sign = self.eigenvectors[i]
                intrinsic_rewards = eigenvector_sign * difference[
                    :, eigenvector_idx
                ].unsqueeze(-1)

        elif self.sources[i] == "drnd":
            with torch.no_grad():
                intrinsic_rewards = self.drnd_policy.intrinsic_reward(next_states)

        # === INTRINSIC REWARD NORMALIZATION === #
        if hasattr(self, "reward_rms") and self.sources[i] != "drnd":
            # drnd has its own normalizer in itself
            self.reward_rms[i].update(intrinsic_rewards.cpu().numpy())
            var_tensor = torch.as_tensor(
                self.reward_rms[i].var,
                device=intrinsic_rewards.device,
                dtype=intrinsic_rewards.dtype,
            )
            intrinsic_rewards = intrinsic_rewards / (torch.sqrt(var_tensor) + 1e-8)

        return intrinsic_rewards, self.sources[i]

    def learn(
        self, states: torch.Tensor, next_states: torch.Tensor, i: int, keyword: str
    ):
        if keyword == "drnd":
            batch_size = states.shape[0]
            iteration = 10
            losses = []
            perm = torch.randperm(batch_size)
            mb_size = batch_size // iteration
            for i in range(iteration):
                indices = perm[i * mb_size : (i + 1) * mb_size]
                drnd_loss = self.drnd_policy.drnd_loss(next_states[indices])

                self.drnd_policy.optimizer.zero_grad()
                drnd_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.drnd_policy.parameters(), max_norm=0.5
                )
                self.drnd_policy.optimizer.step()
                losses.append(drnd_loss.item())

            self.loss_dict[f"IntrinsicRewardModuleLoss/drnd_loss"] = np.array(
                losses
            ).mean()
        else:
            pass

    def extract_agent_positions(
        self, states: torch.Tensor, agent_idx: int
    ) -> torch.Tensor:
        """
        Vectorized extraction of agent positions from (B, W, H, C) states.
        Assumes exactly one agent per sample.

        Args:
            states: Tensor of shape (B, W * H * C)
            agent_idx: Integer value representing the agent

        Returns:
            Tensor of shape (B, 2) with (x, y) positions
        """
        # (B, W * H * C) → (B, W, H, C): True where any channel equals agent_idx
        states = states.reshape(
            states.shape[0], self.extractor_env.width, self.extractor_env.height, -1
        )
        agent_mask = (states == agent_idx).any(dim=-1)  # shape: (B, W, H)

        # Flatten spatial dims
        B, W, H = agent_mask.shape
        flat_mask = agent_mask.view(B, -1)  # (B, W*H)

        # Get index of agent in flattened grid
        flat_indices = flat_mask.float().argmax(dim=-1)  # (B,)

        # Convert flat indices to 2D coordinates (x, y)
        y = flat_indices % H
        x = flat_indices // H

        states = torch.stack([x, y], dim=-1).float()  # (B, 2)
        return states

    def define_extractor(self):
        from policy.uniform_random import UniformRandom
        from trainer.extractor_trainer import ExtractorTrainer
        from utils.rl import get_extractor

        if not os.path.exists("model"):
            os.makedirs("model")

        model_path = f"model/{self.args.env_name}-{self.extractor_mode}-{self.args.num_random_agents}-extractor.pth"
        extractor = get_extractor(self.args)

        if not os.path.exists(model_path):
            uniform_random_policy = UniformRandom(
                state_dim=self.args.state_dim,
                action_dim=self.args.action_dim,
                is_discrete=self.args.is_discrete,
                device=self.args.device,
            )
            sampler = OnlineSampler(
                state_dim=self.args.state_dim,
                action_dim=self.args.action_dim,
                episode_len=self.args.episode_len,
                batch_size=500 * self.args.episode_len,
                verbose=False,
            )
            trainer = ExtractorTrainer(
                env=self.env,
                random_policy=uniform_random_policy,
                extractor=extractor,
                sampler=sampler,
                logger=self.logger,
                writer=self.writer,
                epochs=self.args.extractor_epochs,
            )
            if extractor.name != "DummyExtractor":
                final_timesteps = trainer.train()
                self.current_timesteps += final_timesteps

            torch.save(extractor.state_dict(), model_path)
        else:
            extractor.load_state_dict(
                torch.load(model_path, map_location=self.args.device)
            )
            extractor.to(self.args.device)

        self.extractor = extractor

    def define_eigenvectors(self):
        from utils.rl import get_vector

        # === Define eigenvectors === #
        eigenvectors, heatmaps = get_vector(self.env, self.extractor, self.args)
        if isinstance(eigenvectors, np.ndarray):
            self.eigenvectors = torch.from_numpy(eigenvectors).to(self.args.device)
        else:
            # this is the case for ALLO
            # where eigenvectors are returned as index and sign tuple.
            self.eigenvectors = eigenvectors
            pass

        self.logger.write_images(
            step=self.current_timesteps, images=heatmaps, logdir="Image/Heatmaps"
        )

    def define_intrinsic_reward_normalizer(self):
        from utils.wrapper import RunningMeanStd

        self.reward_rms = []
        # DRND method has its own rms its own class
        for _ in range(self.args.num_options):
            self.reward_rms.append(RunningMeanStd(shape=(1,)))

    def define_drnd_policy(self):
        from policy.drndppo import DRNDPPO_Learner
        from policy.layers.drnd_networks import DRNDModel
        from policy.layers.ppo_networks import PPO_Actor, PPO_Critic

        actor = PPO_Actor(
            input_dim=self.args.state_dim,
            hidden_dim=self.args.actor_fc_dim,
            action_dim=self.args.action_dim,
            is_discrete=self.args.is_discrete,
            device=self.args.device,
        )
        critic = PPO_Critic(self.args.state_dim, hidden_dim=self.args.critic_fc_dim)

        feature_dim = (
            self.args.feature_dim if self.args.feature_dim else self.args.state_dim
        )
        drnd_model = DRNDModel(
            input_dim=self.args.state_dim,
            output_dim=feature_dim,
            num=10,
            device=self.args.device,
        )
        drnd_critic = PPO_Critic(
            self.args.state_dim, hidden_dim=self.args.critic_fc_dim
        )

        self.drnd_policy = DRNDPPO_Learner(
            actor=actor,
            critic=critic,
            drnd_model=drnd_model,
            drnd_critic=drnd_critic,
            actor_lr=self.args.actor_lr,
            critic_lr=self.args.critic_lr,
            drnd_lr=5e-5,
            num_minibatch=self.args.num_minibatch,
            minibatch_size=self.args.minibatch_size,
            eps_clip=self.args.eps_clip,
            entropy_scaler=self.args.entropy_scaler,
            target_kl=self.args.target_kl,
            gamma=self.args.gamma,
            gae=self.args.gae,
            K=self.args.K_epochs,
            device=self.args.device,
        )
