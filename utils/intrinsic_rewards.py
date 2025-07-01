import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from policy.uniform_random import UniformRandom
from utils.functions import call_env
from utils.sampler import OnlineSampler


class IntrinsicRewardFunctions(nn.Module):
    def __init__(self, logger, writer, args):
        super(IntrinsicRewardFunctions, self).__init__()

        # === Parameter saving === #
        self.episode_len_for_sampling = 250_000
        self.num_trials = 10

        self.extractor_env = call_env(deepcopy(args), self.episode_len_for_sampling)
        self.logger = logger
        self.writer = writer
        self.args = args

        self.current_timesteps = 0
        self.loss_dict = {}

        # === MAKE ENV === #
        if self.args.intrinsic_reward_mode == "allo":
            print("NOTE: ALLO is only implemented for discrete environments.")
            self.num_rewards = self.args.num_options
            self.extractor_mode = "allo"
            self.define_extractor()
            self.define_eigenvectors()
            # self.define_intrinsic_reward_normalizer()

            self.sources = ["allo" for _ in range(self.num_rewards)]
        elif self.args.intrinsic_reward_mode == "eigendecomp":
            print(
                "NOTE: Eigendecomposition is only implemented for continuous environments."
            )
            self.num_rewards = self.args.num_options
            self.extractor_mode = "eigendecomp"
            self.define_extractor()
            self.define_eigenvectors()
            # self.define_intrinsic_reward_normalizer()

            self.sources = ["eigendecomp" for _ in range(self.num_rewards)]
        elif self.args.intrinsic_reward_mode == "drnd":
            self.num_rewards = 1
            self.extractor_mode = "drnd"
            self.define_drnd_policy()

            self.sources = ["drnd"]
        elif self.args.intrinsic_reward_mode == "allo-drnd":
            # allo + drnd
            self.num_rewards = self.args.num_options + 1
            self.extractor_mode = "allo"

            self.define_extractor()
            self.define_eigenvectors()
            # self.define_intrinsic_reward_normalizer()
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

            # === REMOVE EIGENVECTORS & NORMALIZER === #
            if source == "allo":
                del self.eigenvectors[i]
                del self.sources[i]
                if hasattr(self, "reward_rms"):
                    del self.reward_rms[i]
            elif source == "drnd":
                del self.sources[i]

            self.num_rewards = len(self.sources)
        else:
            print(
                f"Nothing to prune. Current number of intrinsic rewards: {self.num_rewards}"
            )

    def forward(self, states: torch.Tensor, next_states: torch.Tensor, i: int):
        if self.sources[i] == "allo":
            states = states[:, self.args.positional_indices]
            next_states = next_states[:, self.args.positional_indices]

            with torch.no_grad():
                feature, _ = self.extractor(states)
                next_feature, _ = self.extractor(next_states)
                difference = next_feature - feature

                eigenvector_idx, eigenvector_sign = self.eigenvectors[i]
                intrinsic_rewards = eigenvector_sign * difference[
                    :, eigenvector_idx
                ].unsqueeze(-1)

        elif self.sources[i] == "eigendecomp":
            # get features
            difference = (
                (next_states - states).cpu().numpy()[:, self.args.positional_indices]
            )

            # Calculate the intrinsic reward using the eigenvector
            intrinsic_rewards = np.matmul(
                difference, self.eigenvectors[i][:, np.newaxis]
            )
            intrinsic_rewards = torch.from_numpy(intrinsic_rewards).to(
                states.device, dtype=states.dtype
            )
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

    def define_extractor(self):
        from extractor.base.mlp import NeuralNet
        from extractor.extractor import ALLO
        from policy.uniform_random import UniformRandom
        from trainer.extractor_trainer import ExtractorTrainer

        if not os.path.exists("model"):
            os.makedirs("model")

        env_name, version = self.args.env_name.split("-")
        if env_name in ["fourrooms", "maze"]:
            model_path = f"model/{self.args.env_name}-{self.extractor_mode}-{self.args.num_random_agents}-extractor.pth"
            # === CREATE FEATURE EXTRACTOR === #
            feature_network = NeuralNet(
                state_dim=len(
                    self.args.positional_indices
                ),  # discrete position is always 2d
                feature_dim=(8 // 2 + 1),
                encoder_fc_dim=[512, 512, 512, 512],
                activation=nn.LeakyReLU(),
            )

            # === DEFINE LEARNING METHOD FOR EXTRACTOR === #
            extractor = ALLO(
                network=feature_network,
                positional_indices=self.args.positional_indices,
                extractor_lr=self.args.extractor_lr,
                epochs=self.args.extractor_epochs,
                batch_size=4096,
                device=self.args.device,
            )

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
                    episode_len=self.episode_len_for_sampling,
                    batch_size=self.num_trials * self.episode_len_for_sampling,
                    verbose=False,
                )
                trainer = ExtractorTrainer(
                    env=self.extractor_env,
                    random_policy=uniform_random_policy,
                    extractor=extractor,
                    sampler=sampler,
                    logger=self.logger,
                    writer=self.writer,
                    epochs=self.args.extractor_epochs,
                )
                final_timesteps = trainer.train()
                self.current_timesteps += final_timesteps

                torch.save(extractor.state_dict(), model_path)
            else:
                extractor.load_state_dict(
                    torch.load(model_path, map_location=self.args.device)
                )
                extractor.to(self.args.device)
        else:
            # No extractor is needed for continuous environments
            extractor = None

        self.extractor = extractor

    def define_eigenvectors(self):
        # === Define eigenvectors === #
        env_name, version = self.args.env_name.split("-")
        if env_name in ["fourrooms", "maze"]:
            # ALLO does not have explicit eigenvectors.
            # Instead, we make list that contains the eigenvector index and sign
            eigenvectors = [
                (n // 2 + 1, 2 * (n % 2) - 1) for n in range(self.args.num_options)
            ]
        else:
            sampler = OnlineSampler(
                state_dim=self.args.state_dim,
                action_dim=self.args.action_dim,
                episode_len=self.episode_len_for_sampling,
                batch_size=self.num_trials * self.episode_len_for_sampling,
                verbose=False,
            )

            uniform_random_policy = UniformRandom(
                state_dim=self.args.state_dim,
                action_dim=self.args.action_dim,
                is_discrete=self.args.is_discrete,
                device=self.args.device,
            )
            batch, _ = sampler.collect_samples(
                env=self.extractor_env,
                policy=uniform_random_policy,
                seed=self.args.seed,
            )
            states = torch.from_numpy(batch["states"]).to(self.args.device)[
                :, self.args.positional_indices
            ]

            # Covariance-based PCA
            cov = torch.cov(states.T)
            eigval, eigvec = torch.linalg.eigh(cov)
            sorted_indices = torch.argsort(eigval, descending=True)
            eigval = eigval[sorted_indices]
            eigvec = eigvec[:, sorted_indices]
            eigvec_row = eigvec.T.real
            eig_vec_row = eigvec_row[: int(self.args.num_options / 2)]
            eigenvectors = torch.cat([eig_vec_row, -eig_vec_row], dim=0)
            eigenvectors = eigenvectors.cpu().numpy()

        heatmaps = self.extractor_env.get_rewards_heatmap(self.extractor, eigenvectors)

        self.eigenvectors = eigenvectors
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

        drnd_model = DRNDModel(
            input_dim=len(self.args.positional_indices),
            output_dim=self.args.feature_dim,
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
            positional_indices=self.args.positional_indices,
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
