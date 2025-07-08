import glob
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
        self.episode_len_for_sampling = args.episode_len  # 200_000
        self.num_trials = 2_000

        self.extractor_env = call_env(
            deepcopy(args), self.episode_len_for_sampling, random_spawn=True
        )
        self.logger = logger
        self.writer = writer
        self.args = args

        self.current_timesteps = 0
        self.loss_dict = {}

        # === MAKE ENV === #
        if self.args.intrinsic_reward_mode == "allo":
            self.num_rewards = self.args.num_options
            self.extractor_mode = "allo"
            self.define_extractor()
            self.define_eigenvectors()
            # self.define_intrinsic_reward_normalizer()

            self.sources = ["allo" for _ in range(self.num_rewards)]
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
        if not os.path.exists(f"model/{self.args.env_name}"):
            os.makedirs(f"model/{self.args.env_name}")

        # env_name, version = self.args.env_name.split("-")
        # Directory path based on env_name

        # === CREATE FEATURE EXTRACTOR === #
        feature_network = NeuralNet(
            state_dim=len(
                self.args.positional_indices
            ),  # discrete position is always 2d
            feature_dim=self.args.feature_dim,  # (8 // 2 + 1),
            encoder_fc_dim=[512, 512, 512, 512],
            activation=nn.LeakyReLU(),
        )

        # === DEFINE LEARNING METHOD FOR EXTRACTOR === #
        extractor = ALLO(
            network=feature_network,
            positional_indices=self.args.positional_indices,
            extractor_lr=self.args.extractor_lr,
            epochs=self.args.extractor_epochs,
            batch_size=1024,
            lr_barrier_coeff=self.args.lr_barrier_coeff,  # ALLO uses 0.01 lr_barrier_coeff
            discount=self.args.allo_discount_factor,  # ALLO uses 0.99 discount
            device=self.args.device,
        )

        # Step 1: Search for .pth files in the directory
        model_dir = f"model/{self.args.env_name}/"
        pth_files = glob.glob(os.path.join(model_dir, "*.pth"))

        if not pth_files:
            print(
                f"[INFO] No existing model found in {model_dir}. Training from scratch."
            )
            epochs = 0
            model_path = os.path.join(
                model_dir,
                f"ALLO_{self.args.extractor_epochs}_{self.args.allo_discount_factor}.pth",
            )
        else:
            print(f"[INFO] Found {len(pth_files)} .pth files in {model_dir}")
            epochs = []
            discount_factors = []
            valid_files = []

            for pth_file in pth_files:
                filename = os.path.basename(pth_file)
                parts = filename.replace(".pth", "").split("_")
                if len(parts) != 3:
                    print(f"[WARNING] Skipping malformed file: {filename}")
                    continue

                _, epoch_str, discount_str = parts
                try:
                    epoch = int(epoch_str)
                    discount = float(discount_str)
                    epochs.append(epoch)
                    discount_factors.append(discount)
                    valid_files.append(filename)
                except ValueError:
                    print(f"[WARNING] Failed to parse file: {filename}")
                    continue

            if self.args.allo_discount_factor not in discount_factors:
                print(
                    f"[INFO] No model with discount factor {self.args.allo_discount_factor} found. Starting fresh."
                )
                epochs = 0
                model_path = os.path.join(
                    model_dir,
                    f"ALLO_{self.args.extractor_epochs}_{self.args.allo_discount_factor}.pth",
                )
            else:
                matching = [
                    (e, f, filename)
                    for e, f, filename in zip(epochs, discount_factors, valid_files)
                    if f == self.args.allo_discount_factor
                ]

                max_epoch, _, _ = max(matching, key=lambda x: x[0])
                idx = epochs.index(max_epoch)
                filename = matching[idx][-1]
                model_path = os.path.join(model_dir, filename)
                print(
                    f"[INFO] Loading model from: {model_path} (epoch {max_epoch}, discount {self.args.allo_discount_factor})"
                )

                extractor.load_state_dict(
                    torch.load(model_path, map_location=self.args.device)
                )
                extractor.to(self.args.device)
                epochs = max_epoch  # set current epoch

        if epochs < self.args.extractor_epochs:
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
                epochs=self.args.extractor_epochs - epochs,
            )
            final_timesteps = trainer.train()
            self.current_timesteps += final_timesteps

            torch.save(extractor.state_dict(), model_path)

        self.extractor = extractor

    def define_eigenvectors(self):
        # === Define eigenvectors === #
        # ALLO does not have explicit eigenvectors.
        # Instead, we make list that contains the eigenvector index and sign
        eigenvectors = [
            (n // 2 + 1, 2 * (n % 2) - 1) for n in range(self.args.num_options)
        ]

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
