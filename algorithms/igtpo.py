import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from policy.igtpo import IGTPO_Learner
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from policy.ppo import PPO_Learner
from policy.uniform_random import UniformRandom
from policy.value_functions import Critic_Learner, Critics_Learner
from trainer.base_trainer import Trainer
from trainer.extractor_trainer import ExtractorTrainer
from trainer.igtpo_trainer import IGTPOTrainer
from trainer.metaigtpo_trainer import MetaIGTPOTrainer
from utils.rl import get_extractor, get_vector
from utils.sampler import OnlineSampler
from utils.wrapper import RunningMeanStd


class IntrinsicRewardFunctions(nn.Module):
    def __init__(self, env, logger, writer, args):
        super(IntrinsicRewardFunctions, self).__init__()

        # === Parameter saving === #
        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args

        self.current_timesteps = 0
        self.loss_dict = {}

        self.intrinsic_reward_mode = self.args.intrinsic_reward_mode

        if self.intrinsic_reward_mode == "eigenpurpose":
            self.define_extractor()
            self.define_eigenvectors()
            self.num_rewards = self.args.num_options
        elif self.intrinsic_reward_mode == "allo":
            self.define_allo()
            self.define_eigenvectors()
            self.num_rewards = self.args.num_options
        elif self.intrinsic_reward_mode == "drnd":
            self.define_drnd_policy()
            self.num_rewards = 1
        elif self.intrinsic_reward_mode == "all":
            # eigenpurpose + drnd
            self.define_extractor()
            self.define_eigenvectors()
            self.define_drnd_policy()
            self.num_rewards = self.args.num_options + 1

    def forward(self, states: torch.Tensor, next_states: torch.Tensor, i: int):
        if self.intrinsic_reward_mode == "eigenpurpose":
            # get features
            with torch.no_grad():
                feature, _ = self.extractor(states)
                next_feature, _ = self.extractor(next_states)

                difference = next_feature - feature

            # Calculate the intrinsic reward using the eigenvector
            intrinsic_rewards = torch.matmul(
                difference, self.eigenvectors[i].unsqueeze(-1)
            ).to(self.args.device)
            self.reward_rms[i].update(intrinsic_rewards.cpu().numpy())

            var_tensor = torch.as_tensor(
                self.reward_rms[i].var,
                device=intrinsic_rewards.device,
                dtype=intrinsic_rewards.dtype,
            )
            intrinsic_rewards = intrinsic_rewards / (torch.sqrt(var_tensor) + 1e-8)

            source = "eigenpurpose"

        elif self.intrinsic_reward_mode == "allo":
            with torch.no_grad():
                feature, _ = self.extractor(states)
                next_feature, _ = self.extractor(next_states)

                difference = next_feature - feature
                intrinsic_rewards = difference[:, i].unsqueeze(-1)
            source = "allo"

        elif self.intrinsic_reward_mode == "drnd":
            with torch.no_grad():
                intrinsic_rewards = self.drnd_policy.intrinsic_reward(next_states)
            source = "drnd"
        elif self.intrinsic_reward_mode == "all":
            if i < self.num_rewards - 1:
                # get features
                with torch.no_grad():
                    feature, _ = self.extractor(states)
                    next_feature, _ = self.extractor(next_states)

                    difference = next_feature - feature

                # Calculate the intrinsic reward using the eigenvector
                intrinsic_rewards = torch.matmul(
                    difference, self.eigenvectors[i].unsqueeze(-1)
                ).to(self.args.device)
                self.reward_rms[i].update(intrinsic_rewards.cpu().numpy())

                var_tensor = torch.as_tensor(
                    self.reward_rms[i].var,
                    device=intrinsic_rewards.device,
                    dtype=intrinsic_rewards.dtype,
                )
                intrinsic_rewards = intrinsic_rewards / (torch.sqrt(var_tensor) + 1e-8)

                source = "eigenpurpose"
            else:
                with torch.no_grad():
                    intrinsic_rewards = self.drnd_policy.intrinsic_reward(next_states)
                source = "drnd"

        return intrinsic_rewards, source

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
        if not os.path.exists("model"):
            os.makedirs("model")
        if self.args.intrinsic_reward_mode == "all":
            model_path = f"model/{self.args.env_name}-eigenpurpose-feature_network.pth"
        else:
            model_path = f"model/{self.args.env_name}-{self.args.intrinsic_reward_mode}-feature_network.pth"
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
                batch_size=16384,
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
                batch_size=self.args.batch_size,
            )
            final_timesteps = trainer.train()
            torch.save(extractor.state_dict(), model_path)

            self.current_timesteps += final_timesteps
        else:
            extractor.load_state_dict(
                torch.load(model_path, map_location=self.args.device)
            )
            extractor.to(self.args.device)

        self.extractor = extractor

    def define_eigenvectors(self):
        # === Define eigenvectors === #
        eigenvectors, heatmaps = get_vector(self.env, self.extractor, self.args)
        if eigenvectors is not None:
            self.eigenvectors = torch.from_numpy(eigenvectors).to(self.args.device)
            self.reward_rms = []
            for _ in range(self.eigenvectors.shape[0]):
                self.reward_rms.append(RunningMeanStd(shape=(1,)))
        self.logger.write_images(
            step=self.current_timesteps, images=heatmaps, logdir="Image/Heatmaps"
        )

    def define_allo(self):
        if not os.path.exists("model"):
            os.makedirs("model")
        model_path = f"model/{self.args.env_name}-{self.args.intrinsic_reward_mode}-feature_network.pth"
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
                batch_size=16384,
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
                batch_size=self.args.batch_size,
            )
            final_timesteps = trainer.train()
            torch.save(extractor.state_dict(), model_path)

            self.current_timesteps += final_timesteps
        else:
            extractor.load_state_dict(
                torch.load(model_path, map_location=self.args.device)
            )
            extractor.to(self.args.device)

        self.extractor = extractor

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
            drnd_lr=3e-4,
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


class IGTPO_Algorithm(nn.Module):
    def __init__(self, env, logger, writer, args):
        super(IGTPO_Algorithm, self).__init__()

        # === Parameter saving === #
        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args

        self.intrinsic_reward_fn = IntrinsicRewardFunctions(
            env=env, logger=logger, writer=writer, args=args
        )

        self.args.igtpo_nupdates = args.timesteps // (
            args.batch_size * args.num_inner_updates * args.num_options
        )

        self.current_timesteps = self.intrinsic_reward_fn.current_timesteps

    def begin_training(self):
        # === Sampler === #
        outer_sampler = OnlineSampler(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            episode_len=self.env.max_steps,
            batch_size=self.args.batch_size,
        )
        inner_sampler = OnlineSampler(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            episode_len=self.env.max_steps,
            batch_size=self.args.batch_size // 4,
            verbose=False,
        )
        # === Meta-train using options === #'
        self.define_outer_policy()
        trainer = IGTPOTrainer(
            env=self.env,
            policy=self.policy,
            outer_sampler=outer_sampler,
            inner_sampler=inner_sampler,
            logger=self.logger,
            writer=self.writer,
            init_timesteps=self.current_timesteps,
            timesteps=self.args.timesteps,
            log_interval=self.args.log_interval,
            eval_num=self.args.eval_num,
            rendering=self.args.rendering,
            seed=self.args.seed,
            args=self.args,
        )
        final_steps = trainer.train()
        self.current_timesteps += final_steps

    def define_outer_policy(self):
        # === Define policy === #
        actor = PPO_Actor(
            input_dim=self.args.state_dim,
            hidden_dim=self.args.actor_fc_dim,
            action_dim=self.args.action_dim,
            is_discrete=self.args.is_discrete,
            device=self.args.device,
        )
        critic = PPO_Critic(self.args.state_dim, hidden_dim=self.args.critic_fc_dim)

        self.policy = IGTPO_Learner(
            actor=actor,
            critic=critic,
            intrinsic_reward_fn=self.intrinsic_reward_fn,
            nupdates=self.args.igtpo_nupdates,
            num_inner_updates=self.args.num_inner_updates,
            actor_lr=self.args.igtpo_actor_lr,
            critic_lr=self.args.critic_lr,
            eps_clip=self.args.eps_clip,
            entropy_scaler=self.args.entropy_scaler,
            target_kl=self.args.target_kl,
            gamma=self.args.gamma,
            gae=self.args.gae,
            device=self.args.device,
        )
