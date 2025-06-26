import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from policy.igtpo import IGTPO_Learner
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from trainer.igtpo_trainer import IGTPOTrainer
from utils.intrinsic_rewards import IntrinsicRewardFunctions
from utils.sampler import OnlineSampler


class IGTPO_Algorithm(nn.Module):
    def __init__(self, env, logger, writer, args):
        super(IGTPO_Algorithm, self).__init__()

        # === Parameter saving === #
        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args

        self.intrinsic_reward_fn = IntrinsicRewardFunctions(
            logger=logger,
            writer=writer,
            args=args,
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

        if hasattr(self.env, "get_grid"):
            self.policy.actor.grid = self.env.get_grid()
