import torch
import torch.nn as nn

from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from policy.trpo import TRPO_Learner
from trainer.base_trainer import Trainer
from utils.sampler import OnlineSampler


class TRPO_Algorithm(nn.Module):
    def __init__(self, env, logger, writer, args):
        super(TRPO_Algorithm, self).__init__()

        # === Parameter saving === #
        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args

        self.args.nupdates = args.timesteps // args.batch_size

    def begin_training(self):
        # === Define policy === #
        self.define_policy()

        # === Sampler === #
        sampler = OnlineSampler(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            episode_len=self.env.max_steps,
            batch_size=self.args.batch_size,
        )

        trainer = Trainer(
            env=self.env,
            policy=self.policy,
            sampler=sampler,
            logger=self.logger,
            writer=self.writer,
            timesteps=self.args.timesteps,
            log_interval=self.args.log_interval,
            eval_num=self.args.eval_num,
            rendering=self.args.rendering,
            seed=self.args.seed,
        )

        trainer.train()

    def define_policy(self):
        actor = PPO_Actor(
            input_dim=self.args.state_dim,
            hidden_dim=self.args.actor_fc_dim,
            action_dim=self.args.action_dim,
            is_discrete=self.args.is_discrete,
        )
        critic = PPO_Critic(self.args.state_dim, hidden_dim=self.args.critic_fc_dim)

        self.policy = TRPO_Learner(
            actor=actor,
            critic=critic,
            nupdates=self.args.nupdates,
            critic_lr=self.args.critic_lr,
            entropy_scaler=self.args.entropy_scaler,
            batch_size=self.args.batch_size,
            target_kl=self.args.target_kl,
            gamma=self.args.gamma,
            gae=self.args.gae,
            device=self.args.device,
        )
