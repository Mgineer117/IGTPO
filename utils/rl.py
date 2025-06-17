import os

import numpy as np
import torch
import torch.nn as nn

from utils.sampler import OnlineSampler


class IntrinsicRewardFunctions(nn.Module):
    def __init__(self, env, logger, writer, reward_mode, args):
        super(IntrinsicRewardFunctions, self).__init__()

        # === Parameter saving === #
        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args

        self.current_timesteps = 0
        self.loss_dict = {}

        self.intrinsic_reward_mode = reward_mode

        if self.intrinsic_reward_mode == "eigenpurpose":
            self.define_extractor()
            self.define_eigenvectors()
            self.num_rewards = self.args.num_options
            self.sources = ["eigenpurpose" for _ in range(self.num_rewards)]
        elif self.intrinsic_reward_mode == "allo":
            self.define_allo()
            self.define_eigenvectors()
            self.feature_mask = np.arange(self.args.num_options)
            self.num_rewards = self.args.num_options
            self.sources = ["allo" for _ in range(self.num_rewards)]
        elif self.intrinsic_reward_mode == "drnd":
            self.define_drnd_policy()
            self.num_rewards = 1
            self.sources = ["drnd"]
        elif self.intrinsic_reward_mode == "all":
            # eigenpurpose + drnd
            self.define_extractor()
            self.define_eigenvectors()
            self.define_drnd_policy()
            self.num_rewards = self.args.num_options + 1
            self.sources = ["eigenpurpose" for _ in range(self.args.num_options)]
            self.sources.append("drnd")

    def prune(self, i: int):
        if self.num_rewards > 1:
            source = self.sources[i]
            if source == "eigenpurpose":
                self.eigenvectors = torch.cat(
                    (
                        self.eigenvectors[:i],
                        self.eigenvectors[i + 1 :],
                    ),
                    dim=0,
                )
                del self.reward_rms[i]
                del self.sources[i]
                self.num_rewards = len(self.sources)
            elif source == "drnd":
                del self.sources[i]
                self.num_rewards = len(self.sources)
            elif source == "allo":
                del self.sources[i]
                self.feature_mask = np.delete(self.feature_mask, i)
                self.num_rewards = len(self.sources)

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
            ).to(self.args.device)
            self.reward_rms[i].update(intrinsic_rewards.cpu().numpy())

            var_tensor = torch.as_tensor(
                self.reward_rms[i].var,
                device=intrinsic_rewards.device,
                dtype=intrinsic_rewards.dtype,
            )
            intrinsic_rewards = intrinsic_rewards / (torch.sqrt(var_tensor) + 1e-8)
        elif self.sources[i] == "allo":
            with torch.no_grad():
                feature, _ = self.extractor(states)
                next_feature, _ = self.extractor(next_states)

                difference = next_feature - feature
                difference = difference[:, self.feature_mask]
                intrinsic_rewards = difference[:, i].unsqueeze(-1)
        elif self.sources[i] == "drnd":
            with torch.no_grad():
                intrinsic_rewards = self.drnd_policy.intrinsic_reward(next_states)

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
        from policy.uniform_random import UniformRandom
        from trainer.extractor_trainer import ExtractorTrainer

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
                batch_size=50000,
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
        from utils.wrapper import RunningMeanStd

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
        from policy.uniform_random import UniformRandom
        from trainer.extractor_trainer import ExtractorTrainer

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
                batch_size=50000,
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


def flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model, flat_params):
    pointer = 0
    for p in model.parameters():
        num_param = p.numel()
        p.data.copy_(flat_params[pointer : pointer + num_param].view_as(p))
        pointer += num_param


def compute_kl(old_actor, new_policy, obs):
    _, old_infos = old_actor(obs)
    _, new_infos = new_policy(obs)

    kl = torch.distributions.kl_divergence(old_infos["dist"], new_infos["dist"])
    return kl.mean()


def hessian_vector_product(kl_fn, model, damping, v):
    kl = kl_fn()
    grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
    flat_grads = torch.cat([g.view(-1) for g in grads])
    g_v = (flat_grads * v).sum()
    hv = torch.autograd.grad(g_v, model.parameters())
    flat_hv = torch.cat([h.contiguous().view(-1) for h in hv])
    return flat_hv + damping * v


def conjugate_gradients(Av_func, b, nsteps=10, tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(nsteps):
        Avp = Av_func(p)
        alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        if new_rdotr < tol:
            break
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr
    return x


def estimate_advantages(
    rewards: torch.Tensor,
    terminals: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate advantages and returns using Generalized Advantage Estimation (GAE),
    while keeping all operations on the original device.

    Args:
        rewards (Tensor): Reward at each timestep, shape [T, 1]
        terminals (Tensor): Binary terminal indicators (1 if done), shape [T, 1]
        values (Tensor): Value function estimates, shape [T, 1]
        gamma (float): Discount factor.
        gae (float): GAE lambda.

    Returns:
        advantages (Tensor): Estimated advantages, shape [T, 1]
        returns (Tensor): Estimated returns (value targets), shape [T, 1]
    """
    device = rewards.device  # Infer device from input tensor

    T = rewards.size(0)
    deltas = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)

    prev_value = torch.tensor(0.0, device=device)
    prev_advantage = torch.tensor(0.0, device=device)

    for t in reversed(range(T)):
        non_terminal = 1.0 - terminals[t]
        deltas[t] = rewards[t] + gamma * prev_value * non_terminal - values[t]
        advantages[t] = deltas[t] + gamma * gae * prev_advantage * non_terminal

        prev_value = values[t]
        prev_advantage = advantages[t]

    returns = values + advantages
    return advantages, returns


def get_extractor(args):
    from extractor.base.cnn import CNN
    from extractor.base.vae import VAE
    from extractor.extractor import ALLO, DummyExtractor, Extractor
    from utils.cnn_architecture import get_cnn_architecture

    env_name, version = args.env_name.split("-")

    if env_name in ("fourrooms", "ninerooms", "maze"):
        feature_dim = (
            args.num_options
            if args.intrinsic_reward_mode == "allo"
            else args.feature_dim
        )

        if args.extractor_type == "VAE":
            feature_network = VAE(
                state_dim=args.state_dim,
                action_dim=args.action_dim,
                feature_dim=feature_dim,
                encoder_fc_dim=[512, 512, 256, 256],
                decoder_fc_dim=[256, 256, 512, 512],
                activation=nn.Tanh(),
                device=args.device,
            )
        elif args.extractor_type == "CNN":
            encoder_architecture, decoder_architecture = get_cnn_architecture(args)
            feature_network = CNN(
                state_dim=args.state_dim,
                action_dim=args.action_dim,
                feature_dim=feature_dim,
                encoder_architecture=encoder_architecture,
                decoder_architecture=decoder_architecture,
                activation=nn.Tanh(),
                device=args.device,
            )
        else:
            raise NotImplementedError(f"{args.extractor_type} is not implemented")

        if args.intrinsic_reward_mode == "allo":
            extractor = ALLO(
                d=feature_dim,
                network=feature_network,
                extractor_lr=args.extractor_lr,
                epochs=args.extractor_epochs,
                batch_size=1024,
                device=args.device,
            )
        elif args.intrinsic_reward_mode in ("eigenpurpose", "all"):
            extractor = Extractor(
                network=feature_network,
                extractor_lr=args.extractor_lr,
                epochs=args.extractor_epochs,
                batch_size=1024,
                device=args.device,
            )
        else:
            raise NotImplementedError(
                f"intrinsic reward mode {args.intrinsic_reward_mode} is not available."
            )
    elif env_name == "pointmaze":
        # continuous space has pretty small dimension
        # indices is for feature selection of state
        extractor = DummyExtractor(indices=[-2, -1])
    elif env_name == "fetch":
        # continuous space has pretty small dimension
        # if no indices are used, whole state is a feature
        extractor = DummyExtractor(indices=[-3, -2, -1])

    return extractor


def get_vector(env, extractor, args):
    from policy.uniform_random import UniformRandom

    sampler = OnlineSampler(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        episode_len=args.episode_len,
        batch_size=500 * args.episode_len,
        verbose=False,
    )

    uniform_random_policy = UniformRandom(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        is_discrete=args.is_discrete,
        device=args.device,
    )

    if args.intrinsic_reward_mode in ("eigenpurpose", "all"):
        batch, _ = sampler.collect_samples(env, uniform_random_policy, args.seed)
        states = torch.from_numpy(batch["states"]).to(args.device)
        with torch.no_grad():
            features, _ = extractor(states)

        # Covariance-based PCA
        cov = torch.cov(features.T)
        eigval, eigvec = torch.linalg.eigh(cov)
        sorted_indices = torch.argsort(eigval, descending=True)
        eigval = eigval[sorted_indices]
        eigvec = eigvec[:, sorted_indices]
        eigvec_row = eigvec.T.real
        eig_vec_row = eigvec_row[: int(args.num_options / 2)]
        eigenvectors = torch.cat([eig_vec_row, -eig_vec_row], dim=0)
        eigenvectors = eigenvectors.cpu().numpy()

    elif args.intrinsic_reward_mode == "allo":
        eigenvectors = None
    else:
        raise NotImplementedError(f"Mode {args.intrinsic_reward_mode} not supported.")

    heatmaps = env.get_rewards_heatmap(extractor, eigenvectors)

    return eigenvectors, heatmaps
