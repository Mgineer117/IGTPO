import numpy as np
import torch
import torch.nn as nn

from utils.sampler import OnlineSampler


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
                minibatch_size=args.minibatch_size,
                device=args.device,
            )
        elif args.intrinsic_reward_mode in ("eigenpurpose", "all"):
            extractor = Extractor(
                network=feature_network,
                extractor_lr=args.extractor_lr,
                epochs=args.extractor_epochs,
                minibatch_size=args.minibatch_size,
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

    batch, _ = sampler.collect_samples(env, uniform_random_policy, args.seed)

    if args.intrinsic_reward_mode in ("eigenpurpose", "all"):
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
