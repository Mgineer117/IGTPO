import numpy as np
import torch
import torch.nn as nn

from utils.sampler import OnlineSampler


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
    from extractor.extractor import DummyExtractor, Extractor
    from utils.cnn_architecture import get_cnn_architecture

    env_name, version = args.env_name.split("-")

    if env_name in ("fourrooms", "ninerooms", "maze"):
        if args.extractor_type == "VAE":
            feature_network = VAE(
                state_dim=args.state_dim,
                action_dim=args.action_dim,
                feature_dim=args.feature_dim,
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
                feature_dim=args.feature_dim,
                encoder_architecture=encoder_architecture,
                decoder_architecture=decoder_architecture,
                activation=nn.Tanh(),
                device=args.device,
            )
        else:
            raise NotImplementedError(f"{args.extractor_type} is not implemented")

        extractor = Extractor(
            network=feature_network,
            extractor_lr=args.extractor_lr,
            epochs=args.extractor_epochs,
            minibatch_size=args.minibatch_size,
            device=args.device,
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
        batch_size=8192,
        verbose=False,
    )

    uniform_random_policy = UniformRandom(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        is_discrete=args.is_discrete,
        device=args.device,
    )

    batch, _ = sampler.collect_samples(env, uniform_random_policy, args.seed)
    data = torch.from_numpy(batch["states"]).to(args.device)
    with torch.no_grad():
        features, _ = extractor(data)

    # perform a spectral analysis
    cov = torch.cov(features.T)
    eigval, eigvec = torch.linalg.eigh(cov)

    eigval = eigval.real.to(args.device)  # Ensure eigenvalues are real
    eigvec = eigvec.to(args.device)  # Ensure eigenvectors are on the correct device

    # print(eigval)

    # Sort eigenvalues in descending order and get indices
    sorted_indices = torch.argsort(eigval, descending=True)

    # Reorder eigenvalues and eigenvectors
    eigval = eigval[sorted_indices]
    eigvec = eigvec[:, sorted_indices]

    # If you want row-wise eigenvectors (each row is one eigenvector)
    eigvec_row = eigvec.T.real  # shape: [n_rows, n_cols]
    eig_vec_row = eigvec_row[: int(args.num_options / 2)]
    eigenvectors = torch.concatenate([eig_vec_row, -eig_vec_row], dim=0)

    eigenvectors = eigenvectors.cpu().numpy()

    heatmaps = env.get_rewards_heatmap(extractor, eigenvectors)

    return eigenvectors, heatmaps
