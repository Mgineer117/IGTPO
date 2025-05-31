import os
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

import gymnasium_robotics
from log.wandb_logger import WandbLogger
from utils.wrapper import FetchWrapper, ObsNormWrapper, PointMazeWrapper


def call_env(args):
    """
    Call the environment based on the given name.
    """

    env_name, version = args.env_name.split("-")

    if env_name in ("fourrooms", "ninerooms", "maze"):
        # For grid environment, we do not use an observation normalization
        # since the most of the state is static (walls) that leads to 0 std.
        version = int(version[-1]) if version[-1].isdigit() else version[-1]
        if env_name == "fourrooms":
            from gridworld.envs.fourrooms import FourRooms

            env = FourRooms(grid_type=version)
            # args.is_discrete = True
        elif env_name == "maze":
            from gridworld.envs.maze import Maze

            env = Maze(grid_type=version)
            # args.is_discrete = True
        elif env_name == "ninerooms":
            from gridworld.envs.ninerooms import NineRooms

            env = NineRooms(grid_type=version)
        else:
            raise ValueError(f"Environment {env_name} is not supported.")

        args.state_dim = env.observation_space.shape
        args.action_dim = env.action_space.n
        args.episode_len = env.max_steps
        args.is_discrete = env.action_space.__class__.__name__ == "Discrete"
    elif env_name == "fetch":
        episode_len = 50
        gym.register_envs(gymnasium_robotics)

        if version == "reach":
            env = gym.make(
                "FetchReach-v4", max_episode_steps=episode_len, render_mode="rgb_array"
            )
        elif version == "reachdense":
            env = gym.make(
                "FetchReachDense-v4",
                max_episode_steps=episode_len,
                render_mode="rgb_array",
            )
        elif version == "push":
            env = gym.make(
                "FetchPush-v4",
                max_episode_steps=episode_len,
                render_mode="rgb_array",
            )
        elif version == "pushdense":
            env = gym.make(
                "FetchPushDense-v4",
                max_episode_steps=episode_len,
                render_mode="rgb_array",
            )
        elif version == "pickandplace":
            env = gym.make(
                "FetchPickAndPlace-v4",
                max_episode_steps=episode_len,
                render_mode="rgb_array",
            )
        elif version == "pickandplacedense":
            env = gym.make(
                "FetchPickAndPlaceDense-v4",
                max_episode_steps=episode_len,
                render_mode="rgb_array",
            )
        else:
            NotImplementedError(f"Version {version} is not implemented.")

        env = FetchWrapper(env, episode_len, args.seed)
        env = ObsNormWrapper(env)

        args.state_dim = (
            env.observation_space["observation"].shape[0]
            + env.observation_space["achieved_goal"].shape[0]
            + env.observation_space["desired_goal"].shape[0],
        )
        args.action_dim = env.action_space.shape[0]
        args.episode_len = episode_len
        args.is_discrete = env.action_space.__class__.__name__ == "Discrete"

    elif env_name == "pointmaze":
        episode_len = 300
        gym.register_envs(gymnasium_robotics)
        example_map = [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 1, "g", 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, "r", 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]

        if version == "medium":
            env = gym.make(
                "PointMaze_UMaze-v3",
                maze_map=example_map,
                max_episode_steps=episode_len,
                continuing_task=False,
                render_mode="rgb_array",
            )
        elif version == "densemedium":
            env = gym.make(
                "PointMaze_UMazeDense-v3",
                maze_map=example_map,
                max_episode_steps=episode_len,
                continuing_task=False,
                render_mode="rgb_array",
            )
        else:
            NotImplementedError(f"Version {version} is not implemented.")

        env = PointMazeWrapper(env, example_map, episode_len, args.seed)
        env = ObsNormWrapper(env)

        args.state_dim = (
            env.observation_space["observation"].shape[0]
            + env.observation_space["achieved_goal"].shape[0]
            + env.observation_space["desired_goal"].shape[0],
        )
        args.action_dim = env.action_space.shape[0]
        args.episode_len = episode_len
        args.is_discrete = env.action_space.__class__.__name__ == "Discrete"

    return env


def setup_logger(args, unique_id, exp_time, seed):
    """
    setup logger both using WandB and Tensorboard
    Return: WandB logger, Tensorboard logger
    """
    # Get the current date and time
    if args.group is None:
        args.group = "-".join((exp_time, unique_id))

    if args.name is None:
        args.name = "-".join(
            (args.algo_name, args.env_name, unique_id, "seed:" + str(seed))
        )

    if args.project is None:
        args.project = args.task

    args.logdir = os.path.join(args.logdir, args.group)

    default_cfg = vars(args)
    logger = WandbLogger(
        config=default_cfg,
        project=args.project,
        group=args.group,
        name=args.name,
        log_dir=args.logdir,
        log_txt=True,
    )
    logger.save_config(default_cfg, verbose=True)

    tensorboard_path = os.path.join(logger.log_dir, "tensorboard")
    os.mkdir(tensorboard_path)
    writer = SummaryWriter(log_dir=tensorboard_path)

    return logger, writer


def seed_all(seed=0):
    # Set the seed for hash-based operations in Python
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set the seed for Python's random module
    random.seed(seed)

    # Set the seed for NumPy's random number generator
    np.random.seed(seed)

    # Set the seed for PyTorch (both CPU and GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU setups

    # Ensure reproducibility of PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def temp_seed(seed, pid):
    """
    This saves current seed info and calls after stochastic action selection.
    -------------------------------------------------------------------------
    This is to introduce the stochacity in each multiprocessor.
    Without this, the samples from each multiprocessor will be same since the seed was fixed
    """
    rand_int = random.randint(0, 1_000_000)  # create a random integer

    # Set the temporary seed
    torch.manual_seed(seed + pid + rand_int)
    np.random.seed(seed + pid + rand_int)
    random.seed(seed + pid + rand_int)


def concat_csv_columnwise_and_delete(folder_path, output_file="output.csv"):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in the folder.")
        return

    dataframes = []

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    # Concatenate column-wise (axis=1)
    combined_df = pd.concat(dataframes, axis=1)

    # Save to output file
    output_file = os.path.join(folder_path, output_file)
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to {output_file}")

    # Delete original CSV files
    for file in csv_files:
        os.remove(os.path.join(folder_path, file))

    print("Original CSV files deleted.")
