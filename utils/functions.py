import os
import random

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

import gymnasium_robotics
from log.wandb_logger import WandbLogger
from utils.wrapper import FetchWrapper, GridWrapper, PointMazeWrapper

EPI_LENGTH = {
    "fourrooms-v0": 100,
    "maze-v0": 200,
    "maze-v1": 200,
    "ant-v5": 1000,
    "walker-v5": 1000,
    "hopper-v5": 1000,
    "pointmaze-v0": 500,
    "pointmaze-v1": 500,
    "fetchreach-v0": 50,
}


def call_env(args, random_spawn: bool = False):
    """
    Call the environment based on the given name.
    """

    max_steps = EPI_LENGTH[args.env_name]
    args.episode_len = max_steps

    env_name, version = args.env_name.split("-")
    version = int(version[-1]) if version[-1].isdigit() else version[-1]
    if env_name in ("fourrooms", "ninerooms", "maze"):
        # For grid environment, we do not use an observation normalization
        # since the most of the state is static (walls) that leads to 0 std.
        if env_name == "fourrooms":
            from gridworld.envs.fourrooms import FourRooms

            env = FourRooms(grid_type=version, max_steps=max_steps)

        elif env_name == "maze":
            from gridworld.envs.maze import Maze

            env = Maze(grid_type=version, max_steps=max_steps)

        env = GridWrapper(env)

        args.positional_indices = [0, 1]
        args.state_dim = env.observation_space.shape[0]
        args.action_dim = env.action_space.n
        args.is_discrete = env.action_space.__class__.__name__ == "Discrete"
    elif env_name == "ant":
        env = gym.make("Ant-v5", max_episode_steps=max_steps, render_mode="rgb_array")
        env.max_steps = max_steps
        env.get_rewards_heatmap = lambda extractor, eigenvectors: None

        args.state_dim = env.observation_space.shape
        args.positional_indices = range(0, 105)
        args.action_dim = env.action_space.shape[0]
        args.is_discrete = env.action_space.__class__.__name__ == "Discrete"
    elif env_name == "walker":
        env = gym.make(
            "Walker2d-v5", max_episode_steps=max_steps, render_mode="rgb_array"
        )
        env.max_steps = max_steps
        args.state_dim = env.observation_space.shape
        args.positional_indices = range(0, 17)
        args.action_dim = env.action_space.shape[0]
        args.is_discrete = env.action_space.__class__.__name__ == "Discrete"
    elif env_name == "fetchreach":

        gym.register_envs(gymnasium_robotics)

        env = gym.make(
            "FetchReach-v4",
            max_episode_steps=max_steps,
            render_mode="rgb_array",
        )

        env = FetchWrapper(env, max_steps, args.seed)

        args.positional_indices = [-6, -5, -4]
        args.state_dim = (
            env.observation_space["observation"].shape[0]
            + env.observation_space["achieved_goal"].shape[0]
            + env.observation_space["desired_goal"].shape[0],
        )
        args.action_dim = env.action_space.shape[0]
        args.is_discrete = env.action_space.__class__.__name__ == "Discrete"

    elif env_name == "pointmaze":
        gym.register_envs(gymnasium_robotics)
        if version == 0:
            if random_spawn:
                example_map = [
                    [1, 1, 1, 1, 1, 1],
                    [1, "c", 1, "c", "c", 1],
                    [1, "c", 1, 1, "c", 1],
                    [1, "c", "c", "c", "c", 1],
                    [1, 1, 1, 1, 1, 1],
                ]
                continuing_task = True
            else:
                example_map = [
                    [1, 1, 1, 1, 1, 1],
                    [1, "r", 1, "g", 0, 1],
                    [1, 0, 1, 1, 0, 1],
                    [1, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1],
                ]
                continuing_task = False
        elif version == 1:
            if random_spawn:
                example_map = [
                    [1, 1, 1, 1, 1, 1],
                    [1, "c", "c", "c", "c", 1],
                    [1, 1, 1, 1, "c", 1],
                    [1, "c", "c", "c", "c", 1],
                    [1, 1, 1, 1, 1, 1],
                ]
                continuing_task = True
            else:
                example_map = [
                    [1, 1, 1, 1, 1, 1],
                    [1, "g", 0, 0, 0, 1],
                    [1, 1, 1, 1, 0, 1],
                    [1, "r", 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1],
                ]
                continuing_task = False

        env = gym.make(
            "PointMaze_UMaze-v3",
            maze_map=example_map,
            max_episode_steps=max_steps,
            continuing_task=continuing_task,
            render_mode="rgb_array",
        )

        env = PointMazeWrapper(env, example_map, max_steps, args.seed)

        args.positional_indices = [-4, -3]
        args.state_dim = (
            env.observation_space["observation"].shape[0]
            + env.observation_space["achieved_goal"].shape[0]
            + env.observation_space["desired_goal"].shape[0],
        )
        args.action_dim = env.action_space.shape[0]
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
