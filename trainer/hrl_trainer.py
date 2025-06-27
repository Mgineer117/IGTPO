import gc
import os
import time
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from log.wandb_logger import WandbLogger
from policy.layers.base import Base
from policy.uniform_random import UniformRandom
from utils.rl import estimate_advantages
from utils.sampler import OnlineSampler


def compare_weights(policy1, policy2):
    diffs = {}
    for (name1, param1), (name2, param2) in zip(
        policy1.named_parameters(), policy2.named_parameters()
    ):
        assert name1 == name2, "Parameter names do not match"
        diff = torch.norm(param1.data - param2.data).item()
        diffs[name1] = diff
    return diffs


# model-free policy trainer
class HRLTrainer:
    def __init__(
        self,
        env: gym.Env,
        hl_policy: Base,
        policies: Base,
        intrinsic_reward_fn,
        hl_sampler: OnlineSampler,
        sampler: OnlineSampler,
        logger: WandbLogger,
        writer: SummaryWriter,
        init_timesteps: int = 0,
        timesteps: int = 1e6,
        hl_timesteps: int = 1e6,
        log_interval: int = 100,
        eval_num: int = 10,
        rendering: bool = False,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.hl_policy = hl_policy
        self.policies = policies

        self.intrinsic_reward_fn = intrinsic_reward_fn

        self.num_vectors = len(self.policies) - 1

        self.hl_sampler = hl_sampler
        self.sampler = sampler
        self.eval_num = eval_num

        self.logger = logger
        self.writer = writer

        # training parameters
        self.init_timesteps = init_timesteps
        self.timesteps = timesteps
        self.hl_timesteps = hl_timesteps

        self.log_interval = log_interval
        self.eval_interval = int(self.timesteps / self.log_interval)

        # initialize the essential training components
        self.last_min_return_mean = 1e10
        self.last_min_return_std = 1e10

        self.rendering = rendering
        self.seed = seed

    def train(self) -> dict[str, float]:
        start_time = time.time()

        self.last_return_mean = deque(maxlen=5)
        self.last_return_std = deque(maxlen=5)

        # Train loop
        eval_idx = 0
        total_tiemesteps = int(self.timesteps * self.num_vectors + self.init_timesteps)
        with tqdm(
            total=total_tiemesteps,
            initial=self.init_timesteps,
            desc=f"{self.hl_policy.name} Training (Timesteps)",
        ) as pbar:
            for option_idx in range(self.num_vectors):
                while pbar.n < int(
                    (option_idx + 1) * (self.timesteps + self.init_timesteps)
                ):
                    # --- START OF EPOCH/ITERATION ---
                    current_step = pbar.n

                    policy = self.policies[option_idx]
                    policy.train()

                    # === Initial Iteration ===
                    batch, sample_time = self.sampler.collect_samples(
                        env=self.env,
                        policy=policy,
                        seed=self.seed,
                        # random_init_pos=True,
                    )

                    states, next_states = batch["states"], batch["next_states"]
                    policy.record_state_visitations(states)
                    states = torch.from_numpy(states).to(policy.device)
                    next_states = torch.from_numpy(next_states).to(policy.device)

                    intrinsic_rewards, _ = self.intrinsic_reward_fn(
                        states, next_states, option_idx
                    )
                    batch["rewards"] = intrinsic_rewards.cpu().numpy()
                    loss_dict, timesteps, update_time = policy.learn(batch)

                    # add timesteps
                    current_step += timesteps
                    pbar.update(timesteps)

                    # Calculate expected remaining time
                    elapsed_time = time.time() - start_time
                    avg_time_per_iter = elapsed_time / current_step
                    remaining_time = avg_time_per_iter * (
                        total_tiemesteps - current_step
                    )

                    # Update environment steps and calculate time metrics
                    loss_dict[
                        f"{self.policies[option_idx].name}/analytics/timesteps"
                    ] = (current_step + timesteps)
                    loss_dict[
                        f"{self.policies[option_idx].name}/analytics/sample_time"
                    ] = sample_time
                    loss_dict[
                        f"{self.policies[option_idx].name}/analytics/update_time"
                    ] = update_time
                    loss_dict[
                        f"{self.policies[option_idx].name}/analytics/remaining_time (hr)"
                    ] = (
                        remaining_time / 3600
                    )  # Convert to hours

                    self.write_log(loss_dict, step=current_step)

        # assign trained option policies
        eval_idx = 0
        init_timesteps = current_step
        total_tiemesteps = init_timesteps + self.hl_timesteps
        self.hl_policy.policies = self.policies
        with tqdm(
            total=total_tiemesteps,
            initial=init_timesteps,
            desc=f"{self.hl_policy.name} Training (Timesteps)",
        ) as pbar:
            while pbar.n < total_tiemesteps:
                current_step = pbar.n
                self.hl_policy.train()

                batch, sample_time = self.hl_sampler.collect_samples(
                    env=self.env, policy=self.hl_policy, seed=self.seed
                )
                loss_dict, timesteps, update_time = self.hl_policy.learn(batch)

                # add timesteps
                current_step += timesteps
                pbar.update(timesteps)

                # Calculate expected remaining time
                elapsed_time = time.time() - start_time
                avg_time_per_iter = elapsed_time / current_step
                remaining_time = avg_time_per_iter * (total_tiemesteps - current_step)

                # Update environment steps and calculate time metrics
                loss_dict[f"{self.hl_policy.name}/analytics/timesteps"] = (
                    current_step + timesteps
                )
                loss_dict[f"{self.hl_policy.name}/analytics/sample_time"] = sample_time
                loss_dict[f"{self.hl_policy.name}/analytics/update_time"] = update_time
                loss_dict[f"{self.hl_policy.name}/analytics/remaining_time (hr)"] = (
                    remaining_time / 3600
                )  # Convert to hours

                self.write_log(loss_dict, step=current_step)

                #### EVALUATIONS ####
                if current_step >= self.eval_interval * (eval_idx + 1):
                    ### Eval Loop
                    self.hl_policy.eval()
                    eval_idx += 1

                    eval_dict, running_video = self.hl_evaluate()

                    # Manual logging
                    if self.hl_policy.state_visitation is not None:
                        visitation_map = self.hl_policy.state_visitation
                        vmin, vmax = visitation_map.min(), visitation_map.max()
                        visitation_map = (visitation_map - vmin) / (vmax - vmin + 1e-8)
                        visitation_map = self.visitation_to_rgb(visitation_map)
                        self.write_image(
                            image=visitation_map,
                            step=current_step,
                            logdir="Image",
                            name="visitation map",
                        )

                    self.write_log(eval_dict, step=current_step, eval_log=True)
                    self.write_video(
                        running_video,
                        step=current_step,
                        logdir=f"videos",
                        name="running_video",
                    )

                    self.last_return_mean.append(eval_dict[f"eval/return_mean"])
                    self.last_return_std.append(eval_dict[f"eval/return_std"])

                    self.save_model(current_step, self.hl_policy, "hl_policy")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.logger.print(
            f"Total {self.hl_policy.name} training time: {(time.time() - start_time) / 3600} hours"
        )

        return current_step

    def evaluate(self, policy: nn.Module):
        ep_buffer = []
        image_array = []
        for num_episodes in range(self.eval_num):
            ep_reward = []

            # Env initialization
            state, infos = self.env.reset(seed=self.seed)

            for t in range(self.env.max_steps):
                with torch.no_grad():
                    a, _ = policy(state, deterministic=True)
                    a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]

                if num_episodes == 0 and self.rendering:
                    image = self.env.render()
                    image_array.append(image)

                next_state, rew, term, trunc, infos = self.env.step(a)
                done = term or trunc

                state = next_state
                ep_reward.append(rew)

                if done:
                    ep_buffer.append(
                        {
                            "return": self.discounted_return(ep_reward, policy.gamma),
                        }
                    )

                    break

        return_list = [ep_info["return"] for ep_info in ep_buffer]
        return_mean, return_std = np.mean(return_list), np.std(return_list)

        eval_dict = {
            f"eval/return_mean": return_mean,
            f"eval/return_std": return_std,
        }

        return eval_dict, image_array

    def hl_evaluate(self):
        ep_buffer = []
        image_array = []
        for num_episodes in range(self.eval_num):
            ep_reward = []

            # Env initialization
            state, infos = self.env.reset(seed=self.seed)

            for t in range(self.env.max_steps):
                with torch.no_grad():
                    [option_idx, a], metaData = self.hl_policy(
                        state, None, deterministic=True
                    )
                    a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]

                if num_episodes == 0 and self.rendering:
                    image = self.env.render()
                    image_array.append(image)

                if metaData["is_option"]:
                    option_termination = False
                    for i in range(10):
                        next_state, rew, term, trunc, infos = self.env.step(a)
                        done = term or trunc
                        ep_reward.append(rew)

                        if done or option_termination:
                            break
                        else:
                            with torch.no_grad():
                                [_, a], optionMetaData = self.hl_policy(
                                    next_state,
                                    option_idx=option_idx,
                                    deterministic=True,
                                )
                                a = (
                                    a.cpu().numpy().squeeze(0)
                                    if a.shape[-1] > 1
                                    else [a.item()]
                                )
                            option_termination = optionMetaData["option_termination"]
                else:
                    # env stepping
                    next_state, rew, term, trunc, infos = self.env.step(a)
                    done = term or trunc
                    ep_reward.append(rew)

                state = next_state

                if done:
                    ep_buffer.append(
                        {
                            "return": self.discounted_return(
                                ep_reward, self.hl_policy.gamma
                            ),
                        }
                    )

                    break

        return_list = [ep_info["return"] for ep_info in ep_buffer]
        return_mean, return_std = np.mean(return_list), np.std(return_list)

        eval_dict = {
            f"eval/return_mean": return_mean,
            f"eval/return_std": return_std,
        }

        return eval_dict, image_array

    def discounted_return(self, rewards, gamma):
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
        return G

    def write_log(self, logging_dict: dict, step: int, eval_log: bool = False):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(step, eval_log=eval_log, display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, step)

    def write_image(self, image: np.ndarray, step: int, logdir: str, name: str):
        image_list = image if isinstance(image, list) else [image]
        image_path = os.path.join(logdir, name)
        self.logger.write_images(step=step, images=image_list, logdir=image_path)

    def write_video(self, image: list, step: int, logdir: str, name: str):
        if len(image) > 0:
            tensor = np.stack(image, axis=0)
            video_path = os.path.join(logdir, name)
            self.logger.write_videos(step=step, images=tensor, logdir=video_path)

    def save_model(self, e: int, model: nn.Module, name: str):
        ### save checkpoint
        name = f"{name}_{e}.pth"
        path = os.path.join(self.logger.checkpoint_dir, name)

        if model is not None:
            model = deepcopy(model).to("cpu")
            torch.save(model.state_dict(), path)

            # save the best model
            if (
                np.mean(self.last_return_mean) < self.last_min_return_mean
                and np.mean(self.last_return_std) <= self.last_min_return_std
            ):
                name = f"best_model.pth"
                path = os.path.join(self.logger.log_dir, name)
                torch.save(model.state_dict(), path)

                self.last_min_return_mean = np.mean(self.last_return_mean)
                self.last_min_return_std = np.mean(self.last_return_std)
        else:
            raise ValueError("Error: Model is not identifiable!!!")

    def visitation_to_rgb(self, visitation_map: np.ndarray) -> np.ndarray:
        visitation_map = np.squeeze(visitation_map)  # Make sure it's 2D
        H, W = visitation_map.shape

        rgb_map = np.ones((H, W, 3), dtype=np.float32)  # Start with white

        # Zero visitation → gray
        zero_mask = visitation_map == 0
        rgb_map[zero_mask] = [0.5, 0.5, 0.5]

        # Nonzero visitation → white → blue gradient
        nonzero_mask = visitation_map > 0
        blue_intensity = visitation_map[nonzero_mask]

        rgb_map[nonzero_mask] = np.stack(
            [
                1.0 - blue_intensity,  # Red
                1.0 - blue_intensity,  # Green
                np.ones_like(blue_intensity),  # Blue
            ],
            axis=-1,
        )

        return rgb_map
