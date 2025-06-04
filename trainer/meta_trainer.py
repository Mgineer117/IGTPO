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
from policy.base import Base
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
class MetaTrainer:
    def __init__(
        self,
        env: gym.Env,
        policy: Base,
        extractor: nn.Module,
        meta_critic: nn.Module,
        task_critics: nn.Module,
        subtask_critics: nn.Module,
        eigenvectors: torch.Tensor,
        sampler: OnlineSampler,
        logger: WandbLogger,
        writer: SummaryWriter,
        num_local_updates: int = 10,
        init_timesteps: int = 0,
        timesteps: int = 1e6,
        log_interval: int = 100,
        eval_num: int = 10,
        marker: int = 10,
        rendering: bool = False,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.policy = policy
        self.extractor = extractor
        self.meta_critic = meta_critic
        self.task_critics = task_critics
        self.subtask_critics = subtask_critics

        self.eigenvectors = eigenvectors
        self.num_vectors = eigenvectors.shape[0]
        self.num_vector_names = [f"{i}" for i in range(self.num_vectors)]

        self.sampler = sampler
        self.num_local_updates = num_local_updates
        self.eval_num = eval_num

        self.logger = logger
        self.writer = writer

        # training parameters
        self.init_timesteps = init_timesteps
        self.timesteps = timesteps

        self.log_interval = log_interval
        self.prune_interval = int((self.timesteps / 2) / self.num_vectors)
        self.trim_interval = int((self.timesteps / 2) / (self.num_local_updates - 2))
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
        prune_idx = 0
        trim_idx = int(self.timesteps / 2) / self.trim_interval

        with tqdm(
            total=self.timesteps + self.init_timesteps,
            initial=self.init_timesteps,
            desc=f"{self.policy.name} Training (Timesteps)",
        ) as pbar:
            while pbar.n < self.timesteps + self.init_timesteps:
                # --- START OF EPOCH/ITERATION ---
                current_step = pbar.n
                total_timesteps, total_sample_time, total_update_time = 0, 0, 0
                self.policy.train()

                # === Initial Iteration ===
                meta_batch, sample_time = self.sampler.collect_samples(
                    env=self.env, policy=self.policy, seed=self.seed
                )
                meta_timesteps = meta_batch["rewards"].shape[0]
                current_step += meta_timesteps
                total_timesteps += meta_timesteps

                # Meta-critic update
                meta_critic_loss_dict, meta_mean_value = self.meta_critic.learn(
                    meta_batch
                )

                # Setup tracking variables
                policy_dict, gradient_dict = {}, {}
                loss_dict_list = [meta_critic_loss_dict]

                # === First iteration for each vector ===
                for i in range(self.num_vectors):
                    batch_i = deepcopy(meta_batch)

                    # Use intrinsic rewards from eigenvectors
                    batch_i["rewards"] = self.intrinsic_rewards(
                        batch_i, self.eigenvectors[i]
                    )
                    policy = deepcopy(self.policy)
                    critic = self.subtask_critics.critics[i]

                    loss_dict, timesteps, update_time, new_policy, gradients, _ = (
                        policy.learn(critic, batch_i, "local")
                    )

                    # Logging and bookkeeping
                    total_timesteps += timesteps
                    total_sample_time += sample_time
                    total_update_time += update_time
                    loss_dict_list.append(loss_dict)

                    prev_iter_idx = f"0_{i}"
                    policy_dict[prev_iter_idx] = policy
                    gradient_dict[prev_iter_idx] = gradients

                    iter_idx = f"1_{i}"
                    policy_dict[iter_idx] = new_policy

                # === Subsequent Iterations ===
                value_list = []
                for j in range(1, self.num_local_updates):
                    for i in range(self.num_vectors):
                        prefix = "local" if j != self.num_local_updates - 1 else "meta"
                        prev_iter_idx = f"{j}_{i}"
                        policy = policy_dict[prev_iter_idx]

                        task_batch, sample_time = self.sampler.collect_samples(
                            env=self.env, policy=policy, seed=self.seed
                        )

                        option_batch = deepcopy(task_batch)
                        option_batch["rewards"] = self.intrinsic_rewards(
                            option_batch, self.eigenvectors[i]
                        )

                        # Train critics
                        task_loss_dict = self.task_critics.learn(
                            task_batch, idx=i, prefix="task"
                        )
                        subtask_loss_dict = self.subtask_critics.learn(
                            option_batch, idx=i, prefix="subtask"
                        )

                        if prefix == "local":
                            critic = self.subtask_critics.critics[i]
                            (
                                loss_dict,
                                timesteps,
                                update_time,
                                new_policy,
                                gradients,
                                _,
                            ) = policy.learn(critic, option_batch, prefix)
                            loss_dict[
                                f"{policy.name}-{prefix}/analytics/task_rewards"
                            ] = np.mean(task_batch["rewards"])

                        else:
                            critic = self.task_critics.critics[i]
                            (
                                loss_dict,
                                timesteps,
                                update_time,
                                new_policy,
                                gradients,
                                mean_value,
                            ) = policy.learn(critic, task_batch, prefix)

                            value_list.append(mean_value)

                        # Logging and bookkeeping
                        loss_dict.update(task_loss_dict)
                        loss_dict.update(subtask_loss_dict)
                        loss_dict_list.append(loss_dict)
                        total_timesteps += timesteps
                        total_sample_time += sample_time
                        total_update_time += update_time

                        iter_idx = f"{j+1}_{i}"
                        policy_dict[iter_idx] = new_policy
                        gradient_dict[prev_iter_idx] = gradients

                # === Meta-gradient computation ===
                values = np.array(value_list)
                argmax_idx = np.argmax(values)
                most_contributing_index = self.num_vector_names[argmax_idx]

                meta_gradients = []
                for i in range(self.num_vectors):
                    gradients = gradient_dict[f"{self.num_local_updates - 1}_{i}"]
                    for j in reversed(range(self.num_local_updates - 1)):
                        iter_idx = f"{j}_{i}"
                        Hv = grad(
                            gradient_dict[iter_idx],
                            policy_dict[iter_idx].actor.parameters(),
                            grad_outputs=gradients,
                        )
                        gradients = tuple(
                            g - self.policy.igtpo_actor_lr * h
                            for g, h in zip(gradients, Hv)
                        )
                    meta_gradients.append(gradients)

                # Average across vectors
                meta_gradients_transposed = list(
                    zip(*meta_gradients)
                )  # Group by parameter
                averaged_meta_gradients = tuple(
                    torch.mean(torch.stack(grads_per_param), dim=0)
                    for grads_per_param in meta_gradients_transposed
                )

                # === TRPO update === #
                backtrack_iter, backtrack_success = self.policy.trpo_learn(
                    states=meta_batch["states"],
                    grads=averaged_meta_gradients,
                )

                # === Update progress ===
                pbar.update(total_timesteps)

                elapsed_time = time.time() - start_time
                avg_time_per_iter = elapsed_time / current_step
                remaining_time = avg_time_per_iter * (self.timesteps - current_step)

                # Update environment steps and calculate time metrics
                loss_dict = self.average_dict_values(loss_dict_list)
                loss_dict[f"{self.policy.name}/analytics/timesteps"] = current_step
                loss_dict[f"{self.policy.name}/analytics/sample_time"] = (
                    total_sample_time
                )
                loss_dict[f"{self.policy.name}/analytics/update_time"] = (
                    total_update_time
                )
                loss_dict[f"{self.policy.name}/analytics/remaining_time (hr)"] = (
                    remaining_time / 3600
                )  # Convert to hours
                loss_dict[f"{self.policy.name}/parameters/num vectors"] = (
                    self.num_vectors
                )
                loss_dict[f"{self.policy.name}/parameters/num_local_updates"] = (
                    self.num_local_updates
                )
                loss_dict[f"{self.policy.name}/analytics/Contributing Option"] = int(
                    most_contributing_index
                )
                loss_dict[f"{self.policy.name}/analytics/Backtrack_iter"] = (
                    backtrack_iter
                )
                loss_dict[f"{self.policy.name}/analytics/Backtrack_success"] = (
                    backtrack_success
                )
                loss_dict[f"{self.policy.name}/analytics/target_kl"] = (
                    self.policy.target_kl
                )

                self.write_log(loss_dict, step=current_step)

                # === reduce target_kl === #
                self.policy.lr_scheduler(
                    current_step / (self.timesteps + self.init_timesteps)
                )

                # === PRUNE TWIG === #
                if self.num_vectors > 1:
                    if current_step > self.prune_interval * (prune_idx + 1):
                        prune_idx += 1

                        least_contributing_index = np.argmin(values)
                        self.eigenvectors = np.concatenate(
                            [
                                self.eigenvectors[:least_contributing_index],
                                self.eigenvectors[least_contributing_index + 1 :],
                            ],
                            axis=0,
                        )

                        # Prune the corresponding optimizer
                        del self.subtask_critics.critics[least_contributing_index]
                        del self.subtask_critics.optimizers[least_contributing_index]
                        del self.num_vector_names[least_contributing_index]

                        # Update the number of vectors
                        self.num_vectors = self.eigenvectors.shape[0]

                # === TRIM TWIG === #
                if self.num_local_updates > 2:
                    if current_step > self.trim_interval * trim_idx:
                        self.num_local_updates -= 1

                # === EVALUATIONS === #
                if current_step >= self.eval_interval * eval_idx:
                    ### Eval Loop
                    self.policy.eval()
                    eval_idx += 1

                    eval_dict, running_video = self.evaluate()

                    # Manual logging
                    self.write_log(eval_dict, step=current_step, eval_log=True)
                    self.write_video(
                        running_video,
                        step=current_step,
                        logdir=f"videos",
                        name="running_video",
                    )

                    self.last_return_mean.append(eval_dict[f"eval/return_mean"])
                    self.last_return_std.append(eval_dict[f"eval/return_std"])

                    self.save_model(current_step)

                del meta_batch, option_batch, task_batch, policy_dict, gradient_dict
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.logger.print(
            f"Total {self.policy.name} training time: {(time.time() - start_time) / 3600} hours"
        )

        return current_step

    def evaluate(self):
        ep_buffer = []
        image_array = []
        for num_episodes in range(self.eval_num):
            ep_reward = []

            # Env initialization
            state, infos = self.env.reset(seed=self.seed)

            for t in range(self.env.max_steps):
                with torch.no_grad():
                    a, _ = self.policy(state, deterministic=True)
                    a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]

                if num_episodes == 0 and self.rendering:
                    # Plotting
                    image = self.env.render()
                    image_array.append(image)

                next_state, rew, term, trunc, infos = self.env.step(a)
                done = term or trunc

                state = next_state
                ep_reward.append(rew)

                if done:
                    ep_buffer.append(
                        {
                            "return": self.discounted_return(
                                ep_reward, self.policy.gamma
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

    def intrinsic_rewards(self, batch, eigenvector):
        states = batch["states"]
        next_states = batch["next_states"]

        # get features
        with torch.no_grad():
            feature, _ = self.extractor(states)
            next_feature, _ = self.extractor(next_states)

            difference = next_feature - feature
            difference = difference.cpu().numpy()

        # Calculate the intrinsic reward using the eigenvector
        intrinsic_rewards = np.matmul(difference, eigenvector[:, np.newaxis])

        return intrinsic_rewards

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

    def save_model(self, e):
        ### save checkpoint
        name = f"model_{e}.pth"
        path = os.path.join(self.logger.checkpoint_dir, name)

        model = self.policy.actor

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

    def average_dict_values(self, dict_list):
        if not dict_list:
            return {}

        # Initialize a dictionary to hold the sum of values and counts for each key
        sum_dict = {}
        count_dict = {}

        # Iterate over each dictionary in the list
        for d in dict_list:
            for key, value in d.items():
                if key not in sum_dict:
                    sum_dict[key] = 0
                    count_dict[key] = 0
                sum_dict[key] += value
                count_dict[key] += 1

        # Calculate the average for each key
        avg_dict = {key: sum_val / count_dict[key] for key, sum_val in sum_dict.items()}

        return avg_dict

    def flat_grads(self, grads: tuple):
        """
        Flatten the gradients into a single tensor.
        """
        flat_grad = torch.cat([g.view(-1) for g in grads])
        return flat_grad
