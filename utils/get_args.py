import argparse
import json
from copy import deepcopy

import torch


def override_args(init_args):
    # copy args
    args = deepcopy(init_args)
    env_name, version = args.env_name.split("-")
    file_path = f"config/{env_name}/{args.algo_name}.json"
    current_params = load_hyperparams(file_path=file_path)

    # use pre-defined params if no pram given as args
    for k, v in current_params.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    return args


def load_hyperparams(file_path):
    """Load hyperparameters for a specific environment from a JSON file."""
    try:
        with open(file_path, "r") as f:
            hyperparams = json.load(f)
            return hyperparams  # .get({})
    except FileNotFoundError:
        print(f"No file found at {file_path}. Returning default empty dictionary.")
        return {}


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--project", type=str, default="Exp", help="WandB project classification"
    )
    parser.add_argument(
        "--logdir", type=str, default="log/train_log", help="name of the logging folder"
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Global folder name for experiments with multiple seed tests.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help='Seed-specific folder name in the "group" folder.',
    )
    parser.add_argument(
        "--env-name", type=str, default="fetch-reach", help="Name of the model."
    )
    parser.add_argument("--algo-name", type=str, default="ppo", help="Disable cuda.")
    parser.add_argument("--seed", type=int, default=42, help="Batch size.")
    parser.add_argument(
        "--num-runs", type=int, default=10, help="Number of samples for training."
    )
    parser.add_argument(
        "--num-options", type=int, default=None, help="Number of samples for training."
    )
    parser.add_argument(
        "--extractor-type", type=str, default=None, help="Base learning rate."
    )
    parser.add_argument(
        "--extractor-lr", type=float, default=1e-4, help="Base learning rate."
    )
    parser.add_argument(
        "--igtpo-actor-lr", type=float, default=1e-2, help="Base learning rate."
    )
    parser.add_argument(
        "--actor-lr", type=float, default=1e-4, help="Base learning rate."
    )
    parser.add_argument(
        "--critic-lr", type=float, default=2e-4, help="Base learning rate."
    )
    parser.add_argument(
        "--eps-clip", type=float, default=None, help="Base learning rate."
    )
    parser.add_argument(
        "--actor-fc-dim", type=list, default=[64, 64], help="Base learning rate."
    )
    parser.add_argument(
        "--critic-fc-dim", type=list, default=[64, 64], help="Base learning rate."
    )
    parser.add_argument(
        "--feature-dim", type=int, default=None, help="Base learning rate."
    )
    parser.add_argument(
        "--hl-timesteps", type=int, default=None, help="Number of training epochs."
    )
    parser.add_argument(
        "--timesteps", type=int, default=None, help="Number of training epochs."
    )
    parser.add_argument(
        "--extractor-epochs",
        type=int,
        default=None,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--log-interval", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        "--eval-num", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--marker", type=int, default=2, help="Number of training epochs."
    )
    parser.add_argument("--num-minibatch", type=int, default=None, help="")
    parser.add_argument("--minibatch-size", type=int, default=None, help="")
    parser.add_argument("--batch-size", type=int, default=None, help="")
    parser.add_argument("--num-local-updates", type=int, default=None, help="")
    parser.add_argument("--K-epochs", type=int, default=None, help="")
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="Upper bound of the eigenvalue of the dual metric.",
    )
    parser.add_argument(
        "--gae",
        type=float,
        default=0.95,
        help="Lower bound of the eigenvalue of the dual metric.",
    )
    parser.add_argument(
        "--entropy-scaler", type=float, default=1e-3, help="Base learning rate."
    )
    parser.add_argument("--gamma", type=float, default=None, help="Base learning rate.")
    parser.add_argument(
        "--load-pretrained-model",
        action="store_true",
        help="Path to a directory for storing the log.",
    )
    parser.add_argument(
        "--rendering",
        action="store_true",
        help="Path to a directory for storing the log.",
    )

    parser.add_argument(
        "--gpu-idx", type=int, default=0, help="Number of training epochs."
    )

    args = parser.parse_args()
    args.device = select_device(args.gpu_idx)

    return args


def select_device(gpu_idx=0, verbose=True):
    if verbose:
        print(
            "============================================================================================"
        )
        # set device to cpu or cuda
        device = torch.device("cpu")
        if torch.cuda.is_available() and gpu_idx is not None:
            device = torch.device("cuda:" + str(gpu_idx))
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(device)))
        else:
            print("Device set to : cpu")
        print(
            "============================================================================================"
        )
    else:
        device = torch.device("cpu")
        if torch.cuda.is_available() and gpu_idx is not None:
            device = torch.device("cuda:" + str(gpu_idx))
            torch.cuda.empty_cache()
    return device
