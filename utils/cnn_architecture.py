import torch
import torch.nn as nn


def get_cnn_architecture(args):
    width, height, channel = args.state_dim
    env_name, version = args.env_name.split("-")
    version = int(version[-1]) if version[-1].isdigit() else version[-1]

    if env_name == "fourrooms":
        encoder_architecture = [
            nn.Conv2d(channel, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        ]

        decoder_architecture = [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, channel, kernel_size=3, stride=1, padding=1),
        ]
    elif env_name in ("maze", "ninerooms"):
        encoder_architecture = [
            nn.Conv2d(channel, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        ]

        decoder_architecture = [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(128, 64, kernel_size=2, stride=1, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, channel, kernel_size=3, stride=1, padding=1),
        ]
    else:
        raise NotImplementedError(f"Environment {env_name} is implemented on CNN.")

    return encoder_architecture, decoder_architecture
