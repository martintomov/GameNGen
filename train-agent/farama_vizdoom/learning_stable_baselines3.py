#!/usr/bin/env python3

#####################################################################
# Example script of training agents with stable-baselines3
# on ViZDoom using the Gymnasium API (with MPS support for Apple Silicon, added by @martintomov)
#
# Note: For this example to work, you need to install stable-baselines3, opencv, and torch:
#       pip install stable-baselines3 opencv-python torch torchvision torchaudio
#
# See more stable-baselines3 documentation here:
#   https://stable-baselines3.readthedocs.io/en/master/index.html
#####################################################################

from argparse import ArgumentParser

import cv2
import gymnasium
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import pickle

import vizdoom.gymnasium_wrapper  # noqa


DEFAULT_ENV = "VizdoomBasic-v0"
AVAILABLE_ENVS = [env for env in gymnasium.envs.registry.keys() if "Vizdoom" in env]

# Updated image shapes based on the research paper:
TRAINING_IMAGE_SHAPE = (120, 160)  # For model training
SAVED_IMAGE_SHAPE = (240, 320)  # For generative model training

# Training parameters
TRAINING_TIMESTEPS = int(1e6)
N_STEPS = 128
N_ENVS = 8
FRAME_SKIP = 4


class ObservationWrapper(gymnasium.ObservationWrapper):
    """
    ViZDoom environments return dictionaries as observations, containing
    the main image as well as other info.

    This wrapper replaces the dictionary observation space with a simple
    Box space (i.e., only the RGB image), and also resizes the image to the
    smaller size needed for training.

    NOTE: Ensure image resizing matches the setup in the research paper.
    """

    def __init__(self, env, training_shape=TRAINING_IMAGE_SHAPE, save_shape=SAVED_IMAGE_SHAPE):
        super().__init__(env)
        self.training_shape = training_shape
        self.save_shape = save_shape
        self.training_shape_reverse = training_shape[::-1]
        self.save_shape_reverse = save_shape[::-1]
        self.env.frame_skip = FRAME_SKIP

        # Create new observation space for training
        num_channels = env.observation_space["screen"].shape[-1]
        new_shape = (training_shape[0], training_shape[1], num_channels)
        self.observation_space = gymnasium.spaces.Box(0, 255, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        # Resize to training shape for use in the model
        training_observation = cv2.resize(observation["screen"], self.training_shape_reverse)
        # Resize to save shape for data collection
        save_observation = cv2.resize(observation["screen"], self.save_shape_reverse)
        # Save the save_observation image or store it as needed here
        # E.g., saving images to disk can be done as:
        # cv2.imwrite(f"frame_{step}.png", save_observation)

        return training_observation


def main(args):
    # Set the device to MPS if available, otherwise use CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create multiple environments: this speeds up training with PPO
    # We apply two wrappers on the environment:
    #  1) The above wrapper that modifies the observations (resizes for model training and saving)
    #  2) A reward scaling wrapper. Normally, scenarios use large magnitudes for rewards (e.g., 100, -100).
    #     This may lead to unstable learning, so rewards are scaled by 1/100
    def wrap_env(env):
        env = ObservationWrapper(env)
        env = gymnasium.wrappers.TransformReward(env, lambda r: r * 0.01)
        return env

    envs = make_vec_env(args.env, n_envs=N_ENVS, wrapper_class=wrap_env)

    agent = PPO("CnnPolicy", envs, n_steps=N_STEPS, verbose=1, device=device)
    
    # Do the actual learning
    # If agent gets better, "ep_rew_mean" should increase steadily
    agent.learn(total_timesteps=TRAINING_TIMESTEPS)

    # Save collected data after training
    with open("collected_data.pkl", "wb") as f:
        pickle.dump(data, f)

    print("Data collection complete and saved to 'collected_data.pkl'")


if __name__ == "__main__":
    parser = ArgumentParser("Train stable-baselines3 PPO agents on ViZDoom.")
    parser.add_argument(
        "--env",
        default=DEFAULT_ENV,
        choices=AVAILABLE_ENVS,
        help="Name of the environment to play",
    )
    args = parser.parse_args()
    main(args)