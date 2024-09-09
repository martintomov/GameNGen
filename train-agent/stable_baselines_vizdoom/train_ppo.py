import gymnasium as gym
import torch

from stable_baselines3 import PPO
import gymnasium
from loguru import logger

from vizdoom import gymnasium_wrapper  # noqa

# Check if MPS, CUDA, or CPU is available and set the device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("Using CUDA for training.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("Using MPS for training.")
else:
    device = torch.device("cpu")
    logger.info("Neither CUDA nor MPS available, using CPU for training.")

env = gymnasium.make("VizdoomCorridor-v0", render_mode="rgb_array")

logger.info('Start training')
model = PPO("MultiInputPolicy", env, verbose=1, device=device)
model.learn(total_timesteps=1000000, progress_bar=True)

model.save("trained_agents/ppo_trained_agent")

# VecEnv resets automatically
# if done:
#   obs = vec_env.reset()
