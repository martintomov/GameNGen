#!/usr/bin/env python3

#####################################################################
# Example for running a vizdoom scenario as a gym env
#####################################################################

import gymnasium

from vizdoom import gymnasium_wrapper  # noqa


if __name__ == "__main__":
    env = gymnasium.make("VizdoomHealthGatheringSupreme-v0", render_mode="rgb_array")

    # This is the way to configure the environment
    env.get_wrapper_attr('game').set_render_hud(True)

    # Rendering random rollouts for ten episodes
    for _ in range(10):
        done = False
        obs, info = env.reset(seed=42)
        while not done:
            obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
    
    print('Done')
