import gymnasium as gym
import numpy as np
import ale_py
import os
from tqdm import tqdm

gym.register_envs(ale_py)

def sarsa(env, num_episodes, alpha, gamma, epsilon):
    pass

def record_video(agent_policy, video_folder="videos", video_prefix="pacman_video"):
    # Create the folder if it doesn't exist.
    os.makedirs(video_folder, exist_ok=True)

    # Create a new environment with video recording enabled.
    env = gym.make("MsPacman-v4", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=video_folder, name_prefix=video_prefix)
    state, info = env.reset()
    done = False

    while not done:
        # Use your trained agent's policy here.
        # For demonstration, we use a dummy policy that selects random actions.
        action = agent_policy(state, env.action_space)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()

def egreedy_policy(env, q_table, state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])
