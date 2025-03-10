import gymnasium as gym
import numpy as np
import ale_py
import os
from tqdm import tqdm

def train_agent(num_episodes=1000):
    # For training, you may want to disable rendering to speed up training.
    # You can also use render_mode="rgb_array" or "human" if you want to observe training occasionally.
    gym.register_envs(ale_py)
    env = gym.make("MsPacman-v4", render_mode="rgb_array")
    env = env.unwrapped
    total_rewards = []

    for episode in tqdm(range(num_episodes)):
        state, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Replace this with your agent's action selection logic.
            action = env.action_space.sample()  # dummy agent: random action

            # Step the environment.
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # Here, insert your agent's update logic:
            # e.g., agent.store_transition(state, action, reward, next_state, done)
            # and periodically call agent.learn() to update your model.

            state = next_state
            done = terminated or truncated

        total_rewards.append(episode_reward)

    env.close()

def record_video(agent_policy, video_folder="videos", video_prefix="pacman_video"):
    # Create the folder if it doesn't exist.
    os.makedirs(video_folder, exist_ok=True)

    # Create a new environment with video recording enabled.
    env = gym.make("MsPacmanDeterministic-v4", render_mode="rgb_array")
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

def random_policy(state, action_space):
    # Dummy policy: choose a random action.
    return action_space.sample()

if __name__ == "__main__":
    print("Training agent...")
    train_agent(num_episodes=1000)
    
    print("Recording video of agent performance...")
    # Replace random_policy with your trained agent's policy function.
    record_video(random_policy)