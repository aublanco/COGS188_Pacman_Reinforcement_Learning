import gymnasium as gym
import numpy as np
import ale_py
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.tile_code import TileCoder, discretize_state, extract_features
from utils.record_video import record_video
from utils.reward_shaping import compute_shaped_reward

gym.register_envs(ale_py)
env = gym.make("MsPacman-v4", render_mode=None)

####################################################################################

def sarsa(env, num_episodes, alpha, gamma, initial_epsilon, min_epsilon, decay_rate, tile_coder, optimistic_value, bonus, penalty):
    n_actions = env.action_space.n
    q_table = np.ones((tile_coder.num_tiles(), n_actions)) * optimistic_value
    rewards = []
    epsilon = initial_epsilon

    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        features = extract_features(state)
        tile_indices = discretize_state(features, tile_coder)
        discrete_state = tuple(tile_indices)
        pellet_status = {}
        done = False
        ep_reward = 0

        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_vals = np.mean([q_table[idx] for idx in tile_indices], axis=0)
            action = int(np.argmax(q_vals))

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_features = extract_features(next_state)
            next_tile_indices = discretize_state(next_features, tile_coder)
            next_discrete_state = tuple(next_tile_indices)

            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_q_vals = np.mean([q_table[idx] for idx in next_tile_indices], axis=0)
                next_action = int(np.argmax(next_q_vals))
            
            shaped_reward = compute_shaped_reward(reward, discrete_state, pellet_status, bonus=bonus, penalty=penalty)

            current_q_vals = np.mean([q_table[idx] for idx in tile_indices], axis=0)
            td_target = shaped_reward + gamma * np.mean([q_table[idx][next_action] for idx in next_tile_indices])
            td_error = td_target - current_q_vals[action]

            for idx in tile_indices:
                q_table[idx, action] += alpha * td_error
            
            tile_indices = next_tile_indices
            discrete_state = next_discrete_state
            action = next_action
            ep_reward += reward
        
        rewards.append(ep_reward)

        epsilon = max(min_epsilon, epsilon * decay_rate)

    return rewards, q_table

####################################################################################

if __name__ == "__main__":
    # Create a tile coder.
    # For MsPacman-v4, grayscale mean is roughly in [0, 255] and std is roughly in [0, 128].
    tile_coder = TileCoder(n_tilings=8, n_bins=(10, 10), low=np.array([0, 0]), high=np.array([255, 128]))
    
    num_episodes = 500
    # Define hyperparameter combinations as tuples: (alpha, gamma)
    hyperparams = [
        (0.1, 0.99),
        (0.1, 0.95),
        (0.2, 0.99),
        (0.2, 0.95)
    ]
    # Epsilon decay parameters.
    initial_epsilon = 1.0
    min_epsilon = 0.1
    decay_rate = 0.995

    # Dictionary to hold reward curves.
    reward_curves = {}
    
    for alpha, gamma in hyperparams:
        config_str = f"alpha {alpha} gamma {gamma}"
        print(f"\nTraining with hyperparameters: {config_str}")
        rewards, q_table = sarsa(env, num_episodes, alpha, gamma, initial_epsilon, min_epsilon, decay_rate, tile_coder, optistic_value=10, bonus=10.0, penalty=-1.0)
        reward_curves[config_str] = rewards
        
        video_prefix = f"sarsa_tile_{config_str}"
        print("Recording video for configuration", config_str)
        record_video(q_table, tile_coder, video_folder="videos", video_prefix=video_prefix)
    
    # Plot reward curves for all hyperparameter configurations.
    plt.figure(figsize=(12, 6))
    for config_str, rewards in tqdm(reward_curves.items(), total=len(reward_curves), desc="Plotting reward curves"):
        plt.plot(rewards, label=config_str)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode for Different Configurations")
    plt.legend()
    plt.grid(True)
    plt.savefig("reward_plot.png")
    plt.show()