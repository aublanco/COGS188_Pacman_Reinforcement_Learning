import gymnasium as gym
import numpy as np
import ale_py
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from featureExtractor import extract_pacman_coords, closest_food_distance

# Register Atari environments.
gym.register_envs(ale_py)
env = gym.make("MsPacman-v4", render_mode=None)



def discretize_coords(x, y, num_bins_x, num_bins_y, width, height):
    """
    Discretizes (x,y) by dividing the image into a grid.
    """
    x_bin = int(x / (width / num_bins_x))
    y_bin = int(y / (height / num_bins_y))
    return (min(x_bin, num_bins_x - 1), min(y_bin, num_bins_y - 1))

def restrict_action(action):
    """
    Restricts the full 9-action space to a reduced 5-action space.
    All other actions are mapped to NOOP.
    """
    mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 0, 6: 0, 7: 0, 8: 0}
    return mapping.get(action, 0)

def compute_bonus_reward(original_reward, prev_distance, current_distance, bonus_factor=1.0):
    """
    Computes the reward by adding a bonus proportional to how much closer Pacman moved.
    """
    bonus = bonus_factor * max(0, prev_distance - current_distance)
    return original_reward + bonus

def sarsa_pacman(num_episodes=50, alpha=0.1, gamma=0.95, epsilon=1.0,
                 min_epsilon=0.1, decay_rate=0.995, num_bins_x=10, num_bins_y=10,
                 bonus_factor=1.0):

    n_actions = env.action_space.n  
    Q = {}  
    rewards_per_episode = []
    
    def get_Q(state, action):
        return Q.get((state, action), 0.0)
    
    def set_Q(state, action, value):
        Q[(state, action)] = value
    
    for ep in tqdm(range(num_episodes), desc="Episodes"):
        state, _ = env.reset()
        height, width, _ = state.shape
        x, y = extract_pacman_coords(state)
        state_disc = discretize_coords(x, y, num_bins_x, num_bins_y, width, height)
        
        # Epsilon-greedy initial action.
        if np.random.rand() < epsilon:
            raw_action = env.action_space.sample()
            action = restrict_action(raw_action)
        else:
            q_vals = [get_Q(state_disc, a) for a in range(n_actions)]
            action = restrict_action(int(np.argmax(q_vals)))
            
        ep_reward = 0
        prev_distance, _ = closest_food_distance(state, (x, y))
        done = False
        
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            x_next, y_next = extract_pacman_coords(next_state)
            next_state_disc = discretize_coords(x_next, y_next, num_bins_x, num_bins_y, width, height)
            
            # Epsilon-greedy for next action.
            if np.random.rand() < epsilon:
                raw_next_action = env.action_space.sample()
                next_action = restrict_action(raw_next_action)
            else:
                q_vals_next = [get_Q(next_state_disc, a) for a in range(n_actions)]
                next_action = restrict_action(int(np.argmax(q_vals_next)))
            
            current_distance, _ = closest_food_distance(next_state, (x_next, y_next))
            # Apply bonus reward if Pacman moved closer to the pellet.
            shaped_reward = compute_bonus_reward(reward, prev_distance, current_distance, bonus_factor)
            
            # SARSA TD update.
            td_target = shaped_reward + gamma * get_Q(next_state_disc, next_action)
            td_error = td_target - get_Q(state_disc, action)
            new_q = get_Q(state_disc, action) + alpha * td_error
            set_Q(state_disc, action, new_q)
            
            ep_reward += reward
            state_disc = next_state_disc
            action = next_action
            state = next_state
            prev_distance = current_distance
        
        rewards_per_episode.append(ep_reward)
        epsilon = max(min_epsilon, epsilon * decay_rate)
    
    return Q, rewards_per_episode

def record_video(Q, num_bins_x, num_bins_y, video_folder="videos", video_prefix="sarsa_video"):
    """
    Records a video using the learned Q-table.
    The policy is greedy with respect to Q (after applying the restrict_action mapping).
    """
    os.makedirs(video_folder, exist_ok=True)
    env_video = gym.make("MsPacman-v4", render_mode="rgb_array")
    env_video = gym.wrappers.RecordVideo(env_video, video_folder=video_folder, name_prefix=video_prefix)
    state, _ = env_video.reset()
    height, width, _ = state.shape
    x, y = extract_pacman_coords(state)
    state_disc = discretize_coords(x, y, num_bins_x, num_bins_y, width, height)
    done = False
    while not done:
        if state_disc not in Q:
            raw_action = env_video.action_space.sample()
            action = restrict_action(raw_action)
        else:
            q_vals = [Q.get((state_disc, a), 0.0) for a in range(env_video.action_space.n)]
            action = restrict_action(int(np.argmax(q_vals)))
        state, reward, terminated, truncated, _ = env_video.step(action)
        done = terminated or truncated
        _ = env_video.render()
        x, y = extract_pacman_coords(state)
        state_disc = discretize_coords(x, y, num_bins_x, num_bins_y, width, height)
    env_video.close()
    print(f"Video saved in {video_folder} with prefix {video_prefix}")

if __name__ == "__main__":
    # Hyperparameter sweep: try different (alpha, gamma) pairs.
    num_episodes = 500
    initial_epsilon = 1.0
    min_epsilon = 0.1
    decay_rate = 0.995
    num_bins_x = 16   
    num_bins_y = 25  
    bonus_factor = 1.0

    hyperparams = [(0.1, 0.85), (0.1, 0.8), (0.2, 0.85), (0.2, 0.8)]
    reward_curves = {}
    average_rewards = {}

    for alpha, gamma in hyperparams:
        config_str = f"alpha{alpha}_gamma{gamma}"
        print(f"\nTraining with hyperparameters: {config_str}")
        Q, rewards = sarsa_pacman(num_episodes, alpha, gamma, initial_epsilon, min_epsilon,
                                  decay_rate, num_bins_x, num_bins_y, bonus_factor)
        
        reward_curves[config_str] = rewards
        avg_reward = np.mean(rewards)
        average_rewards[config_str] = avg_reward
        print(f"Average reward for {config_str}: {avg_reward:.2f}")

        video_prefix = f"{config_str}_video"
        print("Recording video for configuration", config_str)
        record_video(Q, num_bins_x, num_bins_y, video_folder="videos", video_prefix=video_prefix)
    
    # Plot reward curves.
    plt.figure(figsize=(12, 6))
    for config_str, rewards in reward_curves.items():
        plt.plot(rewards, label=config_str)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("SARSA Training Rewards for different (alpha, gamma) configurations")
    plt.legend()
    plt.grid(True)
    plt.savefig("reward_plot_6.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    configs = list(average_rewards.keys())
    avg_vals = [average_rewards[c] for c in configs]
    plt.bar(configs, avg_vals)
    plt.xlabel("Hyperparameter Configuration")
    plt.ylabel("Average Reward")
    plt.title("Average Reward Across Episodes")
    plt.savefig("average_reward_plot_4.png")
    plt.show()