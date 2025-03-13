import gymnasium as gym
import numpy as np
import ale_py
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# Register Atari environments.
gym.register_envs(ale_py)
env = gym.make("MsPacman-v4", render_mode=None)

######################################################
# Coordinate Extraction and Discretization Functions
######################################################

def extract_pacman_coords(state):
    """
    Extracts Pacman's (x,y) coordinates from the game image.
    This is a placeholder function. In a robust implementation, you
    would use image processing to detect Pacman (for example, by color).
    Here, we return the center of the image.
    """
    state_bgr = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(state_bgr, cv2.COLOR_BGR2HSV)
    
    # Define HSV range for Pacman's color.
    # For example, if Pacman is yellowish:
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # Create a mask for the yellow color.
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Apply morphological operations to remove noise.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Assume the largest contour corresponds to Pacman.
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    
    # If no contours are found, default to the center of the image.
    height, width, _ = state.shape
    return (width // 2, height // 2)

def discretize_coords(x, y, num_bins_x, num_bins_y, width, height):
    """
    Discretizes the (x, y) coordinates by dividing the screen into a grid.
    
    Args:
        x, y (int): Continuous coordinates.
        num_bins_x, num_bins_y (int): Number of bins along x and y.
        width, height (int): Dimensions of the screen.
    
    Returns:
        tuple: The discrete state as (x_bin, y_bin).
    """
    x_bin = int(x / (width / num_bins_x))
    y_bin = int(y / (height / num_bins_y))
    # Ensure bins are within range.
    x_bin = min(x_bin, num_bins_x - 1)
    y_bin = min(y_bin, num_bins_y - 1)
    return (x_bin, y_bin)

def is_in_corner(discrete_state, num_bins_x, num_bins_y):
    x_bin, y_bin = discrete_state
    return (x_bin == 0 or x_bin == num_bins_x - 1) and (y_bin == 0 or y_bin == num_bins_y - 1)


######################################################
# Reward Shaping Function
######################################################

def compute_shaped_reward(original_reward, discrete_state, pellet_status, bonus=10.0, penalty=-1.0, stuck_penalty=-5.0, prev_state=None):
    """
    Computes a shaped reward that adds:
      - A small time penalty (if needed, here omitted for simplicity),
      - A bonus if the discrete state (e.g. where pellets are available) is visited for the first time,
      - A penalty if it is revisited,
      - An extra penalty if the agent remains in the same state (stuck).
    
    Args:
        original_reward (float): Reward from the environment.
        discrete_state (tuple): The current discrete state (e.g., (x_bin, y_bin)).
        pellet_status (dict): Tracks if this discrete state has been visited.
        bonus (float): Bonus for first visit.
        penalty (float): Penalty for revisiting.
        stuck_penalty (float): Extra penalty if the state doesn't change.
        prev_state (tuple): The previous discrete state.
    
    Returns:
        float: The shaped reward.
    """
    # If a power pellet is collected, treat it as a bonus event.
    if original_reward >= 20:
        return original_reward + bonus

    reward = original_reward
    # Check if the agent is stuck (state hasn't changed).
    if prev_state is not None and discrete_state == prev_state:
        reward += stuck_penalty
    else:
        # If the state is new, add bonus and mark it as visited.
        if discrete_state not in pellet_status:
            pellet_status[discrete_state] = True
            reward += bonus
        else:
            # If already visited, you might choose no extra penalty, or a mild one.
            reward += penalty
    return reward

######################################################
# SARSA Training with Coordinate Discretization, Reward Shaping, and Epsilon Decay
######################################################

def sarsa_with_coords(env, num_episodes, alpha, gamma, initial_epsilon, min_epsilon, decay_rate, num_bins_x, num_bins_y, bonus, penalty, optimistic_value=10.0):
    """
    SARSA training loop using coordinate-based discretization.
    The Q-table is stored in a dictionary mapping discrete states (tuples) to Q-value vectors.
    
    Args:
        env: Gym environment.
        num_episodes (int): Number of training episodes.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        initial_epsilon (float): Starting exploration rate.
        min_epsilon (float): Minimum exploration rate.
        decay_rate (float): Multiplicative decay per episode.
        num_bins_x (int): Number of bins along the x-dimension.
        num_bins_y (int): Number of bins along the y-dimension.
        bonus (float): Bonus for first visit.
        penalty (float): Penalty for revisiting.
        optimistic_value (float): Initial Q-value.
    
    Returns:
        rewards (list): Episode returns.
        Q (dict): Learned Q-table.
    """
    n_actions = env.action_space.n
    Q = {}  # Q will be a dict mapping discrete_state to an array of Q-values.
    rewards = []
    epsilon = initial_epsilon

    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        state, _ = env.reset()
        height, width, _ = state.shape
        x, y = extract_pacman_coords(state)
        discrete_state = discretize_coords(x, y, num_bins_x, num_bins_y, width, height)
        if discrete_state not in Q:
            Q[discrete_state] = np.ones(n_actions) * optimistic_value
        pellet_status = {}  # Reset pellet status for this episode.
        prev_state = None
        stuck_counter = 0

        done = False
        ep_reward = 0

        if is_in_corner(discrete_state, num_bins_x, num_bins_y):
            # Force exploration if in a corner.
            action = env.action_space.sample()
            
        # Epsilon-greedy action selection.
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(Q[discrete_state]))
        
        
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            x_next, y_next = extract_pacman_coords(next_state)
            next_discrete_state = discretize_coords(x_next, y_next, num_bins_x, num_bins_y, width, height)
            if next_discrete_state not in Q:
                Q[next_discrete_state] = np.ones(n_actions) * optimistic_value
            
            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = int(np.argmax(Q[next_discrete_state]))
            
            if prev_state is not None and discrete_state == prev_state:
                stuck_counter += 1
            else:
                stuck_counter = 0

            # If stuck for too many steps, reset epsilon to encourage exploration.
            if stuck_counter >= 5:
                epsilon = initial_epsilon
                stuck_counter = 0
        
            # Compute shaped reward.
            shaped_reward = compute_shaped_reward(reward, discrete_state, pellet_status, bonus=bonus, penalty=penalty, prev_state=prev_state)
            
            # SARSA update.
            td_target = shaped_reward + gamma * Q[next_discrete_state][next_action]
            td_error = td_target - Q[discrete_state][action]
            Q[discrete_state][action] += alpha * td_error

            prev_state = discrete_state
            discrete_state = next_discrete_state
            action = next_action
            ep_reward += reward  # Tracking the original reward.
        
        rewards.append(ep_reward)
        epsilon = max(min_epsilon, epsilon * decay_rate)
    
    return rewards, Q

######################################################
# Video Recording Function using Coordinate Discretization
######################################################

def record_video_coords(Q, num_bins_x, num_bins_y, video_folder="videos", video_prefix="sarsa_coords_video"):
    os.makedirs(video_folder, exist_ok=True)
    env_video = gym.make("MsPacman-v4", render_mode="rgb_array")
    env_video = gym.wrappers.RecordVideo(env_video, video_folder=video_folder, name_prefix=video_prefix)
    state, _ = env_video.reset()
    height, width, _ = state.shape
    done = False
    while not done:
        x, y = extract_pacman_coords(state)
        discrete_state = discretize_coords(x, y, num_bins_x, num_bins_y, width, height)
        if discrete_state not in Q:
            Q[discrete_state] = np.ones(env_video.action_space.n) * 10.0
        action = int(np.argmax(Q[discrete_state]))
        state, reward, terminated, truncated, _ = env_video.step(action)
        done = terminated or truncated
        _ = env_video.render()
    env_video.close()
    print(f"Video saved in {video_folder} with prefix {video_prefix}")

######################################################
# Main Hyperparameter Testing Loop
######################################################

if __name__ == "__main__":
    # Hyperparameters for coordinate discretization.
    num_episodes = 1000
    hyperparams = [
        (0.1, 0.95),
        (0.1, 0.90),
        (0.1, 0.85),
        (0.1, 0.8)
    ]
    initial_epsilon = 1.0
    min_epsilon = 0.1
    decay_rate = 0.995
    num_bins_x = 16
    num_bins_y = 21
    bonus = 10.0
    penalty = -2.0

    reward_curves = {}

    for alpha, gamma in hyperparams:
        config_str = f"alpha{alpha}_gamma{gamma}"
        print(f"\nTraining with hyperparameters: {config_str}")
        rewards, Q = sarsa_with_coords(env, num_episodes, alpha, gamma, initial_epsilon, min_epsilon, decay_rate, num_bins_x, num_bins_y, bonus, penalty, optimistic_value=10.0)
        reward_curves[config_str] = rewards

        video_prefix = f"sarsa_coords_{config_str}"
        print("Recording video for configuration", config_str)
        record_video_coords(Q, num_bins_x, num_bins_y, video_folder="videos", video_prefix=video_prefix)
    
    plt.figure(figsize=(12, 6))
    for config_str, rewards in tqdm(reward_curves.items(), total=len(reward_curves), desc="Plotting reward curves"):
        plt.plot(rewards, label=config_str)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode for Different Alpha/Gamma Configurations (XY Coordinates)")
    plt.legend()
    plt.grid(True)
    plt.savefig("reward_plot_coords.png")
