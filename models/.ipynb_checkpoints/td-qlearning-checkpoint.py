import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
import os
from featureExtractor import crop_gameplay_area

gym.register_envs(ale_py)
env = gym.make("MsPacman-v4", render_mode=None)

def get_position(contours):
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        moment = cv2.moments(largest_contour)
        if moment["m00"] == 0:
            return (0, 0)
        position = (int(moment["m10"] / moment["m00"]), int(moment["m01"] / moment["m00"]))
        return position
    else:
        return (0, 0)

def extract_features(obs):
    obs = crop_gameplay_area(obs, 173)
    hsv = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_red = np.array([0, 160, 170])
    upper_red = np.array([10, 195, 205])
    lower_pink = np.array([140, 100, 100])  
    upper_pink = np.array([170, 255, 255])
    lower_cyan = np.array([80, 100, 100])   
    upper_cyan = np.array([100, 255, 255])
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([15, 255, 255])
    lower_coral = np.array([0, 125, 220])
    upper_coral = np.array([0, 135, 230])
    
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
    cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    coral_mask = cv2.inRange(hsv, lower_coral, upper_coral)
    
    pacman_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pink_contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cyan_contours, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coral_contours, _ = cv2.findContours(coral_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pacman_pos = get_position(pacman_contours)
    ghosts = [get_position(red_contours), get_position(pink_contours), get_position(cyan_contours), get_position(orange_contours)]
    pellets = []

    for contour in coral_contours:
        moment = cv2.moments(contour)
        if moment["m00"] == 0: 
            continue
        x_coord = int(moment["m10"] / moment["m00"])
        y_coord = int(moment["m01"] / moment["m00"])
        if cv2.contourArea(contour) < 5:
            pellets.append((x_coord, y_coord))

    closest_pellet = (0, 0)
    min_distance = float("inf")
    for pellet in pellets:
        distance = np.linalg.norm(np.array(pacman_pos) - np.array(pellet))
        if distance < min_distance:
            min_distance = distance
            closest_pellet = pellet
            
    closest_ghost = (0, 0)
    min_distance = float("inf")
    for ghost in ghosts:
        distance = np.linalg.norm(np.array(pacman_pos) - np.array(ghost))
        if distance < min_distance:
            min_distance = distance
            closest_ghost = ghost
            
    return pacman_pos, closest_pellet, closest_ghost

def q_learning(env, num_episodes, alpha, gamma, initial_epsilon, min_epsilon, decay_rate):
    q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    rewards = []
    epsilon = initial_epsilon

    for episode in range(num_episodes):
        obs, _ = env.reset()
        pacman_pos, closest_pellet, closest_ghost = extract_features(obs)
        state = (pacman_pos, closest_pellet, closest_ghost)
        action = None
        done = False
        truncated = False
        total_reward = 0
        prev_pos = pacman_pos
        prev_distance = np.linalg.norm(np.array(pacman_pos) - np.array(closest_ghost))

        while not (done or truncated):
            if np.random.uniform() < epsilon:
                action = np.random.choice(env.action_space.n)
            else:
                action = np.argmax(q_table[state])

            next_obs, reward, done, truncated, _ = env.step(action)
            pacman_pos, closest_pellet, closest_ghost = extract_features(next_obs)
            next_state = (pacman_pos, closest_pellet, closest_ghost)
            
            if reward == 10: # 10 is the default for eating a pellet
                reward += 5

            if pacman_pos == prev_pos:
                reward -= 5
                
            curr_distance = np.linalg.norm(np.array(pacman_pos) - np.array(closest_ghost))
            if curr_distance > prev_distance:
                reward += 3
                
            prev_distance = curr_distance
            total_reward += reward
            q_table[state][action] += alpha * (reward + (gamma * max(q_table[next_state])) - q_table[state][action])
            state = next_state

        rewards.append(total_reward)
        epsilon = max(min_epsilon, epsilon * decay_rate)
        if episode % 10 == 0:
            print(f"Episode {episode} - Total Reward: {total_reward}")

    return q_table, rewards

def create_video(q_table, env, filename="pacman_q_learning.mp4"):
    """Create a video of the agent's performance."""
    video_env = gym.wrappers.RecordVideo(env, './video', episode_trigger=lambda x: True, video_length=0)
    obs, _ = video_env.reset()
    pacman_pos, closest_pellet, closest_ghost = extract_features(obs)
    state = (pacman_pos, closest_pellet, closest_ghost)
    action = None
    done = False
    truncated = False
    while not (done or truncated):
        action = np.argmax(q_table[state])
        next_obs, reward, done, truncated, _ = video_env.step(action)
        pacman_pos, closest_pellet, closest_ghost = extract_features(next_obs)
        state = (pacman_pos, closest_pellet, closest_ghost)
        
    video_env.close()

    # Rename the video file to the desired filename
    video_dir = './video'
    video_file = [f for f in os.listdir(video_dir) if f.endswith('.mp4')][0]
    os.rename(os.path.join(video_dir, video_file), filename)
    

num_episodes = 1000
alpha = 0.1
gamma = 0.9
initial_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.995

q_table, rewards = q_learning(env, num_episodes, alpha, gamma, initial_epsilon, min_epsilon, decay_rate)

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning")
plt.show()
print(np.mean(rewards))

v_env = gym.make("MsPacman-v4", render_mode="rgb_array")
create_video(q_table, v_env)
