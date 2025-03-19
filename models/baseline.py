import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
import os

gym.register_envs(ale_py)
env = gym.make("MsPacman-v4", render_mode=None)

def get_state(obs):
    state = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state, (20, 20))
    return tuple(state.flatten()) 

def q_learning(env, num_episodes, alpha, gamma, epsilon):
    q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = get_state(obs)
        action = None
        done = False
        truncated = False
        total_reward = 0

        while not (done or truncated):

            if np.random.uniform() < epsilon:
                action = np.random.choice(env.action_space.n)
            else:
                action = np.argmax(q_table[state])

            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = get_state(next_obs)
            next_action = np.argmax(q_table[next_state])
            total_reward += reward
            q_table[state][action] += alpha * (reward + (gamma * q_table[next_state][next_action]) - q_table[state][action])
            state = next_state

        rewards.append(total_reward)

        if episode % 10 == 0:
            print(f"Episode {episode} - Total Reward: {total_reward}")

    return q_table, rewards


def create_video(q_table, env, filename="pacman_baseline.mp4"):
    video_env = gym.wrappers.RecordVideo(env, './video', episode_trigger=lambda x: True, video_length=0)
    obs, _ = video_env.reset()
    state = get_state(obs)
    action = None
    done = False
    truncated = False
    while not (done or truncated):
        action = np.argmax(q_table[state])
        next_obs, reward, done, truncated, _ = video_env.step(action)
        state = get_state(next_obs)
        
    video_env.close()

    # Rename the video file to the desired filename
    video_dir = './video'
    video_file = [f for f in os.listdir(video_dir) if f.endswith('.mp4')][0]
    os.rename(os.path.join(video_dir, video_file), filename)


num_episodes = 500
alpha = 0.1
gamma = 0.95
epsilon = 0.2

q_table, rewards = q_learning(env, num_episodes, alpha, gamma, epsilon)

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning")
plt.show()

v_env = gym.make("MsPacman-v4", render_mode="rgb_array")
create_video(q_table, v_env)
print(max(rewards))
print(min(rewards))
print(np.mean(rewards))