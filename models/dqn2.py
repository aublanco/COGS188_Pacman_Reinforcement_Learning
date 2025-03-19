import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
import cv2
import ale_py
import gymnasium as gym

from dataclasses import dataclass

@dataclass
class HyperParams:
    BATCH_SIZE: int = 512
    GAMMA: float = 0.99
    EPS_START: float = 0.9
    EPS_END: float = 0.05
    EPS_DECAY: int = 1000
    TAU: float = 0.005
    LR: float = 1e-4

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))

is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

gym.register_envs(ale_py)

class PacManWrapper(gym.Wrapper):
    def __init__(self, env):
        super(PacManWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84,84,1), dtype=np.uint8)

    def reset(self, **kwargs):
        observation, info = self.env.reset()
        return self.preprocess(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.preprocess(observation), reward, terminated, truncated, info

    def preprocess(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84,84), interpolation=cv2.INTER_AREA)
        processed_obs = np.expand_dims(resized, axis=-1)
        return processed_obs


env = gym.make("MsPacman-v4")
env = PacManWrapper(env)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_observations[-1], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=3, stride=2),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64*4*4 ,512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        return self.fc(x)

class DQNTrainer:
    def __init__(self, env, memory, device, params, max_steps_per_episode, num_episodes):
        self.env = env
        self.policy_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
        self.target_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=params.LR, amsgrad=True)
        self.memory = memory
        self.device = device
        self.params = params
        self.max_steps_per_episode = max_steps_per_episode
        self.num_episodes = num_episodes
        self.episode_rewards = []
        self.steps_done = 0

    def select_action(self, state_tensor):
        sample = random.random()
        eps_threshold = (self.params.EPS_END + (self.params.EPS_START - self.params.EPS_END) 
                         * math.exp(-1 * self.steps_done / self.params.EPS_DECAY))

        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state_tensor).max(1)[1].view(1,1)
        else:
            return torch.tensor([[random.randrange(self.env.action_space.n)]], 
            device=self.device,
            dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.params.BATCH_SIZE:
            return 
        
        transitions = self.memory.sample(self.params.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda x: x is not None, batch.next_state)), 
                                                device=self.device, dtype=torch.bool)
        non_final_next_state = torch.cat([x for x in batch.next_state if x is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float, device=self.device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_value = torch.zeros(self.params.BATCH_SIZE, device=self.device)

        with torch.no_grad():
            next_state_value[non_final_mask] = self.target_net(non_final_next_state).max(1)[0]
        
        expected_state_action_values = reward_batch + (self.params.GAMMA * next_state_value)

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def soft_update(self):
        with torch.no_grad():
            target_net_dict = self.target_net.state_dict()
            policy_net_dict = self.policy_net.state_dict()

            for key in policy_net_dict:
                target_net_dict[key] = (self.params.TAU * policy_net_dict[key] + 
                                        (1 - self.params.TAU) * target_net_dict[key])

            self.target_net.load_state_dict(target_net_dict)

    def plot_rewards(self, show_result = False):
        plt.figure(1)
        rewards_t = torch.tensor(self.episode_rewards, dtype=torch.float)

        if show_result:
            plt.title("Result")
        else:
            plt.clf()
            plt.title("Training Reward")

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.plot(rewards_t.numpy(), label="Episode Reward")
        plt.pause(0.001)
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def train(self):
        for _ in tqdm(range(self.num_episodes)):
            obs, info = self.env.reset()
            obs = np.transpose(obs, (2, 0, 1))
            state = torch.tensor(obs, dtype=torch.float32, device = self.device).unsqueeze(0)
            episode_reward = 0.0

            for _ in range(self.max_steps_per_episode):
                action = self.select_action(state)
                obs, reward, term, trunc, _ = self.env.step(action.item())
                obs = np.transpose(obs, (2, 0, 1))
                done = term or trunc
                
                if term:
                    next_state = None
                else:
                    next_state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward, done)
                state = next_state
                self.optimize_model()
                self.soft_update()
                episode_reward += reward
                if done:
                    break

            self.episode_rewards.append(episode_reward)
            self.plot_rewards()

        print("Training Complete")
        self.plot_rewards(show_result=True)
        plt.ioff()
        plt.show()
        plt.savefig("rewards_plot_dqn.png")

from gymnasium.wrappers import RecordVideo

def record_video(env, policy_net, device, video_folder="videos", episode_trigger=lambda ep: True):
    video_env = RecordVideo(env, video_folder=video_folder, episode_trigger=episode_trigger)
    obs, info = video_env.reset()
    done = False

    while not done:
        # Preprocess observation: convert (84,84,1) to (1,84,84)
        obs_proc = np.transpose(obs, (2, 0, 1))
        state = torch.tensor(obs_proc, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            # Get the action from the policy network.
            action = policy_net(state).max(1)[1].view(1, 1).item()
        obs, reward, terminated, truncated, info = video_env.step(action)
        done = terminated or truncated

    video_env.close()
    print(f"Video saved in folder: {video_folder}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("MsPacman-v4", render_mode="rgb_array")
    env = PacManWrapper(env)
    params = HyperParams()
    memory = ReplayMemory(10000)
    trainer = DQNTrainer(env, memory, device, params, max_steps_per_episode=1000, num_episodes=1000)
    trainer.train()
    record_video(env, trainer.policy_net, device)