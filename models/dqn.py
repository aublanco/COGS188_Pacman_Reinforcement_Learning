import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import ale_py
from tqdm import tqdm

from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformObservation

gym.register_envs(ale_py)
env = gym.make('MsPacman-v4', render_mode='rgb_array') #(210, 160, 3) 210x160 RGB image

# add enviornment wrapper to resize the image to 84x84 and convert to grayscale
env = AtariPreprocessing(env, frame_skip=1, screen_size=84, terminal_on_life_loss=True) # (84, 84)
env = FrameStackObservation(env, stack_size=4) # (4, 84, 84) 4 frames stacked
env = TransformObservation(env, lambda obs: obs / 255.0, env.observation_space) 
# normalize the image

# s, _ = env.reset()
# print(s.shape) # (4, 84, 84)
# print(s.dtype)  # float64 (due to TransformObservation)
# print(np.max(s))


# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Define the Q-network

class QNetwork(nn.Module):
    """
    Neural Network for approximating Q-values
    """
    
    def __init__(self, action_size, seed): #TODO
        """
        initialize parameters and build the network

            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        
        
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4) # 4 input channels, 32 output channels, kernel size 8, stride 4
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) # 32 input channels, 64 output channels, kernel size 4, stride 2
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1) # 64 input channels, 64 output channels, kernel size 3, stride 1
        
        self.fc1 = nn.Linear(64*7*7, 256) # 64 input channels, 256 output channels
        self.fc2 = nn.Linear(256, action_size) # 256 input channels, (action_size) output channels
        
        
    def forward(self, state):
        """
        Forward pass of the network
        
            state (torch.Tensor): Input tensor
            
        Returns
        =======
            Q-values (torch.Tensor): The predicted Q-values
        """
        
        state = self.conv1(state)
        state = F.relu(state)
        
        state = self.conv2(state)
        state = F.relu(state)
        
        state = self.conv3(state)
        state = F.relu(state)
        
        state = state.view(state.size(0), -1)
        
        state = self.fc1(state)
        state = F.relu(state)
        state = self.fc2(state)
        
        return state
        
class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples
    """
    def __init__(self, buffer_size, batch_size, action_size, seed=0):
        """
        Initialize a ReplayBuffer object
        
            action_size (int): Dimension of each action
            buffer_size (int): Maximum size of buffer
            batch_size (int): Size of each training batch
            seed (int): Random seed
        """
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memoty"""
        
        state = np.array(state).astype(np.float32)
        next_state = np.array(next_state).astype(np.float32)
        
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        
        # experiences = random.sample(self.memory, k=self.batch_size)
        # states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        # actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        # next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        # dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        experiences = random.sample(self.memory, k=self.batch_size)
        # Use np.stack instead of np.vstack
        states = torch.from_numpy(np.stack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)


class DQNAgent:
    """Interacts with and learns from the environment"""
    def __init__(self, action_size, seed=0,  gamma=0.99, tau=1e-3): # update_every=4
        
        self.action_size = action_size
        self.seed = seed
        self.gamma = gamma
        self.tau = tau
        
        # self.update_every = update_every
        
        # Q-networks
        self.qnetwork_local = QNetwork(action_size, seed).to(device)
        self.qnetwork_target = QNetwork(action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=1e-4, eps=0.01) # learning rate
        
        
        # Replay memory
        
        buffer_size = int(1e5)
        batch_size = 64
        self.memory = ReplayBuffer(buffer_size, batch_size, action_size, seed)
        
        # time step
        self.t_step = 0
        
        
    
    def step(self, state, action, reward, next_state, done): # to save the experience in the replay buffer
        """save exp in replay memory, use random sample from buffer to learn"""
        
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % 4
        
        if self.t_step == 0:
            if len(self.memory) > self.memory.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, gamma=self.gamma)

    
    def act(self, state, eps=0.): # to select an action
        """
        Returns actions for given state for current policy
        e-greedy exploration
        
            state(np.array): current state
            eps(float): epsilon for e-greedy
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        
        self.qnetwork_local.train()
        
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.randint(self.action_size)
        
        
    
    def learn(self, experiences, gamma): # to update the Q-network
        """
        Update value parameters
        
        experiences(Tuple[torch.Tensor]): (state, actions, rewards, next_states, done)
        """
        
        states, actions, rewards, next_states, dones = experiences
        
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Q_targets_next = self.qnetwork_target(next_states).gather(1, self.qnetwork_local(next_states).argmax(1, keepdim=True)).detach()
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        loss = F.mse_loss(Q_expected, Q_targets)
        
        self.optimizer.zero_grad()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1) # to prevent exploding gradients
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    
    def soft_update(self, local_model, target_model, tau): # to update the target network
        """
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)
    
    # def save(self, filename): # to save the model
    #     pass
    
    # def load(self, filename): # to load the model
    #     pass

# Initialize the agent
agent = DQNAgent(action_size=5, seed=0)

def dqn_scores(n_episodes=2000, max_t=100000, eps_start=1.0, eps_end=0.1, eps_decay=0.99):
    """
    Deep Q-Learning.

        n_episodes (int): Maximum number of training episodes
        max_t (int): Maximum number of timesteps per episode
        eps_start (float): Starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): Minimum value of epsilon
        eps_decay (float): Multiplicative factor (per episode) for decreasing epsilon
        
        return:
        scores (list): list of scores from each episode
    """
    
    scores = [] # list of scores from each episode
    scores_window = deque(maxlen=100) # last 100 scores
    eps = eps_start
    
    for i_episode in tqdm(range(1, n_episodes+1)):
        state, _ = env.reset()
        score = 0
        
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            
            # reward = np.clip(reward, -1, 1) # clip the reward to be between -1 and 1
            if reward <= 0:
                reward = -0.1
            
            score += reward
    
            if done:
                break
            
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        eps = eps
        
        scores.append(score)
        scores_window.append(score)
        
        if i_episode % 10 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
        
        if np.mean(scores_window) >= 300.0:
            print(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
        
        if np.max(scores) >= 1000:
            print(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
        
    return scores

scores = dqn_scores()


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
fig.savefig('dqn_training_scores.png')

def create_video(agent, env, filename='dqn_training_video.mp4', folder='videos'):
    """
    create a video of the agent
    """

    video_env = gym.wrappers.RecordVideo(env, video_folder=folder, episode_trigger=lambda x: True)
    state, _ = video_env.reset()
    done = False
    
    while not done:
        action = agent.act(state, eps=0.0) # greedy action
        state, reward, terminated, truncated, _ = video_env.step(action)
        done = terminated or truncated
    video_env.close()
    
    final_path = os.path.join(folder, filename)
    
    if os.path.exists(final_path): # 
        os.remove(final_path)
    
    # os.rename(video_file, filename)
    
    video_dir = folder
    video_file = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')], reverse=True)[0]
    
    latest_video = os.path.join(video_dir, video_file)
    os.rename(latest_video, final_path)
    
    print(f"Video saved as {filename}")
    
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))


create_video(agent, env)