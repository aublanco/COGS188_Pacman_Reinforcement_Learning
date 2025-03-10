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

gym.register_envs(ale_py)
env = gym.make('MsPacman-v4', render_mode='rgb_array')

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork:
    pass
class ReplayBuffer:
    pass
class DQNAgent:
    pass