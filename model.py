import copy
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from collections import deque, namedtuple
from typing import Deque, Dict, List, Tuple
import settings as s

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size



class DQN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)

class Agents:
    def __init__(self, device, world):
        self.world = world
        self.num_vehicles = world.num_vehicles
        self.num_customers = world.num_customers
        self.input_dim = self.num_customers * 2 + 4
        self.output_dim = self.num_customers
        self.policy_net = DQN(self.input_dim, self.output_dim).to(device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.optim = optim.RMSprop(self.policy_net.parameters(), s.LEARNING_RATE)
        self.memory = ReplayBuffer(self.input_dim, self.output_dim, s.BATCH_SIZE)
        self.vehicles = [Agent(i, self.num_customers) for i in range(self.num_vehicles)]

class Agent:
    def __init__(self, vehicle_id, num_customers):
        self.vehicle_id = vehicle_id