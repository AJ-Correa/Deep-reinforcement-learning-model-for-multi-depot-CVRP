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
    def __init__(self, world):
        self.world = world
        self.num_vehicles = world.num_vehicles
        self.num_customers = world.num_customers
        self.input_dim = self.num_customers * 2 + 4
        self.output_dim = self.num_customers + 1
        self.policy_net = DQN(self.input_dim, self.output_dim).to(s.DEVICE)
        self.target_net = copy.deepcopy(self.policy_net)
        self.optim = optim.RMSprop(self.policy_net.parameters(), s.LEARNING_RATE)
        self.memory = ReplayBuffer(self.input_dim, s.REPLAY_SIZE, s.BATCH_SIZE)
        self.epoch_id = 0
        self.loss_log = [0 for _ in range(s.NUM_EPOCHS)]

    def transition(self, state, feasible_actions_indexes):
        sample = random.random()

        self.epsilon_threshold = s.EPSILON_END + (s.EPSILON_START - s.EPSILON_END) * math.exp(
            -1. * self.epoch_id / s.EPSILON_DECAY)

        q_values = self.policy_net(torch.tensor(np.array(state).astype(np.float32)).to(s.DEVICE))

        if sample < self.epsilon_threshold:
            op_id = random.choice(feasible_actions_indexes)
        else:
            with torch.no_grad():
                temp_best_q = -100000000000
                for i in feasible_actions_indexes:
                    if q_values[i] > temp_best_q:
                        temp_best_q = q_values[i]
                        op_id = i

        return op_id

    def optimize_model(self):
        if len(self.memory) < s.BATCH_SIZE:
            return

        samples = self.memory.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(s.DEVICE)
        next_state = torch.FloatTensor(samples["next_obs"]).to(s.DEVICE)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(s.DEVICE)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(s.DEVICE)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(s.DEVICE)

        expected_q_values = self.policy_net(state).gather(1, action)
        target_q_values = self.target_net(next_state).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target_q_values = (reward + s.GAMMA * target_q_values * mask).to(s.DEVICE)

        loss = F.smooth_l1_loss(expected_q_values, target_q_values)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.loss_log[self.epoch_id] += loss.item()