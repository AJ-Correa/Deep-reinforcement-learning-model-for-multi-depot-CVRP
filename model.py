import copy
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GINConv, global_add_pool
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from collections import deque, namedtuple
from typing import Deque, Dict, List, Tuple
import settings as s

class VehicleReplayBuffer:
    """Replay buffer for customer agent, handling graph structures."""

    def __init__(self, size: int, batch_size: int = 32):
        self.obs_buf = []
        self.next_obs_buf = []
        self.acts_buf = np.zeros(size, dtype=np.int32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0

    def store(self, obs, act, rew, next_obs, done):
        if self.size < self.max_size:
            self.obs_buf.append(obs)
            self.next_obs_buf.append(next_obs)
        else:
            self.obs_buf[self.ptr] = obs
            self.next_obs_buf[self.ptr] = next_obs

        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self):
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            obs=[self.obs_buf[i] for i in idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            next_obs=[self.next_obs_buf[i] for i in idxs]
        )

    def __len__(self):
        return self.size

class CustomerReplayBuffer:
    """Replay buffer for customer agent, handling graph structures."""

    def __init__(self, size: int, batch_size: int = 32):
        self.customer_features_buf = []
        self.edge_index_buf = []
        self.next_customer_features_buf = []
        self.next_edge_index_buf = []
        self.acts_buf = np.zeros(size, dtype=np.int32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0

    def store(self, customer_features, edge_index, act, rew, next_customer_features, next_edge_index, done):
        if self.size < self.max_size:
            self.customer_features_buf.append(customer_features)
            self.edge_index_buf.append(edge_index)
            self.next_customer_features_buf.append(next_customer_features)
            self.next_edge_index_buf.append(next_edge_index)
        else:
            self.customer_features_buf[self.ptr] = customer_features
            self.edge_index_buf[self.ptr] = edge_index
            self.next_customer_features_buf[self.ptr] = next_customer_features
            self.next_edge_index_buf[self.ptr] = next_edge_index

        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self):
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            customer_features=[self.customer_features_buf[i] for i in idxs],
            edge_index=[self.edge_index_buf[i] for i in idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            next_customer_features=[self.next_customer_features_buf[i] for i in idxs],
            next_edge_index=[self.next_edge_index_buf[i] for i in idxs]
        )

    def __len__(self):
        return self.size

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

class CustomerActor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim=1, eps=False):
        super(CustomerActor, self).__init__()

        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.eps = eps

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.convs.append(GINConv(MLP(self.input_dim, self.hidden_dim, self.hidden_dim)))
        self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))

        for layer in range(self.num_layers - 1):
            self.convs.append(GINConv(MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim)))
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))

        self.fc = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x, edge_index, sample=False):
        if sample == True:
            q_values = []
            for state, edge in zip(x, edge_index):
                x = state
                for conv, bn in zip(self.convs, self.batch_norms):
                    x = conv(x, edge)
                    x = bn(x)
                    x = F.relu(x)
                x = self.fc(x)
                q_values.append(x.squeeze(1))

            return torch.stack(q_values)
        else:
            for conv, bn in zip(self.convs, self.batch_norms):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)

            x = self.fc(x)

            return x

class VehicleActor(nn.Module):
    def __init__(self, hidden_dim, output_dim=1):
        super(VehicleActor, self).__init__()

        self.fc = nn.Linear(2, hidden_dim)
        # self.bn = nn.BatchNorm1d(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim, output_dim)

    def forward(self, x, sample=False):
        if sample:
            q_values = []
            for state in zip(x):
                x = self.fc(state[0])
                # x = self.bn(x)
                x = F.relu(x)
                x = self.mlp(x)

                q_values.append(x.squeeze(1))
            return torch.stack(q_values)
        else:
            x = self.fc(x)
            # x = self.bn(x)
            x = F.relu(x)
            x = self.mlp(x)
            return x

class Agents:
    def __init__(self, world):
        self.world = world
        self.num_vehicles = world.num_vehicles
        self.num_customers = world.num_customers

        self.customer_policy_net = CustomerActor(3, 3, 128).to(s.DEVICE)
        self.customer_target_net = copy.deepcopy(self.customer_policy_net)
        self.vehicle_policy_net = VehicleActor(128).to(s.DEVICE)
        self.vehicle_target_net = copy.deepcopy(self.vehicle_policy_net)
        self.customer_optim = optim.Adam(self.customer_policy_net.parameters(), s.LEARNING_RATE)
        self.vehicle_optim = optim.Adam(self.vehicle_policy_net.parameters(), s.LEARNING_RATE)

        self.customer_memory = CustomerReplayBuffer(s.REPLAY_SIZE, s.BATCH_SIZE)
        self.vehicle_memory = VehicleReplayBuffer(s.REPLAY_SIZE, s.BATCH_SIZE)
        self.epoch_id = 0
        self.loss_log = [0 for _ in range(s.NUM_EPOCHS)]

    def transition(self, state, feasible_actions_indexes, edge_indexes, mode):
        sample = random.random()

        self.epsilon_threshold = s.EPSILON_END + (s.EPSILON_START - s.EPSILON_END) * math.exp(
            -1. * self.epoch_id / s.EPSILON_DECAY)

        if mode == "customer":
            q_values = self.customer_policy_net(torch.tensor(np.array(state).astype(np.float32)).to(s.DEVICE), edge_indexes)
        else:
            q_values = self.vehicle_policy_net(torch.tensor(np.array(state).astype(np.float32)).to(s.DEVICE))

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
        if len(self.customer_memory) > s.BATCH_SIZE:
            samples = self.customer_memory.sample_batch()
            state = torch.FloatTensor(samples["customer_features"]).to(s.DEVICE)
            next_state = torch.FloatTensor(samples["next_customer_features"]).to(s.DEVICE)
            action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(s.DEVICE)
            reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(s.DEVICE)
            done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(s.DEVICE)

            edge_index = [torch.LongTensor(ei).to(s.DEVICE) for ei in samples["edge_index"]]
            next_edge_index = [torch.LongTensor(nei).to(s.DEVICE) for nei in samples["next_edge_index"]]

            expected_q_values = self.customer_policy_net(state, edge_index, True).gather(1, action)
            target_q_values = self.customer_target_net(next_state, next_edge_index, True).max(dim=1, keepdim=True)[0].detach()
            mask = 1 - done
            target_q_values = (reward + s.GAMMA * target_q_values * mask).to(s.DEVICE)

            loss = F.smooth_l1_loss(expected_q_values, target_q_values)

            self.customer_optim.zero_grad()
            loss.backward()
            self.customer_optim.step()

            self.loss_log[self.epoch_id] += loss.item()

        if len(self.vehicle_memory) > s.BATCH_SIZE:
            samples = self.vehicle_memory.sample_batch()
            state = torch.FloatTensor(samples["obs"]).to(s.DEVICE)
            next_state = torch.FloatTensor(samples["next_obs"]).to(s.DEVICE)
            action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(s.DEVICE)
            reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(s.DEVICE)
            done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(s.DEVICE)

            expected_q_values = self.vehicle_policy_net(state, True).gather(1, action)
            target_q_values = self.vehicle_target_net(next_state, True).max(dim=1, keepdim=True)[0].detach()
            mask = 1 - done
            target_q_values = (reward + s.GAMMA * target_q_values * mask).to(s.DEVICE)

            loss = F.smooth_l1_loss(expected_q_values, target_q_values)

            self.vehicle_optim.zero_grad()
            loss.backward()
            self.vehicle_optim.step()

            self.loss_log[self.epoch_id] += loss.item()