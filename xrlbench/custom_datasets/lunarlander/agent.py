# -*- coding: utf-8 -*-

import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from xrlbench.custom_datasets.lunarlander.model import QNetwork
from xrlbench.custom_datasets.lunarlander.buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
UPDATE_EVERY = 4
GAMMA = 0.99
LR = 5e-4
TAU = 1e-3


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.memory = ReplayBuffer(action_size, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE)
        self.t_step = 0

    def inference(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        return action_values.cpu()

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        q_hat = self.qnetwork_target.forward(next_states).detach().max(dim=1)[0].unsqueeze(1)
        target = rewards + gamma * q_hat * (1 - dones)
        q_local = self.qnetwork_local.forward(states).gather(1, actions)
        loss = F.mse_loss(q_local, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


