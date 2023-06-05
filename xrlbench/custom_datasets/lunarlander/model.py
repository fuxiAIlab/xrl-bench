# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        hidden_layer = 64
        self.fc1 = nn.Linear(state_size, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer)
        self.fc3 = nn.Linear(hidden_layer, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
