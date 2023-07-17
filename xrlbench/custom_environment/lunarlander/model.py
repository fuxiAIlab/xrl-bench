# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        """
        Class for constructing a deep Q-network (DQN) for reinforcement learning.

        Parameters:
        -----------
        state_size : int
            The number of features in the state representation.
        action_size : int
            The number of possible actions.

        Attributes:
        -----------
        fc1 : torch.nn.Linear
            The first fully connected layer of the network.
        fc2 : torch.nn.Linear
            The second fully connected layer of the network.
        fc3 : torch.nn.Linear
            The output layer of the network.
        """
        super(QNetwork, self).__init__()
        hidden_layer = 64
        self.fc1 = nn.Linear(state_size, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer)
        self.fc3 = nn.Linear(hidden_layer, action_size)

    def forward(self, state):
        """
        Feed-forward computation of the Q-network.

        Parameters:
        -----------
        state : torch.Tensor
            The input state.

        Returns:
        --------
        torch.Tensor
            The output of the network.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
