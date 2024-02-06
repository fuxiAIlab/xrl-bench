# -*- coding: utf-8 -*-

import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from xrlbench.custom_environment.dunkcitydynasty.model import Model, XRLModel


class Agent:
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qnetwork_local = XRLModel().to(self.device)

    def inference(self, state):
        """
        Perform inference on the Q-network.

        Parameters:
        -----------
        state : numpy.ndarray
            The current state.

        Returns:
        --------
        action_values : torch.Tensor
            The action values.
        """
        new_states = []
        i = 0
        for size in [30, 73, 73, 73, 73, 73, 73, 52]:
            new_states.append(torch.tensor(state[i: i+size][np.newaxis, :]).to(self.device)) #
            i += size
        value, probs = self.qnetwork_local(new_states[0].float(), new_states[1], new_states[2], new_states[3], new_states[4], new_states[5], new_states[6], new_states[7].float())
        return probs.detach().cpu() #

    def act(self, states_dic):
        """
        Perform an action based on the current state.

        Parameters:
        -----------
        states : numpy.ndarray
            The current state.
        eps : float
            The epsilon value for epsilon-greedy action selection.

        Returns:
        --------
        action : int
            The selected action.
        """
        actions = {}
        for key in states_dic:
            states = states_dic[key]
            new_states = []
            for state in states:
                new_states.append(state[np.newaxis, :])
            new_states = [torch.tensor(state) for state in new_states]
            value, probs = self.qnetwork_local(new_states[0].float(), new_states[1], new_states[2], new_states[3], new_states[4], new_states[5], new_states[6], new_states[7].float())
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            # action = np.argmax(probs.cpu().data.numpy())
            actions[key] = action
        return actions

        # state = torch.from_numpy(states).float().unsqueeze(0).to(self.device)
        # self.qnetwork_local.eval()
        # with torch.no_grad():
        #     value, probs = self.qnetwork_local(state)
        # return np.argmax(probs.cpu().data.numpy())