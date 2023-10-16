# -*- coding: utf-8 -*-

import random
import torch
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size):
        """
        Class for storing and sampling experiences for training a RL agent.

        Parameters:
        -----------
        buffer_size : int
            The maximum size of the replay buffer.

        Attribute:
        ----------
        buffer : deque
            The replay buffer deque.
        device : torch.device
            The device used for computations.
        """
        self.buffer = deque(maxlen=buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer.

        Parameters:
        -----------
        state : numpy.ndarray
            The state.
        action : int
            The action.
        reward : float
            The reward.
        next_state : numpy.ndarray
            The next state.
        done : bool
            Whether the episode has ended.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a mini-batch of experiences from the buffer.

        Parameters:
        -----------
        batch_size : int
            The size of the mini-batch.

        Returns:
        --------
        states : torch.Tensor
            The states.
        actions : torch.Tensor
            The actions.
        rewards : torch.Tensor
            The rewards.
        next_states : torch.Tensor
            The next states.
        dones : torch.Tensor
            Whether the episodes have ended.
        """
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Get the length of the buffer.

        Returns:
        --------
        int
            The length of the buffer.
        """
        return len(self.buffer)