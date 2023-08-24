# -*- coding: utf-8 -*-

import random
import torch
from collections import namedtuple, deque
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        """
        Class for storing and sampling experiences for training a RL agent.

        Parameters:
        -----------
        buffer_size : int
            The maximum size of the replay buffer.
        batch_size : int
            The size of each batch sampled from the replay buffer.

        Attribute:
        ----------
        memory : collections.deque
            The replay buffer.
        experience : collections.namedtuple
            A name tuple representing a single experience.
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.memory)

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
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """
        Sample a mini-batch of experiences from the buffer.

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
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return states, actions, rewards, next_states, dones