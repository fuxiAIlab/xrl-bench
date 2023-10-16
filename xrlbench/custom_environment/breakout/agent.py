# -*- coding: utf-8 -*-

import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from xrlbench.custom_environment.breakout.model import QNetwork
from xrlbench.custom_environment.breakout.buffer import ReplayBuffer


class Agent:
    def __init__(self, action_size, buffer_size=100000, batch_size=32, lr=0.0000625, gamma=0.99, policy_update=4, target_update=10000,
                 initial_epsilon=1.0, final_epsilon=0.1, eps_decay=1000000):
        """
        Class for constructing a reinforcement learning agent.

        Parameters :
        ------------
        action_size : int
            Number of possible actions.
        buffer_size : int
            Size of the replay buffer.
        batch_size : int
            Batch size for training.
        lr : float
            Learning rate for the optimizer.
        gamma : float
            Discount factor for future rewards.
        policy_update : int
            Frequency of updating the policy network.
        target_update : int
            Frequency of updating the target network.
        initial_epsilon : float
            Initial value of epsilon for epsilon-greedy exploration.
        final-epsilon : float
            Final value of epsilon.
        eps_decay : int
            Number of steps for decaying epsilon from initial to final value.

        Attributes:
        -----------
        policy_network : QNetwork
            Policy network.
        qnetwork_target : QNetwork
            Target network.
        optimizer : torch.optim.Adam
            Optimizer for training the policy network.
        memory : ReplayBuffer
            Replay buffer for storing experiences.
        t-step : int
            Number of steps taken.
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.policy_update = policy_update
        self.target_update = target_update
        self._eps = initial_epsilon
        self._final_epsilon = final_epsilon
        self._initial_epsilon = initial_epsilon
        self._eps_decay = eps_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network = QNetwork(action_size).to(self.device)
        self.target_network = QNetwork(action_size).to(self.device)
        self.policy_network.apply(self.policy_network.init_weights)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr, eps=1.5e-4)
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size)
        self.t_step = 0

    def inference(self, state):
        """
        Perform inference on the Q-network.

        Parameters:
        -----------
        state: numpy.ndarray
            The current state.

        Returns:
        --------
        action_values : torch.Tensor
            The action values.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.policy_network.eval()
        with torch.no_grad():
            action_values = self.policy_network(state)
        return action_values.cpu()

    def act(self, state, training=False, inferring=False):
        """
        Perform an action based on the current state.

        Parameters:
        -----------
        state : numpy.ndarray
            The current state.
        training : bool
            Whether the agent is in training mode.
        inferring : bool
            Whether the agent is in referring model.

        Returns:
        --------
        action : int
            The selected action.
        """
        state = torch.from_numpy(state).float().to(self.device)
        if inferring:
            with torch.no_grad():
                action_values = self.policy_network(state).max(1)[1].cpu().view(1, 1)
        else:
            sample = random.random()
            if training:
                self._eps -= (self._initial_epsilon - self._final_epsilon) / self._eps_decay
                self._eps = max(self._eps, self._final_epsilon)
            if sample > self._eps:
                with torch.no_grad():
                    action_values = self.policy_network(state).max(1)[1].cpu().view(1, 1)
            else:
                action_values = torch.tensor([[random.randrange(self.action_size)]], device=self.device, dtype=torch.long)
        return action_values.cpu().numpy()[0, 0].item()

    def optimize_model(self, train):
        """
        Optimize the policy network.

        Parameters:
        -----------
        train : bool
            Whether to perform training or not.
        """
        if not train:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards[:, 0] + self.gamma * next_q_values * (1. - dones[:, 0])

        loss = F.smooth_l1_loss(q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()