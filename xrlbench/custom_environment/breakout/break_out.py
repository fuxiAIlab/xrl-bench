# -*- coding: utf-8 -*-

import gymnasium as gym
import torch
import os
import numpy as np
import torchvision.transforms as T
from collections import deque
from d3rlpy.dataset import MDPDataset
from xrlbench.custom_environment.breakout.agent import Agent


def preprocess_state(state):
    """
    Preprocess the image state.

    Parameters:
    -----------
    state : numpy.ndarray
        The input state.

    Returns:
    --------
    torch.Tensor
        The preprocessed state.
    """
    return T.Compose([T.ToPILImage(), T.Resize((84, 84)), T.ToTensor()])(state).unsqueeze(0)


class BreakOut:
    def __init__(self, env_id="Breakout-v0"):
        """
        Class for constructing a BreakOut environment.

        Parameters:
        -----------
        env_id : str, optional
            The ID of the environment. (default='Breakout-v0')

        Attributes:
        -----------
        env : gym.Env
            The environment.
        agent : Agent
            The reinforcement learning agent.
        model : QNetwork
            The policy network.
        device : torch.device
            The device used for computations.
        categorical_states : None
            Placeholder for categorical states.
        """
        self.env = gym.make(env_id)
        self.agent = Agent(action_size=self.env.action_space.n)
        self.model = self.agent.policy_network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.categorical_states = None
        self.load_model()

    def load_model(self):
        """
        Load model for the reinforcement learning agent.
        """
        try:
            self.model.load_state_dict(torch.load(os.path.join(".", "model", "BreakOut.pth")))
        except:
            print("This model is not existing, please train it.")

    def train_model(self, n_episodes=50000, max_t=10000, ending_score=16):
        """
        Train the reinforcement learning agent.

        Parameters:
        -----------
        n_episodes : int, optional
            The maximum number of episodes to train for. (default=500000)
        max_t : int, optional
            The maximum number of timesteps per episode. (default=10000)
        ending_score : float, optional
            The average score at which to consider the environment solved. (default=16)
        """
        scores_window = deque(maxlen=100)
        for i in range(1, n_episodes + 1):
            train = len(self.agent.replay_buffer) > 5000
            score = 0
            state = preprocess_state(self.env.reset()[0])
            for t in range(max_t):
                action = self.agent.act(np.array(state), train)
                next_state, reward, done, _, _ = self.env.step(action)
                # print(next_state, reward, done)
                next_state = preprocess_state(next_state)
                self.agent.replay_buffer.add(state, action, reward, next_state, done)
                self.agent.t_step += 1
                if self.agent.t_step % self.agent.policy_update == 0:
                    self.agent.optimize_model(train)

                if self.agent.t_step % self.agent.target_update == 0:
                    self.agent.target_network.load_state_dict(self.agent.policy_network.state_dict())
                state = next_state
                score += reward
                if done:
                    print("score:", score, "t:", t)
                    break
            scores_window.append(score)
            print(f'\rEpisode {i}\tAverage Score: {np.mean(scores_window):.2f}', end='')
            if i % 100 == 0:
                print(f'\rEpisode {i}\tAverage Score: {np.mean(scores_window):.2f}')
            if np.mean(scores_window) >= ending_score:
                print("\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(i, np.mean(scores_window)))
                break
        torch.save(self.agent.policy_network.state_dict(), os.path.join(".", "model", "BreakOut.pth"))
        return self.agent.policy_network

    def get_dataset(self, generate=False, n_episodes=1, max_t=500, data_format="h5"):
        """
        Get the dataset for the Break Out environment.

        Parameters:
        -----------
        generate : bool, optional
            Whether to generate a new dataset or use an existing one. (default=False)
        n_episode : int, optional
            The number of episodes to generate the dataset from. (default=10)
        max_t : int, optional
            The maximum number of timesteps per episode. (default=1000)
        data_format : str, optional
            The format of the dataset. Currently only supports 'h5'. (default='h5')

        Returns:
        --------
        dataset : MDPDataset
            The dataset including state, action, reward and terminal.
        """
        if generate:
            data = []
            for i in range(n_episodes):
                state = preprocess_state(self.env.reset()[0])
                for t in range(max_t):
                    action = self.agent.act(np.array(state), inferring=True)
                    next_state, reward, done, _, _ = self.env.step(action)
                    next_state = preprocess_state(next_state)
                    data.append({"state": np.array(state*255, dtype=np.uint8), "action": np.array([action]), "reward": np.array([reward]), "terminal": np.array([done])})
                    state = next_state
                    if done:
                        break
            if data_format == "h5":
                observations = np.vstack([row["state"] for row in data])
                actions = np.vstack([row["action"] for row in data])
                rewards = np.vstack([row["reward"] for row in data])
                terminals = np.vstack([row["terminal"] for row in data])
                dataset = MDPDataset(observations, actions, rewards, terminals)
                dataset.dump(os.path.join(".", "data", "BreakOut_dataset.h5"))
                return dataset
            else:
                raise NotImplementedError("This data format is not supported at the moment.")
        else:
            try:
                if data_format == "h5":
                    dataset = MDPDataset.load(os.path.join(".", "data", "BreakOut_dataset.h5"))
                    return dataset
                else:
                    raise NotImplementedError("This data format is not supported at the moment.")
            except:
                print("This dataset is not existing, please generate it.")
