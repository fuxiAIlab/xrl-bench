# -*- coding: utf-8 -*-

import gymnasium as gym
import flappy_bird_gymnasium
import torch
import os
import numpy as np
import pandas as pd
from collections import deque
from d3rlpy.dataset import MDPDataset
from xrlbench.custom_environment.flappybird.agent import Agent


class FlappyBird:
    def __init__(self, env_id='FlappyBird-v0',  state_names=None, categorical_states=None):
        """
        Class for constructing a FlappyBird environment.

        Parameters:
        -----------
        env_id : str (default='FlappyBird-v0')
            The ID of the environment.
        state_names : list of str
            The names of the state features.
        categorical_states : list of str
            The names of the categorical state features.

        Attributes:
        -----------
        env : gym.Env
            The environment.
        agent : Agent
            The reinforcement learning agent.
        model : QNetwork
            The local Q-network.
        """
        self.env = gym.make(env_id)
        self.agent = Agent(state_size=self.env.observation_space.shape[0], action_size=self.env.action_space.n)
        self.model = self.agent.qnetwork_local
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_names = ["last_pipe_horizontal_position", "last_top_pipe_vertical_position", "last_bottom_pipe_vertical_position", "next_pipe_horizontal_position", "next_top_pipe_vertical_position",
                             "next_bottom_pipe_vertical_position", "next_next_pipe_horizontal_position", "next_next_top_pipe_vertical_position", "next_next_bottom_pipe_vertical_position",
                            "player_vertical_position", "player_vertical_velocity", "player_rotation"] if state_names is None else state_names
        self.categorical_states = [] if categorical_states is None else categorical_states
        self.load_model()

    def load_model(self):
        """
        Load model for the reinforcement learning agent.
        """
        try:
            self.model.load_state_dict(torch.load(os.path.join(".", "model", "FlappyBird.pth")))
        except:
            print("This model is not existing, please train it.")

    def train_model(self, n_episodes=200000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, ending_score=100):
        """
        Train the reinforcement learning agent.

        Parameters:
        -----------
        n_episodes : int
            The maximum number of episodes to train for.
        max_t : int
            The maximum number of timesteps per episode.
        eps_start : float
            The starting value of epsilon for epsilon-greedy action selection.
        eps_end : float
            The minimum value of epsilon for epsilon-greedy action selection.
        eps_decay : float
            The rate at which to decay epsilon.
        ending_score : float
            The average score at which to consider the environment solved.
        """
        scores_window = deque(maxlen=100)
        eps = eps_start
        for i in range(1, n_episodes + 1):
            state = self.env.reset()[0]
            score = 0
            for t in range(max_t):
                action = self.agent.act(np.array(state), eps)
                next_state, reward, done, _, _ = self.env.step(action)
                self.agent.step(np.array(state), action, reward, np.array(next_state), done)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)
            eps = max(eps_end, eps_decay * eps)
            print("\rEpisode {}\tAverage Score: {:.2f}".format(i, np.mean(scores_window)), end="")
            if np.mean(scores_window) >= ending_score:
                print("\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(i, np.mean(scores_window)))
                break
        torch.save(self.agent.qnetwork_local.state_dict(), os.path.join(".", "model", "FlappyBird.pth"))
        return self.agent.qnetwork_local

    def get_dataset(self, generate=False, episodes=500, max_t=1000, data_format="csv"):
        if generate:
            self.model.load_state_dict(torch.load(os.path.join(".", "model", "FlappyBird.pth")))
            data = []
            for e in range(episodes):
                state = self.env.reset()[0]
                for t in range(max_t):
                    action = self.agent.act(state)
                    next_state, reward, done, _, _ = self.env.step(action)
                    data.append({"state": np.array(state), "action": np.array([action]), "reward": np.array([reward]), "terminal": np.array([done])})
                    state = next_state
                    if done:
                        break
            if data_format == "h5":
                observations = np.vstack([row["state"] for row in data])
                actions = np.vstack([row["action"] for row in data])
                rewards = np.vstack([row["reward"] for row in data])
                terminals = np.vstack([row["terminal"] for row in data])
                dataset = MDPDataset(observations, actions, rewards, terminals)
                dataset.dump(os.path.join(".", "data", "FlappyBird_dataset.h5"))
            else:
                dataset = [np.concatenate([row["state"], row["action"], row["reward"]], axis=0) for row in data]
                columns_name = self.state_names + ["action", "reward"]
                dataset = pd.DataFrame(dataset, columns=columns_name)
                dataset.to_csv(os.path.join(".", "data", "FlappyBird_dataset.csv"), index=False)
            return dataset
        else:
            try:
                if data_format == "h5":
                    dataset = MDPDataset.load(os.path.join(".", "data", "FlappyBird_dataset.h5"))
                    dataset = np.hstack((dataset.observations, dataset.actions[:, np.newaxis], dataset.rewards[:, np.newaxis]))
                    columns_name = self.state_names + ["action", "reward"]
                    dataset = pd.DataFrame(dataset, columns=columns_name)
                else:
                    dataset = pd.read_csv(os.path.join(".", "data", "FlappyBird_dataset.csv"))
                return dataset
            except:
                print("This dataset is not existing, please generate it.")


if __name__ == "__main__":
    bird = FlappyBird()
    bird.train_model()

