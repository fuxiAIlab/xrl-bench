# -*- coding: utf-8 -*-

import gymnasium as gym
import torch
import numpy as np
import pandas as pd
from collections import deque
from xrlbench.custom_datasets.lunarlander.agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LunarLander:
    def __init__(self, env_id='LunarLander-v2', state_size=8, action_size=4, load_model=True, state_names=None, categorical_states=None):
        self.env = gym.make(env_id)
        self.agent = Agent(state_size=state_size, action_size=action_size)
        self.state_names = ["Horizontal_coordinates", "Vertical_coordinates", "Horizontal_speed", "Vertical_speed", "Angle",
                             "Angular_velocity", "Leg1_touchdown", "Leg2_touchdown"] if state_names is None else state_names
        self.categorical_states = [] if categorical_states is None else categorical_states
        if load_model:
            try:
                self.agent.qnetwork_local.load_state_dict(torch.load("./model/LunarLander.pth"))
            except:
                print("This model is not existing, please train it.")

    def train_model(self, n_episodes=10000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, ending_score=220):
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
        torch.save(self.agent.qnetwork_local.state_dict(), "./model/LunarLander.pth")

    def get_dataset(self, generate=False, n_episodes=500, max_t=1000):
        if generate:
            self.agent.qnetwork_local.load_state_dict(torch.load("./model/LunarLander.pth"))
            data = []
            for i in range(n_episodes):
                state = self.env.reset()[0]
                for t in range(max_t):
                    action = self.agent.act(state)
                    next_state, reward, done, _, _ = self.env.step(action)
                    data.append({"state": np.array(state), "action": np.array([action]), "reward": np.array([reward])})
                    state = next_state
                    if done:
                        break
            data = [np.concatenate([row["state"], row["action"], row["reward"]], axis=0) for row in data]
            columns_name = ["Horizontal_coordinates", "Vertical_coordinates", "Horizontal_speed", "Vertical_speed", "Angle",
                             "Angular_velocity", "Leg1_touchdown", "Leg2_touchdown", "action", "reward"]
            df = pd.DataFrame(data, columns=columns_name)
            df.to_csv("./data/LunarLander_dataset.csv", index=False)
            return df
        else:
            try:
                df = pd.read_csv("./data/LunarLander_dataset.csv")
                return df
            except:
                print("This dataset is not existing, please generate it.")


if __name__ == "__main__":
    ll = LunarLander()
    # ll.train_model()
    df = ll.get_dataset(generate=True)











