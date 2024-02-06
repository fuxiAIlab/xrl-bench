# -*- coding: utf-8 -*-

import gymnasium as gym
import torch
import os
import numpy as np
import pandas as pd
import torchvision.transforms as T
from collections import deque
from d3rlpy.dataset import MDPDataset
from xrlbench.custom_environment.dunkcitydynasty.agent import Agent
from xrlbench.custom_environment.dunkcitydynasty.common.wrappers import RLWrapper
from xrlbench.custom_environment.dunkcitydynasty.env.gym_env import GymEnv


def global_state_names():
    global_state_names = ['attack_remain_time', 'match_remain_time', 'is_home_team', 'ball_position_x',
                          'ball_position_y', 'ball_position_z', 'vec_ball_basket_x', 'vec_ball_basket_y',
                          'vec_ball_basket_z', 'team_own_ball', 'enemy_team_own_ball', 'ball_clear', 'ball_status_0',
                          'ball_status_1', 'ball_status_2', 'ball_status_3', 'ball_status_4', 'ball_status_5',
                          'can_rebound', 'dis_to_rebound_x', 'dis_to_rebound_z', 'dis_to_rebound_y', 'can_block',
                          'shoot_block_pos_x', 'shoot_block_pos_z', 'dis_to_block_pos_x', 'dis_to_block_pos_z',
                          'dis_to_block_pos_y', 'block_diff_angle', 'block_diff_r']
    return global_state_names


def agent_state_names(agent_id):
    agent_state_names = ['character_id', 'position_type', 'buff_key', 'buff_value', 'stature', 'rational_shoot_distance',
                         'position_x', 'position_y', 'position_z', 'v_delta_x', 'v_delta_z', 'player_to_me_dis_x',
                         'player_to_me_dis_z', 'basket_to_me_dis_x', 'basket_to_me_dis_z', 'ball_to_me_dis_x',
                         'ball_to_me_dis_z', 'polar_to_me_angle', 'polar_to_me_r', 'polar_to_basket_angle',
                         'polar_to_basket_r', 'polar_to_ball_angle', 'polar_to_ball_r', 'facing_x', 'facing_y',
                         'facing_z', 'block_remain_best_time', 'block_remain_time', 'is_out_three_line', 'is_ball_owner',
                         'own_ball_duration', 'cast_duration', 'power', 'is_cannot_dribble', 'is_pass_receiver',
                         'is_marking_opponent', 'is_team_own_ball', 'inside_defence', 'is_my_team_0', 'is_my_team_1',
                         'player_state_0', 'player_state_1', 'player_state_2', 'player_state_3', 'player_state_4',
                         'player_state_5', 'skill_state_0', 'skill_state_1', 'skill_state_2', 'skill_state_3', 'skill_state_4',
                         'skill_state_5', 'skill_state_6', 'skill_state_7', 'skill_state_8', 'skill_state_9', 'skill_state_10',
                         'skill_state_11', 'skill_state_12', 'skill_state_13', 'skill_state_14', 'skill_state_15', 'skill_state_16',
                         'skill_state_17', 'skill_state_18', 'skill_state_19', 'skill_state_20', 'skill_state_21', 'skill_state_22',
                         'skill_state_23', 'skill_state_24', 'skill_state_25', 'skill_state_26']
    agent_state_names = ['{}_{}'.format(state, agent_id) for state in agent_state_names]
    return agent_state_names


def legal_action_names(action_size):
    legal_action_names = ['legal_{}'.format(i) for i in range(action_size)]
    return legal_action_names


def categorical_names(agent_id):
    categorical_state_names = ['character_id', 'position_type', 'buff_key', 'buff_value', 'is_out_three_line', 'is_ball_owner', 'is_cannot_dribble', 'is_pass_receiver',
                            'is_marking_opponent', 'is_team_own_ball', 'is_my_team_0', 'is_my_team_1',
                         'player_state_0', 'player_state_1', 'player_state_2', 'player_state_3', 'player_state_4',
                         'player_state_5', 'skill_state_0', 'skill_state_1', 'skill_state_2', 'skill_state_3', 'skill_state_4',
                         'skill_state_5', 'skill_state_6', 'skill_state_7', 'skill_state_8', 'skill_state_9', 'skill_state_10',
                         'skill_state_11', 'skill_state_12', 'skill_state_13', 'skill_state_14', 'skill_state_15', 'skill_state_16',
                         'skill_state_17', 'skill_state_18', 'skill_state_19', 'skill_state_20', 'skill_state_21', 'skill_state_22',
                         'skill_state_23', 'skill_state_24', 'skill_state_25', 'skill_state_26']
    categorical_state_names = ['{}_{}'.format(state, agent_id) for state in categorical_state_names]
    return categorical_state_names


class DunkCityDynasty:
    def __init__(self, env_id=1,  state_names=None, categorical_states=None):
        """
        Class for constructing a Dunk City Dynasty environment.

        Parameters:
        -----------
        env_id : int (default=1)
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
        try:
            self.env_config = {
                'id': env_id,
                'env_setting': 'win',
                'client_path': 'E:/rl-project/166364_train',
                'rl_server_ip': '127.0.0.1',
                'rl_server_port': 6666,
                'game_server_ip': '42.186.153.157',
                'game_server_port': 18100,
                'machine_server_ip': '',
                'machine_server_port': 0,
                "user_name": "",
                'render': False,
            }
            self.wrapper = RLWrapper({})
            self.env = GymEnv(self.env_config, wrapper=self.wrapper)
        except:
            print("The environment is only supported in Python3.8")
        self.action_size = 52
        self.state_names = global_state_names() + agent_state_names('self_state') + agent_state_names(
            'ally_0_state') + agent_state_names('ally_1_state') + agent_state_names(
            'enemy_0_state') + agent_state_names('enemy_1_state') + agent_state_names(
            'enemy_2_state') + legal_action_names(self.action_size) if state_names is None else state_names
        self.state_size = len(self.state_names)
        self.agent = Agent(state_size=self.state_size, action_size=self.action_size)
        self.model = self.agent.qnetwork_local
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.categorical_states = ['is_home_team', 'team_own_ball', 'enemy_team_own_ball', 'ball_clear', 'ball_status_0',
                                   'ball_status_1', 'ball_status_2', 'ball_status_3', 'ball_status_4', 'ball_status_5',
                                   'can_rebound', 'can_block'] + categorical_names('self_state') + categorical_names('ally_0_state') + categorical_names('ally_1_state') + categorical_names('enemy_0_state') + categorical_names('enemy_1_state') + categorical_names('enemy_2_state') + legal_action_names(self.action_size) if categorical_states is None else categorical_states
        self.load_model()

    def load_model(self):
        """
        Load model for the reinforcement learning agent.
        """
        try:
            self.model.load_state_dict(torch.load(os.path.join(".", "model", "DunkCityDynasty.pth"), map_location=self.device)) # DunkCityDynasty
        except:
            print("This model is not existing, please train it.")

    def train_model(self):
        print("Please refer to the following code repository for model training: "
              "https://github.com/FuxiRL/DunkCityDynasty")

    def get_dataset(self, generate=False, n_episodes=80, max_t=2000, data_format="csv"):
        """
        Get the dataset for the Dunk City Dynasty environment.

        Parameters:
        -----------
        generate : bool
            Whether to generate a new dataset or use an existing one.
        n_episode : int
            The number of episodes to generate the dataset from.
        max_t : int
            The maximum number of timesteps per episode.

        Returns:
        --------
        df : pandas.DataFrame
            The dataset including state, action and reward.
        """
        if generate:
            self.agent.qnetwork_local.load_state_dict(torch.load(os.path.join(".", "model", "DunkCityDynasty.pth")))
            data = []
            for i in range(n_episodes):
                states = self.env.reset()[0]
                print("episode:", i)
                for t in range(max_t):
                    actions = self.agent.act(states)
                    next_states, rewards, dones, truncated, infos = self.env.step(actions)
                    for key in states:
                        state_data = []
                        for state in states[key]:
                            state_data += list(state)
                        data.append({"state": np.array(state_data), "action": np.array([actions[key]])})
                    if dones['__all__']:
                        break
                    states = next_states
            if data_format == "csv":
                dataset = [np.concatenate([row['state'], row['action']], axis=0) for row in data]
                columns_name = self.state_names + ["action"]
                dataset = pd.DataFrame(dataset, columns=columns_name)
                # dataset = dataset[dataset['character_id_self_state'] == 9]
                dataset.to_csv(os.path.join(".", "data", "DunkCityDynasty_dataset.csv"), index=False)
                return dataset
        else:
            try:
                if data_format == "csv":
                    dataset = pd.read_csv(os.path.join(".", "data", "DunkCityDynasty_dataset.csv"))
                    return dataset
                else:
                    raise NotImplementedError("This data format is not supported at the moment.")
            except:
                print("This dataset is not existing, please generate it.")


if __name__ == "__main__":
    env = DunkCityDynasty()
    print(env.state_names)
    print(len(env.state_names))
