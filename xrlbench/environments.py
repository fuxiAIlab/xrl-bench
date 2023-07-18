# -*- coding: utf-8 -*-

import xrlbench.custom_environment

valid_environments = {
    "lunarLander": xrlbench.custom_environment.LunarLander

}


class DataSet:
    def __init__(self, data_name, **kwargs):
        if data_name not in valid_environments.keys():
            raise NotImplementedError(
                f"This dataset is not supported at the moment. Environments supported are: {list(valid_environments.keys())}"
            )
        self.data_name = data_name
        self.environment = valid_environments[data_name]()
        self.agent = self.environment.agent
        self.categorical_states = self.environment.categorical_states
        # .get_dataset(generate=generate)

    def train_model(self, ending_score=220):
        return self.environment.train_model(ending_score=ending_score)

    def get_dataset(self, generate=False):
        return self.environment.get_dataset(generate=generate)




