# -*- coding: utf-8 -*-

import xrlbench.custom_environment

valid_environments = {
    "lunarLander": xrlbench.custom_environment.LunarLander,
    "cartPole": xrlbench.custom_environment.CartPole,
    "flappyBird": xrlbench.custom_environment.FlappyBird,
    "breakOut": xrlbench.custom_environment.BreakOut,
    "pong": xrlbench.custom_environment.Pong
}


class Environment:
    def __init__(self, environment_name, **kwargs):
        """
        Constructs an instance of an environment for a given environment.

        Parameters:
        -----------
        environment_name : str
            The name of the environment to be used. Supported environments are:
            "lunarLander"
        **kwargs :
            Keyword arguments to be passed to the environment.
        """
        if environment_name not in valid_environments.keys():
            raise NotImplementedError(
                f"This dataset is not supported at the moment. Environments supported are: {list(valid_environments.keys())}"
            )
        self.environment_name = environment_name
        self.environment = valid_environments[environment_name]()
        self.agent = self.environment.agent
        self.model = self.environment.model
        self.categorical_states = self.environment.categorical_states
        # .get_dataset(generate=generate)

    def train_model(self, **kwargs):
        """
        Train the model on the selected environment.

        Parameters:
        -----------
        ending_score : int, optional
            The score at which the training will stop. Default is 220.
        """
        return self.environment.train_model(**kwargs)

    def get_dataset(self, generate=False, data_format="csv", **kwargs):
        """
        Return the dataset for the selected environment.

        Parameters:
        -----------
        generate : bool, optional
            Whether to generate a new dataset or use the cached one. Default is False.
        data_format: string, optional
            Define the format of saved dataset. "csv" and "h5" are supported. Default is "csv".
        """
        return self.environment.get_dataset(generate=generate, data_format=data_format, **kwargs)




