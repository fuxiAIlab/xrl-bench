# -*- coding: utf-8 -*-

import xrlbench.custom_datasets

valid_datasets = {
    "lunarLander": xrlbench.custom_datasets.LunarLander

}


class DataSet:
    def __init__(self, data_name, **kwargs):
        if data_name not in valid_datasets.keys():
            raise NotImplementedError(
                f"This dataset is not supported at the moment. Datasets supported are: {list(valid_datasets.keys())}"
            )
        self.data_name = data_name
        self.dataset = valid_datasets[data_name]()
        self.agent = valid_datasets[data_name]().agent
        # .get_dataset(generate=generate)

    def train_model(self, ending_score=220):
        return self.dataset.train_model(ending_score=ending_score)

    def get_dataset(self, generate=False):
        return self.dataset.get_dataset(generate=generate)




