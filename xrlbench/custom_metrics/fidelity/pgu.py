# -*- coding: utf-8 -*-

import copy
import numpy as np
from xrlbench.utils.perturbation import get_normal_perturbed_inputs


class PGU:
    def __init__(self, dataset, **kwargs):
        """
        Prediction Gap on Unimportant feature pertubation [PGU]
        """
        self.dataset = dataset

    def evaluate(self, X, y, feature_weights, k=5):
        prediction_gap = []
        if len(np.array(feature_weights).shape) == 2:
            weights_ranks = [np.argsort(feature_weights[i]) for i in range(len(feature_weights))]
        elif len(np.array(feature_weights).shape) == 3:
            weights_ranks = [np.argsort(feature_weights[i, :, int(y[i])]) for i in range(len(feature_weights))]
        for i in range(X.shape[0]):
            if i % 1000 == 0:
                print(i)
            action_value = self.dataset.agent.inference(X[i])[y]
            state_backen = copy.deepcopy(X[i])
            state_perturbed = get_normal_perturbed_inputs(state_backen, weights_ranks[:-k], X)
            action_values_perturbed = self.dataset.agent.inference(state_perturbed)[y]
            prediction_gap.append(np.abs(action_value - action_values_perturbed))
        return np.mean(prediction_gap)

