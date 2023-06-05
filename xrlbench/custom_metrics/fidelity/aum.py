# -*- coding: utf-8 -*-

import numpy as np


class AUM:
    def __init__(self, dataset, **kwargs):
        """
        Accuracy on Unimportant feature Masked by zero padding [AUM]
        """
        self.dataset = dataset

    def evaluate(self, X, y, feature_weights, k=5):
        accuracy = []
        if len(np.array(feature_weights).shape) == 2:
            weights_ranks = [np.argsort(feature_weights[i]) for i in range(len(feature_weights))]
        elif len(np.array(feature_weights).shape) == 3:
            weights_ranks = [np.argsort(feature_weights[i, :, int(y[i])]) for i in range(len(feature_weights))]
        for i in range(X.shape[0]):
            if i % 1000 == 0:
                print(i)
            X[i][weights_ranks[i][:-k]] = 0
            action = self.dataset.agent.act(X[i])
            if action == y[i]:
                accuracy.append(1)
            else:
                accuracy.append(0)
        return np.mean(accuracy)
