# -*- coding: utf-8 -*-

import xrlbench.custom_metrics

valid_metrics = {
    "AIM": xrlbench.custom_metrics.AIM,
    "AUM": xrlbench.custom_metrics.AUM,
    "PGI": xrlbench.custom_metrics.PGI,
    "PGU": xrlbench.custom_metrics.PGU,
    "RIS": xrlbench.custom_metrics.RIS
}


class Evaluator:
    def __init__(self, metric, dataset, **kwargs):
        if metric not in valid_metrics.keys():
            raise NotImplementedError(
                f"This metirc is not supported at the moment. Metrics supported are {list(valid_metrics.keys())}"
            )
        self.metric = metric
        self.evaluator = valid_metrics[metric](dataset=dataset, **kwargs)

    def evaluate(self, X, y, feature_weights, k=5):
        results = self.evaluator.evaluate(X, y, feature_weights, k)
        return results
