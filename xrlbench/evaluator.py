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
        """
        Constructs an instance of an evaluator for a given metric.

        Parameters:
        -----------
        metric : str
            The name of the metirc to be used. Supported metrics are : "AIM", "AUM", "PGI", "PGU", "RIS"
        dataset : pandas.DataFrame
            The dataset to be used for evaluation.
        **kwargs :
            Keyword arguments to be passed to the evaluator.
        """
        if metric not in valid_metrics.keys():
            raise NotImplementedError(
                f"This metirc is not supported at the moment. Metrics supported are {list(valid_metrics.keys())}"
            )
        self.metric = metric
        self.evaluator = valid_metrics[metric](dataset=dataset, **kwargs)

    def evaluate(self, X, y, feature_weights, **kwargs):
        """
        Evaluates the model using the selected metric.

        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The input data.
        y : pandas.Series or numpy.ndarray
            The true labels for the input data.
        feature_weights : numpy.ndarray
            The feature weights computed using an XRL method.
        **kwargs :
            Keyword arguments to be passed to the evaluator.
        """
        results = self.evaluator.evaluate(X, y, feature_weights, **kwargs)
        return results
