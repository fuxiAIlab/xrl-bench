# -*- coding: utf-8 -*-

import xrlbench.custom_metrics

valid_metrics = {
    "AIM": xrlbench.custom_metrics.AIM,
    "AUM": xrlbench.custom_metrics.AUM,
    "PGI": xrlbench.custom_metrics.PGI,
    "PGU": xrlbench.custom_metrics.PGU,
    "RIS": xrlbench.custom_metrics.RIS,
    "imageAIM": xrlbench.custom_metrics.ImageAIM,
    "imageAUM": xrlbench.custom_metrics.ImageAUM,
    "imagePGI": xrlbench.custom_metrics.ImagePGI,
    "imagePGU": xrlbench.custom_metrics.ImagePGU,
    "imageRIS": xrlbench.custom_metrics.ImageRIS
}


class Evaluator:
    def __init__(self, metric, environment, **kwargs):
        """
        Constructs an instance of an evaluator for a given metric.

        Parameters:
        -----------
        metric : str
            The name of the metirc to be used. Supported metrics are : "AIM", "AUM", "PGI", "PGU", "RIS"
        environment : object
            The environment to be used for evaluation.
        **kwargs :
            Keyword arguments to be passed to the evaluator.
        """
        if metric not in valid_metrics.keys():
            raise NotImplementedError(
                f"This metirc is not supported at the moment. Metrics supported are {list(valid_metrics.keys())}"
            )
        self.metric = metric
        self.evaluator = valid_metrics[metric](environment=environment, **kwargs)

    def evaluate(self, X, y, feature_weights, **kwargs):
        """
        Evaluates the model using the selected metric.

        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The input data.
        y : pandas.Series or numpy.ndarray
            The true labels for the input data.
        feature_weights : numpy.ndarray, shap.Explanation or list
            The feature weights computed using an XRL method.
        **kwargs :
            Keyword arguments to be passed to the evaluator.
        """
        X = X / 255 if self.metric.find("image") != -1 else X
        results = self.evaluator.evaluate(X, y, feature_weights, **kwargs)
        return results
