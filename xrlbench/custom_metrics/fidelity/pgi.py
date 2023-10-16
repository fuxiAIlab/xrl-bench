# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from xrlbench.utils.perturbation import get_normal_perturbed_inputs
import shap


class PGI:
    def __init__(self, environment, **kwargs):
        """
        Class for evaluating the Prediction Gap on Important feature perturbation [PGI].

        Parameters:
        -----------
        environment : object
            The environment used for evaluating XRL methods.
        """
        self.environment = environment

    def evaluate(self, X, y, feature_weights, k=3):
        """
        Evaluate the performance of XRL methods using PGI metric.

        Parameters:
        -----------
        X : pandas.DataFrame
            The input data.
        y : pandas.Series or numpy.ndarray
            The true labels for the input data.
        feature_weights : numpy.ndarray or shap.Explanation
            The feature weights computed using an XRL method.
        k : int, optional (default=5)
            The number of top feature to mask.

        Returns:
        --------
        mean prediction gap : float
            The prediction gap on important feature perturbation.
        """
        # Check inputs
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas.DataFrame")
        if not isinstance(y, (np.ndarray, pd.Series)):
            raise TypeError("y must be a numpy.ndarray or pandas.Series")
        if not isinstance(feature_weights, (np.ndarray, shap.Explanation)):
            raise TypeError("feature_weights must be a np.ndarray or shap.Explanation")
        feature_names = list(X.columns)
        X = X.values
        y = y.values if isinstance(y, pd.Series) else y
        feature_weights = feature_weights.values if isinstance(feature_weights, shap.Explanation) else feature_weights
        if len(np.array(feature_weights).shape) == 3:
            feature_weights = [feature_weights[i, :, int(y[i])] for i in range(len(feature_weights))]
        elif len(np.array(feature_weights).shape) != 2:
            raise ValueError("Invalid shape for feature weights.")
        prediction_gap = []
        weights_ranks = [np.argsort(-feature_weights[i])[:k] for i in range(len(feature_weights))]
        X_perturbed = X.copy()
        categorical_feature_inds = [feature_names.index(name) for name in self.environment.categorical_states]
        X_perturbed = get_normal_perturbed_inputs(X_perturbed, weights_ranks, categorical_feature_inds)
        for i in range(X.shape[0]):
            y_pred = self.environment.agent.inference(X[i])[0][int(y[i])]
            perturbed_y_pred = self.environment.agent.inference(X_perturbed[i])[0][int(y[i])]
            prediction_gap.append(np.abs(y_pred - perturbed_y_pred))
        return np.mean(prediction_gap)


class ImagePGI:
    def __init__(self, environment, **kwargs):
        """
        Class for evaluating the Prediction Gap on Important feature perturbation [PGI].

        Parameters:
        -----------
        environment : object
            The environment used for evaluating XRL methods.
        """
        self.environment = environment

    def evaluate(self, X, y, feature_weights, k=30):
        """
        Evaluate the performance of XRL methods using PGI metric.

        Parameters:
        -----------
        X : numpy.ndarray
            The input data.
        y : pandas.Series or numpy.ndarray
            The true labels for the input data.
        feature_weights : numpy.ndarray or shap.Explanation
            The feature weights computed using an XRL method.
        k : int, optional (default=5)
            The number of top feature to mask.

        Returns:
        --------
        mean prediction gap : float
            The prediction gap on important feature perturbation.
        """
        # Check inputs
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy.ndarray")
        if not isinstance(y, (np.ndarray, pd.Series)):
            raise TypeError("y must be a numpy.ndarray or pandas.Series")
        if not isinstance(feature_weights, (np.ndarray, shap.Explanation)):
            raise TypeError("feature_weights must be a np.ndarray or shap.Explanation")
        y = y.values if isinstance(y, pd.Series) else y
        feature_weights = feature_weights.values if isinstance(feature_weights, shap.Explanation) else feature_weights
        if len(np.array(feature_weights).shape) == 5:
            feature_weights = [np.sum(feature_weights[i, :, :, :, int(y[i])], axis=0) for i in range(len(feature_weights))]
        elif len(np.array(feature_weights).shape) == 4:
            feature_weights = [feature_weights[i, :, :, int(y[i])] for i in range(len(feature_weights))]
        elif len(np.array(feature_weights).shape) != 3:
            raise ValueError("Invalid shape for feature weights.")

        prediction_gap = []
        weights_ranks = [np.argsort(-feature_weights[i], axis=None)[:k] for i in range(len(feature_weights))]
        X_perturbed = X.copy()
        X_perturbed = get_normal_perturbed_inputs(X_perturbed, weights_ranks)
        for i in range(X.shape[0]):
            y_pred = self.environment.agent.inference(X[i])[0][int(y[i])]
            perturbed_y_pred = self.environment.agent.inference(X_perturbed[i])[0][int(y[i])]
            prediction_gap.append(np.abs(y_pred - perturbed_y_pred))
        return np.mean(prediction_gap)


