# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import shap


class AIM:
    def __init__(self, environment, **kwargs):
        """
        Class for evaluating the Accuracy on Important feature Masked by zero padding [AIM].

        Parameters:
        -----------
        environment : object
            The environment used for evaluating XRL methods.
        """
        self.environment = environment

    def evaluate(self, X, y, feature_weights, k=3):
        """
        Evaluate the performance of XRL methods using AIM metric.

        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The input data.
        y : pandas.Series or numpy.ndarray
            The true labels for the input data.
        feature_weights : numpy.ndarray or shap.Explanation
            The feature weights computed using an XRL method.
        k : int, optional (default=5)
            The number of top feature to mask.

        Returns:
        --------
        accuracy : float
            The mean accuracy on important feature masked by zero padding.
        """
        # Check inputs
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be a pandas.DataFrame or a numpy.ndarray")
        if not isinstance(y, (np.ndarray, pd.Series)):
            raise TypeError("y must be a numpy.ndarray or pandas.Series")
        if not isinstance(feature_weights, (np.ndarray, shap.Explanation)):
            raise TypeError("feature_weights must be a np.ndarray or shap.Explanation")
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.Series) else y
        feature_weights = feature_weights.values if isinstance(feature_weights, shap.Explanation) else feature_weights
        if len(np.array(feature_weights).shape) == 3:
            feature_weights = [feature_weights[i, :, int(y[i])] for i in range(len(feature_weights))]
        elif len(np.array(feature_weights).shape) != 2:
            raise ValueError("Invalid shape for feature_weights.")
        weights_ranks = [np.argsort(-feature_weights[i])[:k] for i in range(len(feature_weights))]
        masked_X = X.copy()
        for i in range(X.shape[0]):
            masked_X[i][weights_ranks[i]] = 0
        y_pred = [np.argmax(self.environment.agent.inference(masked_X[i]).data.numpy()) for i in range(masked_X.shape[0])]
        # y_pred = [self.environment.agent.act(masked_X[i]) for i in range(masked_X.shape[0])]
        accuracy = np.mean(y_pred == y)
        return accuracy


class ImageAIM:
    def __init__(self, environment, **kwargs):
        """
        Class for evaluating the Accuracy on Important feature Masked by zero padding [AIM].

        Parameters:
        -----------
        environment : object
            The environment used for evaluating XRL methods.
        """
        self.environment = environment

    def evaluate(self, X, y, feature_weights, k=30):
        """
        Evaluate the performance of XRL methods using AIM metric.

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
        accuracy : float
            The mean accuracy on important feature masked by zero padding.
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
            raise ValueError("Invalid shape for feature_weights.")
        # print(np.sum(feature_weights))
        # print(feature_weights)
        weights_ranks = [np.argsort(-feature_weights[i], axis=None)[:k] for i in range(len(feature_weights))]
        # print(weights_ranks)
        masked_X = X.copy()
        for i in range(X.shape[0]):
            for c in range(len(masked_X[i])):
                flat = masked_X[i][c].flatten()
                flat[weights_ranks[i]] = 0
                masked_X[i][c] = flat.reshape((masked_X.shape[2], masked_X.shape[3]))
        y_pred = [np.argmax(self.environment.agent.inference(masked_X[i]).data.numpy()) for i in range(masked_X.shape[0])]
        # y_pred = [self.environment.agent.act(masked_X[i]) for i in range(masked_X.shape[0])]
        accuracy = np.mean(y_pred == y)
        return accuracy