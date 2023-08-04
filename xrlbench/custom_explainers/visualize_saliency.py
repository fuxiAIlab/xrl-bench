# -*- coding: utf-8 -*-

import copy
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VisualizeSaliency:
    def __init__(self, X, y, model, categorical_names=None):
        """
        Class for generating visualized saliency scores in reinforcement learning. https://arxiv.org/abs/1711.00138

        Parameters:
        -----------
        X : pandas.DataFrame
            The input data of shape (n_samples, n_features).
        y : pandas.Series or numpy.ndarray
            The label of the input data of shape (n_sample,).
        model : torch.nn.Module
            The trained model used for predicting Q-values.
        categorical_names : list, optional (default=None)
            List of categorical feature names.

        Attributes:
        -----------
        feature_dim : int
            The number of feature in the input data.
        feature_names : list of str
            The names of features in the input data.
        saliency_scores : numpy.ndarray
            The saliency scores of the input data of feature (n_samples, n_features).

        Methods:
        --------
        _add_noise(feature, categorical_names):
            Add noise to the input feature data.
        _calculate_saliency(feature):
            Calculate the saliency score of the input feature data.
        explain(X=None):
            Explain the input feature data by calculating the saliency scores.
        """
        # Check inputs
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas.DataFrame")
        if not isinstance(y, (np.ndarray, pd.Series)):
            raise TypeError("y must be a numpy.ndarray or pandas.Series")
        self.feature_names = list(X.columns)
        self.X = X.values
        self.y = y.values if isinstance(y, pd.Series) else y
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.saliency_scores = np.zeros((len(X), X.shape[1]))
        self.feature_dim = X.shape[1]
        self.categorical_names = categorical_names if categorical_names else []

    def _add_noise(self, feature, mean=0, std=0.05):
        """
        Add noise to the input feature data.

        Parameters:
        -----------
        feature : numpy.ndarray
            The input feature data of shape (n_features,).

        Returns:
        --------
        res : numpy.ndarray
            The noised feature data of shape (n_features, n_features).
        """
        feature_backend = copy.deepcopy(feature)
        for i, name in enumerate(self.feature_names):
            if name in self.categorical_names:
                feature[i] = random.choice(np.unique(self.X[:, i]))
            else:
                # feature_min, feature_max = np.min(self.X[:, i]), np.max(self.X[:, i])
                # feature_mean, feature_std = np.mean(self.X[:, i]), np.std(self.X[:, i])
                # feature_noised = max(min(torch.normal(feature_mean, feature_std, size=(1,)).cpu().item(), feature_max),
                #                      feature_min)
                # feature[i] = feature_noised
                noise = np.random.normal(mean, std)
                feature[i] += noise
        res = np.tile(feature_backend, (self.feature_dim, 1))
        res[np.arange(self.feature_dim), np.arange(self.feature_dim)] = feature
        return res

    def _calculate_saliency(self, feature):
        """
        Calculate the saliency score of the input feature data.

        Parameters:
        -----------
        feature : numpy.ndarray
            The input feature data of shape (n_features,).
        Return:
        -------
        score : numpy.ndarray
            The saliency scores of the input feature data of shape (n_features,).
        """
        feature_noised = self._add_noise(feature)
        with torch.no_grad():
            Q = self.model(torch.from_numpy(feature).float().unsqueeze(0).to(self.device))
            Q_perturbed = self.model(torch.from_numpy(feature_noised).float().unsqueeze(0).to(self.device))
        Q = np.squeeze(Q.cpu().numpy())
        Q_max_idx = np.argmax(Q)
        Q_perturbed = np.squeeze(Q_perturbed.cpu().numpy())
        score = np.abs(Q_perturbed[:, Q_max_idx] - Q[Q_max_idx])
        return score

    def explain(self, X=None, batch_size=128):
        """
        Explain the input feature data by calculating the saliency scores.

        Parameters:
        -----------
        X : pandas.DataFrame, optional (default=None)
            The input data of shape (n_samples, n_features).
        batch_size : int, optional (default=128)
            The batch size for processing the input data.

        Returns:
        --------
        saliency_scores : numpy.ndarray
            The saliency scores of the input data of feature (n_samples, n_features).
        """
        if X is None:
            X = self.X
        X = X.values if isinstance(X, pd.DataFrame) else X
        saliency_scores = np.zeros((len(X), self.feature_dim))
        for i in tqdm(range(0, len(X), batch_size), ascii=True):
            batch = X[i:i + batch_size]
            scores = np.array([self._calculate_saliency(x) for x in batch])
            saliency_scores[i:i+batch_size] = scores
        self.saliency_scores = saliency_scores
        return self.saliency_scores






