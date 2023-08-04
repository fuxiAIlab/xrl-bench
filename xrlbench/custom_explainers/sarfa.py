# -*- coding: utf-8 -*-

import torch
import copy
import random
from tqdm import tqdm
from scipy.stats import entropy
from scipy.special import logsumexp
import numpy as np
import pandas as pd


class SARFA:
    def __init__(self, X, y, model, categorical_names=None):
        """
        Class for generating saliency scores using SARFA (Specific and Relevant Feature Attribution). https://arxiv.org/abs/1912.12191

        Parameters:
        -----------
        X : pandas.DataFrame
            The input data of shape (n_samples, n_features).
        y : numpy.ndarray or pandas.Series
            The label of the input data of shape (n_sample,).
        model : pytorch model
            The trained reinforcement learning model.
        categorical_names : list, optional (default=None)
            A list of names of categorical features in X.

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
        _add_noise(feature):
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
                # feature_noised = max(min(torch.normal(feature_mean, feature_std, size=(1,)).cpu().item(), feature_max), feature_min)
                # feature[i] = feature_noised
                noise = np.random.normal(mean, std)
                feature[i] += noise

        res = np.tile(feature_backend, (self.feature_dim, 1))
        res[np.arange(self.feature_dim), np.arange(self.feature_dim)] = feature
        return res

    def _calculate_saliency(self, feature):
        """
        Calculate the saliency score of the input feature data according to specificity and relevance. score = 2*dP*K/( K + dP).

        Parameters:
        -----------
        feature : numpy.ndarray
            The input feature data of shape (n_features,).
        Return:
        -------
        score : numpy.ndarray
            The saliency scores of the input feature data of shape (n_features,).
        """
        score = np.zeros(self.feature_dim)
        feature_noised = self._add_noise(feature)
        with torch.no_grad():
            Q = self.model(torch.from_numpy(feature).float().unsqueeze(0).to(self.device))
            Q_perturbed = self.model(torch.from_numpy(feature_noised).float().unsqueeze(0).to(self.device))
        Q = Q.squeeze().cpu().numpy()
        Q_P_log = Q - np.logaddexp.reduce(Q)
        Q_P = np.exp(Q_P_log)
        Q_idx = np.argmax(Q_P)
        Q_perturbed = Q_perturbed.squeeze().cpu().numpy()

        # Q_perturbed_P_log = Q_perturbed - np.logaddexp.reduce(Q_perturbed, axis=1, keepdims=True)
        # Q_perturbed_P = np.exp(Q_perturbed_P_log)
        # dP = Q_P[Q_idx] - Q_perturbed_P[Q_idx]
        # P_rem = np.delete(Q_P, Q_idx)
        # P_perturbed_rem = np.delete(Q_perturbed_P, Q_idx, axis=1)
        # P_KL = np.sum(P_rem * (np.log(P_rem) - np.logaddexp.reduce(P_perturbed_rem, axis=1)))
        # K = 1. / (1. + P_KL)
        # score[dP > 0] = 2 * dP[dP > 0] * K / (K + dP[dP > 0])
        for idx in range(self.feature_dim):
            Q_perturbed_P_log = Q_perturbed[idx] - logsumexp(Q_perturbed[idx])
            Q_perturbed_P = np.exp(Q_perturbed_P_log)
            dP = Q_P[Q_idx] - Q_perturbed_P[Q_idx]
            if dP > 0:
                P_rem = np.append(Q_P[:Q_idx], Q_P[Q_idx + 1:])
                P_perturbed_rem = np.append(Q_perturbed_P[:Q_idx], Q_perturbed_P[Q_idx + 1:])
                P_KL = entropy(P_rem, P_perturbed_rem)
                K = 1. / (1. + P_KL)
                score[idx] = 2 * dP * K / (K + dP)
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
            batch = X[i:i+batch_size]
            scores = np.array([self._calculate_saliency(x) for x in batch])
            saliency_scores[i:i+batch_size] = scores
        self.saliency_scores = saliency_scores
        return self.saliency_scores










