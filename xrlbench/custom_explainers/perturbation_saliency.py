# -*- coding: utf-8 -*-

import copy
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter


class PerturbationSaliency:
    def __init__(self, X, y, model, categorical_names=None):
        """
        Class for generating perturbation-based saliency scores in reinforcement learning. https://arxiv.org/abs/1711.00138

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
        self.std = np.std(self.X, axis=0)

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
                noise = np.random.normal(mean, std)
                feature[i] += noise*self.std[i]
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
        Q_perturbed = np.squeeze(Q_perturbed.cpu().numpy())
        score = [np.sqrt(np.sum(np.square(Q - Q_perturbed[i]))) for i in range(Q_perturbed.shape[0])]
        return np.array(score)

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


class ImagePerturbationSaliency:
    def __init__(self, X, y, model):
        """
        Class for calculating image perturbation saliency scores.

        Parameters:
        -----------
        X : numpy.ndarray
            The input images.
        y : numpy.ndarray or pandas.Series
            The labels corresponding to the input images.
        model : torch.nn.Module
            The model used for prediction.

        Attributes:
        -----------
        saliency_scores : numpy.ndarray
            The saliency scores of the input images.
        """
        # Check inputs
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy.ndarray")
        if not isinstance(y, (np.ndarray, pd.Series)):
            raise TypeError("y must be a numpy.ndarray or pandas.Series")
        self.X = X
        self.y = y.values if isinstance(y, pd.Series) else y
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.saliency_scores = np.zeros((len(X), X.shape[2], X.shape[3]))

    def _get_mask(self, center, size, r=5):
        """
        Generate a circular mask.

        Parameters:
        -----------
        center : list
            The center coordinates of the circle.
        size : list
            The size of the mask.
        r : int
            The radius of the circle.

        Returns:
        --------
        numpy.ndarray
            The circular mask.
        """
        y, x = np.ogrid[-center[0]:size[0]-center[0], -center[1]:size[1]-center[1]]
        keep = x*x + y*y <= 1
        mask = np.zeros(size)
        mask[keep] = 1
        mask = gaussian_filter(mask, sigma=r)
        return mask / mask.max()

    def _add_noise(self, feature, mask, r=3):
        """
        Add noise to the feature using the mask.

        Parameters:
        -----------
        feature : numpy.ndarray
            The input feature.
        mask : numpy.ndarray
            The mask to be applied.
        r : int
            The sigma value for the Gaussian filter.

        Returns:
        --------
        numpy.ndarray
            The feature with added noise.
        """
        feature_backend = copy.deepcopy(feature)
        for c in range(0, feature.shape[0]):
            feature_backend[c] = feature_backend[c] * (1 - mask) + gaussian_filter(feature_backend[c], sigma=r) * mask
        return feature_backend

    def _calculate_saliency(self, feature):
        """
        Calculate the saliency scores for a given feature.

        Parameters:
        -----------
        feature : numpy.ndarray
            The input feature.

        Returns:
        --------
        numpy.ndarray
            The saliency scores for the feature.
        """
        scores = np.zeros((feature.shape[1], feature.shape[2]))
        with torch.no_grad():
            Q = self.model(torch.from_numpy(feature).float().unsqueeze(0).to(self.device))
        Q = np.squeeze(Q.cpu().numpy())
        for i in range(0, feature.shape[1]):
            for j in range(0, feature.shape[2]):
                mask = self._get_mask(center=[i, j], size=[feature.shape[1], feature.shape[2]], r=5)
                feature_noised = self._add_noise(feature, mask, r=3)
                with torch.no_grad():
                    Q_perturbed = self.model(torch.from_numpy(feature_noised).float().unsqueeze(0).to(self.device))
                Q_perturbed = np.squeeze(Q_perturbed.cpu().numpy())
                scores[i, j] = np.sqrt(np.sum(np.square(Q - Q_perturbed)))
        return scores

    def explain(self, X=None, batch_size=1):
        """
        Compute the saliency scores for the input images.

        Parameters:
        -----------
        X : numpy.ndarray, optional
            The input images. If not provided, the images from the initialization will be used.
        batch_size : int, optional
            The batch size for computing the seliency scores.

        Returns:
        --------
        numpy.ndarray
            The saliency scores for the feature.
        """
        if X is None:
            X = self.X
        saliency_scores = np.zeros((len(X), X.shape[2], X.shape[3]))
        for i in tqdm(range(0, len(X), batch_size), ascii=True):
            batch = X[i:i + batch_size]
            scores = np.array([self._calculate_saliency(x) for x in batch])
            saliency_scores[i:i+batch_size] = scores
        self.saliency_scores = saliency_scores
        return self.saliency_scores



