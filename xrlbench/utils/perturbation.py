# -*- coding: utf-8 -*-

import numpy as np
import random


def get_normal_perturbed_inputs(X, perturbed_inds, categorical_feature_inds=None, mean=0.0, std=0.05):
    """
    Perturb the input data by adding Gaussian noise to selected features.

    Parameters:
    -----------
    X : numpy.ndarray
        The input data to perturb, (n_samples, n_features).
    perturbed_inds : list
        The indices of the features to perturb, (n_samples, k).
    categorical_feature_inds : list, optional (default=None)
        The indices of the categorical features.
    mean : float, optional (default=0)
        The mean of the Gaussian noise to add.
    std : float, optional (default=0.05)
        The standard deviation of the Gaussian noise to add.

    Returns:
    --------
    numpy.ndarray
        The perturbed input data.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("Input data must be a numpy.ndarray")
    if not isinstance(perturbed_inds, list):
        raise TypeError("Perturbed indices must be a list")
    if categorical_feature_inds is not None and not isinstance(categorical_feature_inds, list):
        raise TypeError("Categorical feature indices must be a list or None.")
    if not isinstance(mean, float):
        raise TypeError("Mean must be a float.")
    if not isinstance(std, float):
        raise TypeError("Standard deviation must be a float.")
    if std < 0:
        raise ValueError("Standard deviation must be non-negative.")

    if len(X.shape) == 2:
        for i in range(X.shape[0]):
            for ind in perturbed_inds[i]:
                if ind in categorical_feature_inds:
                    X[i, ind] = random.choice(np.unique(X[:, ind]))
                else:
                    noise = np.random.normal(mean, std)
                    X[i, ind] += noise
    elif len(X.shape) == 4:
        for i in range(X.shape[0]):
            for c in range(X[i].shape[0]):
                flat = X[i, c].flatten()
                for ind in perturbed_inds[i]:
                    noise = np.random.normal(mean, std)
                    flat[ind] += noise
                X[i, c] = flat.reshape(X.shape[2], X.shape[3])
    else:
        raise ValueError("Invalid shape for inputs.")
    return X

    # if categorical_state_inds is None:
    #     categorical_state_inds = []
    #
    # for i in range(states.shape[0]):
    #     for ind in perturbed_inds:
    #         if ind in categorical_state_inds:
    #             states[i, ind] = random.choice(np.unique(states[:, ind]))
    #         else:
    #             noise = np.random.normal(mean, std, 1)[0]
    #             states[i, ind] = states[i, ind] + noise
    # return states







