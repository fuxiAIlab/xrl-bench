# -*- coding: utf-8 -*-

import numpy as np
import random
import copy
from scipy.ndimage.filters import gaussian_filter


def get_mask(feature, centers, size, zero_perturbed):
    """
    Generate a circular mask.

    Parameters:
    -----------
    center : list
        The center coordinates of the circle.
    size : list
        The size of the mask.

    Returns:
    --------
    numpy.ndarray
        The circular mask.
    """
    mask = np.zeros(size)
    for _center in centers:
        center = [_center // size[1], _center % size[1]]
        if not zero_perturbed and feature[0, center[0], center[1]] == 0 and feature[1, center[0], center[1]] == 0 and feature[2, center[0], center[1]] == 0:
            continue
        mask[center[0], center[1]] = 1
    return mask


def add_noise(feature, mask, r=3):
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


def get_normal_perturbed_inputs(X, perturbed_inds, categorical_feature_inds=None, mean=0.0, std=0.5, zero_perturbed=True):
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
        # X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        for i in range(X.shape[0]):
            for ind in perturbed_inds[i]:
                if ind in categorical_feature_inds:
                    X[i, ind] = random.choice(np.unique(X[:, ind]))
                else:
                    noise = np.random.normal(mean, std)
                    X[i, ind] += noise*X_std[ind]
    elif len(X.shape) == 4:
        for i in range(X.shape[0]):
            mask = get_mask(X[i], centers=perturbed_inds[i], size=[X.shape[2], X.shape[3]], zero_perturbed=zero_perturbed)
            X[i] = add_noise(X[i], mask, r=5)
    else:
        raise ValueError("Invalid shape for inputs.")
    return X







