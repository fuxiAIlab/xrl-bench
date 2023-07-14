# -*- coding: utf-8 -*-

import copy
import numpy as np
import pandas as pd
from xrlbench.utils.perturbation import get_normal_perturbed_inputs


class RIS:
    def __init__(self, environment, **kwargs):
        """
        Class for evaluating the Relative Input Stability [RIS].

        Parameters:
        -----------
        environment : object
            The environment used for evaluating XRL methods.
        """
        self.environment = environment

    def evaluate(self, X, y, feature_weights, explainer):
        """
        Evaluate the RIS of an XRL method.

        Parameters:
        -----------
        X : pandas.DataFrame
            The input data.
        y : pandas.Series or numpy.ndarray
            The labels of the input data.
        feature_weights : numpy.ndarray
            The feature weights computed using an XRL method.
        explainer : object
            The explainer used to compute the feature weights.

        Returns:
        --------
        float
            The RIS of the XRL method.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input data must be a pandas.DataFrame.")
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y must be a numpy.ndarray or pandas.Series")
        if not hasattr(explainer, 'explain'):
            raise AttributeError("Explainer object must have an 'explain' method.")

        feature_names = list(X.columns)
        X = X.values
        y = y.values if isinstance(y, pd.Series) else y
        stability_ratios = []
        X_perturbed = X.copy()
        perturbed_inds = [range(X.shape[1]) for _ in range(X.shape[0])]
        categorical_feature_inds = [feature_names.index(name) for name in self.environment.categorical_states]
        X_perturbed = get_normal_perturbed_inputs(X_perturbed, perturbed_inds, categorical_feature_inds)
        perturbed_weights = explainer.explain(pd.DataFrame(X_perturbed, columns=feature_names))
        if not isinstance(perturbed_weights, np.ndarray):
            perturbed_weights = perturbed_weights.values
        if len(np.array(feature_weights).shape) == 3:
            feature_weights = [feature_weights[i, :, int(y[i])] for i in range(len(feature_weights))]
            perturbed_weights = [perturbed_weights[i, :, int(y[i])] for i in range(len(perturbed_weights))]
        for i in range(X.shape[0]):
            # compute x's difference ratio
            x_diff = X[i] - X_perturbed[i]
            x_clip = np.clip(X[i], a_min=0.001, a_max=None)
            x_flat_diff_norm = np.linalg.norm(np.divide(x_diff, x_clip), ord=2)

            # compute weight's difference ratio
            weight_diff = feature_weights[i] - perturbed_weights[i]
            weight_clip = np.clip(feature_weights[i], a_min=0.001, a_max=None)
            weight_flat_diff_norm = np.linalg.norm(np.divide(weight_diff, weight_clip), ord=2)

            stability = np.divide(weight_flat_diff_norm, x_flat_diff_norm)
            stability_ratios.append(stability)
        return max(stability_ratios)


        # stability_ratios = []
        # state_names = list(X.columns)
        # X = X.values
        # states_backen = copy.deepcopy(X)
        # states_perturbed = get_normal_perturbed_inputs(states_backen, range(X.shape[1]), self.dataset.categorical_states)
        # states_perturbed_weights = explainer.explain(pd.DataFrame(states_perturbed, columns=state_names))
        # if len(np.array(feature_weights).shape) == 3:
        #     feature_weights = [feature_weights[i, :, int(y[i])] for i in range(len(feature_weights))]
        #     states_perturbed_weights = [states_perturbed_weights[i, :, int(y[i])] for i in range(len(states_perturbed_weights))]
        # for i in range(X.shape[0]):
        #     # x_diff
        #     x_diff = X[i] - states_perturbed[i]
        #     X[i] = np.clip(X[i], a_min=0.001, a_max=None)
        #     x_flat_diff = np.divide(x_diff, X[i], where=X[i] != 0)
        #     x_flat_diff_norm = np.linalg.norm(x_flat_diff, ord=2)
        #
        #     weight_diff = feature_weights[i] - states_perturbed_weights[i]
        #     feature_weights[i] = np.clip(feature_weights[i], a_min=0.001, a_max=None)
        #     weight_flat_diff = np.divide(weight_diff, feature_weights[i], where=feature_weights[i] != 0)
        #     weight_flat_diff_norm = np.linalg.norm(weight_flat_diff, ord=2)
        #
        #     stability_measure = np.divide(weight_flat_diff_norm, x_flat_diff_norm)
        #     stability_ratios.append(stability_measure)
        # ind_max = np.argmax(stability_ratios)
        # return stability_ratios[ind_max]


