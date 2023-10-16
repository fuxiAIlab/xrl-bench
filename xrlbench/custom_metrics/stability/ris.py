# -*- coding: utf-8 -*-

import shap
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
        feature_weights : numpy.ndarray or shap.Explanation
            The feature weights computed using an XRL method.
        explainer : object
            The explainer used to compute the feature weights.

        Returns:
        --------
        float
            The RIS of the XRL method.
        """
        # Check inputs
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input data must be a pandas.DataFrame.")
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y must be a numpy.ndarray or pandas.Series")
        if not isinstance(feature_weights, (np.ndarray, shap.Explanation)):
            raise TypeError("feature_weights must be a np.ndarray or shap.Explanation")
        if not hasattr(explainer, 'explain'):
            raise AttributeError("Explainer object must have an 'explain' method.")

        feature_names = list(X.columns)
        X = X.values
        y = y.values if isinstance(y, pd.Series) else y
        feature_weights = feature_weights.values if isinstance(feature_weights, shap.Explanation) else feature_weights
        stability_ratios = []
        X_perturbed = X.copy()
        perturbed_inds = [range(X.shape[1]) for _ in range(X.shape[0])]
        categorical_feature_inds = [feature_names.index(name) for name in self.environment.categorical_states]
        X_perturbed = get_normal_perturbed_inputs(X_perturbed, perturbed_inds, categorical_feature_inds)
        perturbed_weights = explainer.explain(pd.DataFrame(X_perturbed, columns=feature_names))
        if isinstance(perturbed_weights, shap.Explanation):
            perturbed_weights = perturbed_weights.values
        if len(np.array(feature_weights).shape) == 3:
            feature_weights = [feature_weights[i, :, int(y[i])] for i in range(len(feature_weights))]
            perturbed_weights = [perturbed_weights[i, :, int(y[i])] for i in range(len(perturbed_weights))]
        elif len(np.array(feature_weights).shape) != 2:
            raise ValueError("Invalid shape for feature weights.")
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


class ImageRIS:
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
        X : numpy.ndarray
            The input data.
        y : pandas.Series or numpy.ndarray
            The labels of the input data.
        feature_weights : numpy.ndarray or shap.Explanation
            The feature weights computed using an XRL method.
        explainer : object
            The explainer used to compute the feature weights.

        Returns:
        --------
        float
            The RIS of the XRL method.
        """
        # Check inputs
        if not isinstance(X, np.ndarray):
            raise TypeError("Input data must be a numpy.ndarray.")
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y must be a numpy.ndarray or pandas.Series")
        if not isinstance(feature_weights, (np.ndarray, shap.Explanation)):
            raise TypeError("feature_weights must be a np.ndarray or shap.Explanation")
        if not hasattr(explainer, 'explain'):
            raise AttributeError("Explainer object must have an 'explain' method.")

        y = y.values if isinstance(y, pd.Series) else y
        feature_weights = feature_weights.values if isinstance(feature_weights, shap.Explanation) else feature_weights

        stability_ratios = []
        X_perturbed = X.copy()
        perturbed_inds = [range(X.shape[2] * X.shape[3]) for _ in range(X.shape[0])]
        X_perturbed = get_normal_perturbed_inputs(X_perturbed, perturbed_inds)
        perturbed_weights = explainer.explain(X_perturbed)
        if len(np.array(feature_weights).shape) == 5:
            feature_weights = [np.sum(feature_weights[i, :, :, :, int(y[i])], axis=0) for i in range(len(feature_weights))]
            perturbed_weights = [np.sum(perturbed_weights[i, :, :, :, int(y[i])], axis=0) for i in
                               range(len(perturbed_weights))]
        elif len(np.array(feature_weights).shape) == 4:
            feature_weights = [feature_weights[i, :, :, int(y[i])] for i in range(len(feature_weights))]
            perturbed_weights = [perturbed_weights[i, :, :, int(y[i])] for i in range(len(perturbed_weights))]
        elif len(np.array(feature_weights).shape) != 3:
            raise ValueError("Invalid shape for feature weights.")
        for i in range(X.shape[0]):
            # compute x's difference ratio
            x_diff = X[i] - X_perturbed[i]
            x_clip = np.clip(X[i], a_min=0.001, a_max=None)
            x_flat_diff_norm = np.linalg.norm(np.divide(x_diff, x_clip).flatten(), ord=2)

            # compute weight's difference ratio
            weight_diff = feature_weights[i] - perturbed_weights[i]
            weight_clip = np.clip(feature_weights[i], a_min=0.001, a_max=None)
            weight_flat_diff_norm = np.linalg.norm(np.divide(weight_diff, weight_clip).flatten(), ord=2)

            stability = np.divide(weight_flat_diff_norm, x_flat_diff_norm)
            stability_ratios.append(stability)
        return max(stability_ratios)