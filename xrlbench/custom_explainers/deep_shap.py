# -*- coding: utf-8 -*-

import shap
import torch
import numpy as np
import pandas as pd


class DeepSHAP:
    def __init__(self, X, y, model, background=None):
        """
        Class for explaining the model prediction using DeepSHAP. https://arxiv.org/abs/1704.02685

        Parameters:
        -----------
        X : pandas.DataFrame
            The input data for the model.
        y : pandas.Series or numpy.ndarray
            The output data for the model.
        model : object
            The trained deep model used for making predictions.
        background : numpy.ndarray or pandas.DataFrame, optional (default=None)
            The background dataset to use for integrating out features. 100-1000 samples will be good.

        Attributes:
        -----------
        explainer : shap.DeepExplainer
            The SHAP explainer used for computing the SHAP values.
        """
        # Check inputs
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas.DataFrame")
        if not isinstance(y, (np.ndarray, pd.Series)):
            raise TypeError("y must be a numpy.ndarray or pandas.Series")
        self.X = X
        self.y = y
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.background = background if background else X.values[np.random.choice(X.shape[0], 100, replace=False)]
        self.explainer = shap.DeepExplainer(model, torch.from_numpy(self.background).float().to(self.device))

    def explain(self, X=None):
        """
        Explain the input data.

        Parameters:
        -----------
        X : pandas.DataFrame, optional (default=None)
            The input data of shape (n_samples, n_features).

        Returns:
        --------
        shap_values : array or list
            For a models with a single output this returns a tensor of SHAP values with the same shape
            as X. For a model with multiple outputs this returns a list of SHAP value tensors, each of
            which are the same shape as X.
        """
        if X is None:
            X = self.X
        shap_values = self.explainer.shap_values(torch.from_numpy(X.values).float().to(self.device))
        return np.array(shap_values).transpose((1, 2, 0))