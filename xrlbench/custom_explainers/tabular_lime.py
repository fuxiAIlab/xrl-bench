# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
try:
    import lime
    import lime.lime_tabular
except ImportError:
    pass


class TabularLime:
    def __init__(self, X, y, model, categorical_names=None, mode="classification"):
        """
        Class for explaining the predictions of tarbular models using LIME. https://arxiv.org/abs/1602.04938

        Parameters:
        -----------
        X : pandas.DataFrame
            The input data of shape (n_samples, n_features).
        y : pandas.Series or numpy.ndarray
            The label of the input data of shape (n_sample,).
        model : callable or object
            The trained model used for making predictions.
        categorical_names : list, optional (default=None)
            List of categorical feature names.
        mode : str, optional (default="classification")
            The mode of the model, either "classification" or "regression".

        Attributes:
        -----------
        feature_names : list
            List of feature names.
        categorical_index : list
            List of categorical feature indices.
        explainer : lime.lime_tabular.LimeTabularExplainer
            The LIME explainer object.
        out_dim : int
            Number of output dimensions.
        flat_out : bool
            Whether the output is flat (1D) or not.

        Methods:
        --------
        explain(X=None):
            Explain the input feature data by calculating the importance scores.
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
        assert mode in ["classification", "regression"]
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.categorical_names = categorical_names if categorical_names else []
        self.categorical_index = [self.feature_names.index(state) for state in categorical_names] if categorical_names else []
        self.explainer = lime.lime_tabular.LimeTabularExplainer(self.X, mode=mode, feature_names=self.feature_names, categorical_features=self.categorical_index)
        out = self.model(torch.from_numpy(self.X[0:1]).float().to(self.device))
        if len(out.shape) == 1:
            self.out_dim = 1
            self.flat_out = True
            if mode == "classification":
                def pred(X):
                    preds = self.model(X).reshape(-1, 1)
                    p0 = 1 - preds
                    return np.hstack((p0, preds))
                self.model = pred
        else:
            self.out_dim = self.model(torch.from_numpy(self.X[0:1]).float().to(self.device)).shape[1]
            self.flat_out = False

    def explain(self, X=None):
        """
        Explain the input feature data by calculating the importance scores.

        Parameters:
        -----------
        X : pandas.DataFrame, optional (default=None)
            The feature data for which to generate explanations. If None, use the original feature data.

        Returns:
        --------
        importance_scores : list
            List of explanations for each output dimension.
        """
        if X is None:
            X = self.X
        self.model.to("cpu")
        X = X.values if isinstance(X, pd.DataFrame) else X
        importance_scores = [np.zeros(X.shape) for _ in range(self.out_dim)]
        for i in tqdm(range(X.shape[0])):
            x = X[i]
            exp = self.explainer.explain_instance(x, self.model, labels=range(self.out_dim), num_features=X.shape[1])
            for j in range(self.out_dim):
                for k, v in exp.local_exp[j]:
                    importance_scores[j][i, k] = v
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return np.array(importance_scores).transpose((1, 2, 0))