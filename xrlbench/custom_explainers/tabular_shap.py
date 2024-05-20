# -*- coding: utf-8 -*-


import lightgbm
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class TabularSHAP:
    def __init__(self, X, y, categorical_names=None):
        """
        Class for generating SHAP values for tabular data using LightGBM model.

        Parameters:
        -----------
        X : pandas.DataFrame
            The input data of shape (n_samples, n_features).
        y : numpy.ndarray or pandas.Series
            The labels of the input data of shape (n_samples,).
        categorical_names : list, optional (default=None)
            A list of names of categorical features in X.

        Attributes:
        -----------
        predictions : numpy.ndarray
            The predicted labels of the input data of shape (n_samples,).
        report : str
            The classification report of the predicted labels.
        explainer : shap.Explainer
            The SHAP explainer object.

        Methods:
        --------
        _encode_categorical_features():
            Encode the categorical features using LabelEncoder.
        _fit_model():
            Fit the LightGBM model on the training data.
        _generate_shap_values():
            Generate shap values using SHAP explainer.
        """
        self.X = X
        self.y = y
        self.categorical_names = categorical_names if categorical_names else []

        # Check inputs
        if not isinstance(self.X, pd.DataFrame):
            raise TypeError("X must be a pandas.DataFrame object.")
        if not isinstance(self.y, (np.ndarray, pd.Series)):
            raise TypeError("y must be a numpy.ndarray or pandas.Series object.")
        if len(self.X) != len(self.y):
            raise ValueError("The length of X and y does not match.")
        if not isinstance(self.categorical_names, list):
            raise TypeError("categorical_names must be a list.")
        if any([not isinstance(name, str) for name in self.categorical_names]):
            raise TypeError("All elements in categorical_names must be strings.")

        self.X_enc = self.X.copy()
        self.encoders = self._encode_categorical_features()
        self._fit_model()

    def _encode_categorical_features(self):
        encoders = []
        for name in self.categorical_names:
            if self.X_enc[name].dtype != object:
                self.X_enc[name] = self.X_enc[name].astype(str)
            encoder = LabelEncoder()
            encoder.fit(self.X_enc[name])
            encoders.append(encoder)
            self.X_enc[name] = encoder.transform(self.X_enc[name])
        return encoders

    def _fit_model(self):
        """
        TODO: Build a good ensemble tree student model to fit the policy model.
        """
        model = lightgbm.LGBMClassifier()
        self.explainer = shap.Explainer(model)


    def _generate_shap_values(self, X):
        X_enc = X.copy()
        for i in range(len(self.categorical_names)):
            encoder = self.encoders[i]
            if X_enc[self.categorical_names[i]].dtype != object:
                X_enc[self.categorical_names[i]] = X_enc[self.categorical_names[i]].astype(str)
                X[self.categorical_names[i]] = X[self.categorical_names[i]].astype(str)
            X_enc.loc[~X_enc[self.categorical_names[i]].isin(encoder.classes_), self.categorical_names[i]] = 'unknow'
            encoder.classes_ = np.append(encoder.classes_, 'unknow')
            X_enc[self.categorical_names[i]] = encoder.transform(X_enc[self.categorical_names[i]])
        shap_values = self.explainer(X_enc)
        shap_values.display_data = X.values
        return shap_values

    def explain(self, X=None):
        """
        Explain the input data.

        Parameters:
        -----------
        X : pandas.DataFrame, optional (default=None)
            The input data of shape (n_samples, n_features).

        Returns:
        --------
        shap_values : shap.Explanation
            The SHAP values of the input data.
        """
        if X is None:
            X = self.X

        # Check inputs
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas.DataFrame object.")

        X_enc = X.copy()
        shap_values = self._generate_shap_values(X_enc)
        return shap_values




