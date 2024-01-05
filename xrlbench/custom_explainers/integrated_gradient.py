# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset


class IntegratedGradient:
    def __init__(self, X, y, model, steps=50, background=None):
        """
        Class for explaining the model prediction using IntegratedGradient. https://arxiv.org/pdf/1703.01365.pdf
        Original codes. https://github.com/TianhongDai/integrated-gradient-pytorch/blob/master/integrated_gradients.py

        Parameters:
        -----------
        X : pandas.DataFrame
            The input data for the model.
        y : pandas.Series or numpy.ndarray
            The output data for the model.
        model : object
            The trained deep model used for making predictions.
        steps : int (default=50)
            The number of steps in the approximation of integral.
        background : numpy.ndarray or pandas.DataFrame, optional (default=None)
            The background dataset to use for integrating out features. 100-1000 samples will be good.
        Attributes:
        -----------
        explainer : shap.GradientExplainer
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
        self.steps = steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        background = background.values if isinstance(background, pd.DataFrame) else background
        np.random.seed(42)
        self.background = background if background else X.values[np.random.choice(X.shape[0], 100, replace=False)]
        self.background = torch.from_numpy(self.background).float().to(self.device)

    def integrated_gradients(self, x, y):
        all_intgrads = []
        for i in range(self.background.shape[0]):
            baseline = self.background[i]
            scaled_inputs = [baseline + (k / self.steps) * (x - baseline) for k in range(self.steps + 1)]
            inputs = torch.stack(scaled_inputs, dim=0)
            inputs.requires_grad = True
            with torch.autograd.set_grad_enabled(True):
                outputs = self.model(inputs)
                outputs = outputs[:, int(y.item())]
                grads = torch.autograd.grad(torch.unbind(outputs), inputs)[0]
            avg_grads = torch.mean(grads[:-1], dim=0)
            integrated_grad = (x - baseline) * avg_grads
            all_intgrads.append(integrated_grad.cpu().numpy())
        avg_grads = np.average(all_intgrads, axis=0)
        return avg_grads

    def explain(self, X=None, y=None, batch_size=1):
        """
        Explain the input data.

        Parameters:
        -----------
        X : pandas.DataFrame, optional (default=None)
            The input data of shape (n_samples, n_features).

        Returns:
        --------
        gradient_scores : numpy.ndarray
        """
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.Series) else y
        if len(X) != len(y):
            raise ValueError("X must have the same length as y")
        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).to(self.device)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        integrated_gradients = []
        self.model.eval()
        for batch in tqdm(dataloader, ascii=True):
            inputs, labels = batch[0], batch[1]
            batch_intgrads = [self.integrated_gradients(x, y) for x, y in zip(inputs, labels)]
            integrated_gradients.append(batch_intgrads)
        integrated_gradients = np.concatenate(integrated_gradients, axis=0)
        return integrated_gradients


class ImageIntegratedGradient:
    def __init__(self, X, y, model, steps=100, background=None):
        """
        Class for explaining the model prediction using IntegratedGradient. https://arxiv.org/pdf/1703.01365.pdf

        Parameters:
        -----------
        X : numpy.ndarray
            The input data for the model.
        y : pandas.Series or numpy.ndarray
            The output data for the model.
        model : object
            The trained deep model used for making predictions.
        steps : int (default=50)
            The number of steps in the approximation of integral.
        background : numpy.ndarray or pandas.DataFrame, optional (default=None)
            The background dataset to use for integrating out features. 100-1000 samples will be good.
        Attributes:
        -----------
        explainer : shap.GradientExplainer
            The SHAP explainer used for computing the SHAP values.
        """
        # Check inputs
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy.ndarray")
        if not isinstance(y, (np.ndarray, pd.Series)):
            raise TypeError("y must be a numpy.ndarray or pandas.Series")
        self.X = X
        self.y = y
        self.model = model
        self.steps = steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        np.random.seed(42)
        self.background = background if background else X[np.random.choice(X.shape[0], 100, replace=False)]
        self.background = torch.from_numpy(self.background).float().to(self.device)

    def integrated_gradients(self, x, y):
        all_intgrads = []
        for i in range(self.background.shape[0]):
            baseline = self.background[i]
            scaled_inputs = [baseline + (k / self.steps) * (x - baseline) for k in range(self.steps + 1)]
            inputs = torch.stack(scaled_inputs, dim=0)
            inputs.requires_grad = True
            with torch.autograd.set_grad_enabled(True):
                outputs = self.model(inputs)
                outputs = outputs[:, int(y.item())]
                grads = torch.autograd.grad(torch.unbind(outputs), inputs)[0]
            avg_grads = torch.mean(grads[:-1], dim=0)
            integrated_grad = (x - baseline) * avg_grads
            all_intgrads.append(integrated_grad.cpu().numpy())
        avg_grads = np.average(all_intgrads, axis=0)
        return avg_grads

    def explain(self, X=None, y=None, batch_size=1):
        """
        Explain the input data.

        Parameters:
        -----------
        X : numpy.ndarray, optional (default=None)
            The input data of shape (n_samples, n_channels, n_widths, n_heights).

        Returns:
        --------
        gradient_scores : numpy.ndarray
        """
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        y = y.values if isinstance(y, pd.Series) else y
        if len(X) != len(y):
            raise ValueError("X must have the same length as y")
        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).to(self.device)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        integrated_gradients = []
        self.model.eval()
        for batch in tqdm(dataloader, ascii=True):
            inputs, labels = batch[0], batch[1]
            batch_intgrads = [self.integrated_gradients(x, y) for x, y in zip(inputs, labels)]
            integrated_gradients.append(batch_intgrads)
        integrated_gradients = np.concatenate(integrated_gradients, axis=0)
        return integrated_gradients