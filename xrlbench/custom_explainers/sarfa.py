# -*- coding: utf-8 -*-

import os
import torch
import copy
import random
from tqdm import tqdm
from scipy.stats import entropy
from scipy.special import logsumexp
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SARFA:
    def __init__(self, X, y, model, categorical_states=None):
        self.state = X
        self.action = y
        self.model = model
        self.saliency = []
        self.state_dim = X.shape[1]
        self.state_names = list(X.columns)
        self.categorical_states = categorical_states if categorical_states else []
        self.use_top_score = False

    def add_noise(self, state, categorical_states):
        state_backen = copy.deepcopy(state)
        for i in range(len(state)):
            if self.state_names[i] in categorical_states:
                state[i] = random.choice(np.unique(self.state[self.state_names[i]]))
            else:
                value_min, value_max = np.min(self.state[self.state_names[i]]), np.max(self.state[self.state_names[i]])
                value_mean, value_std = np.mean(self.state[self.state_names[i]]), np.std(self.state[self.state_names[i]])
                noised_state = np.random.normal(value_mean, value_std, 1)[0]
                noised_state = max(min(noised_state, value_max), value_min)
                state[i] = noised_state
        res = np.array([state_backen] * len(state), dtype=float)
        for i in range(len(state)):
            res[i, i] = state[i]
        return res

    def explain(self, states=None):
        """
        calculate saliency scores according to specificity and relevance. score = 2*dP*K/( K + dP), dp
        Args:
        Returns: saliency_scores
        """
        if states is None:
            states = self.state
        for i in tqdm(range(states.shape[0]), ascii=True):
            state = states.values[i]
            with torch.no_grad():
                Q = self.model(torch.from_numpy(state).float().unsqueeze(0).to(device))
            Q = np.squeeze(Q.cpu())
            # Q_P = np.exp(Q) / sum(np.exp(Q))
            Q_P_log = Q - logsumexp(Q)
            Q_P = np.exp(Q_P_log)
            Q_idx = np.argmax(Q_P)
            scores = np.zeros(self.state_dim)
            state = self.add_noise(state, self.categorical_states)
            with torch.no_grad():
                Q_perturbed = self.model(torch.from_numpy(state).float().unsqueeze(0).to(device))
            Q_perturbed = np.squeeze(Q_perturbed.cpu())
            for idx in range(self.state_dim):
                # Q_perturbed_P = np.exp(Q_perturbed[idx]) / sum(np.exp(Q_perturbed[idx]))
                Q_perturbed_P_log = Q_perturbed[idx] - logsumexp(Q_perturbed[idx])
                Q_perturbed_P = np.exp(Q_perturbed_P_log)
                dP = Q_P[Q_idx] - Q_perturbed_P[Q_idx]
                if dP > 0:
                    # cross_entropy
                    P_rem = np.append(Q_P[:Q_idx], Q_P[Q_idx + 1:])
                    P_perturbed_rem = np.append(Q_perturbed_P[:Q_idx], Q_perturbed_P[Q_idx + 1:])
                    P_KL = entropy(P_rem, P_perturbed_rem)
                    K = 1. / (1. + P_KL)
                    scores[idx] = 2 * dP * K / (K + dP)
            # if self.use_top_score:
            #     if np.max(scores) > 0:
            #         self.cal[action][np.argmax(scores)] += 1
            # else:
            self.saliency.append(scores)
        return self.saliency










