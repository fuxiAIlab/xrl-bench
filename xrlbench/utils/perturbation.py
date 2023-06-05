# -*- coding: utf-8 -*-

import numpy as np
import random


def get_normal_perturbed_inputs(state, perturbed_inds, states, categorical_state_inds=None, mean=0, std=0.05):
    if categorical_state_inds is None:
        categorical_state_inds = []

    for i in perturbed_inds:
        if i in categorical_state_inds:
            state[i] = random.choice(np.unique(states[:, i]))
        else:
            noise = np.random.normal(mean, std, 1)[0]
            state[i] = state[i] + noise
    return state







