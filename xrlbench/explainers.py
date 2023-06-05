# -*- coding: utf-8 -*-

import xrlbench.custom_explainers

valid_explainers = {
    "tarbularShap": xrlbench.custom_explainers.TarbularSHAP,
    "sarfa": xrlbench.custom_explainers.SARFA

}


class Explainer:
    def __init__(self, method, state, action, **kwargs):
        if method not in valid_explainers.keys():
            raise NotImplementedError(
                f"This explainer is not supported at the moment. Explainers supported are {list(valid_explainers.keys())}"
            )
        self.method = method
        self.state = state
        self.action = action
        self.explainer = valid_explainers[method](X=state, y=action, **kwargs)

    def explain(self, state=None):
        if state is None:
            state = self.state
        results = self.explainer.explain(state)
        return results

