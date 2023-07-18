# -*- coding: utf-8 -*-

import xrlbench.custom_explainers

valid_explainers = {
    "tarbularShap": xrlbench.custom_explainers.TabularSHAP,
    "sarfa": xrlbench.custom_explainers.SARFA,
    "visualizeSaliency": xrlbench.custom_explainers.VisualizeSaliency,
    "tarbularLime": xrlbench.custom_explainers.TabularLime,
    "deepShap": xrlbench.custom_explainers.DeepSHAP,
    "gradientShap": xrlbench.custom_explainers.GradientSHAP
}


class Explainer:
    def __init__(self, method, state, action, **kwargs):
        """
        Constructs an instance of an explainer for a given method.

        Parameters:
        -----------
        method : str
            The name of the explainer method to be used. Supported methods are:
            "tabularShap", "sarfa", "visualizeSaliency", "tabularLime", "deepShap", "gradientShap"
        state : numpy.ndarray
            The state array.
        action : numpy.ndarray
            The action array.
        **kwargs :
            Keyword arguments to be passed to the explainer.
        """
        if method not in valid_explainers.keys():
            raise NotImplementedError(
                f"This explainer is not supported at the moment. Explainers supported are {list(valid_explainers.keys())}"
            )
        self.method = method
        self.state = state
        self.action = action
        self.explainer = valid_explainers[method](X=state, y=action, **kwargs)

    def explain(self, state=None):
        """
        Explains the given state using the selected explainer.

        Parameters:
        -----------
        state : numpy.ndarray, optional
            The state array. If None, the state passed to the constructor will be used.

        Returns:
        --------
        A dictionary containing the explanation results.
        """
        if state is None:
            state = self.state
        results = self.explainer.explain(state)
        return results

