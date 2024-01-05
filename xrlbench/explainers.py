# -*- coding: utf-8 -*-

import xrlbench.custom_explainers

valid_explainers = {
    "tabularShap": xrlbench.custom_explainers.TabularSHAP,
    "sarfa": xrlbench.custom_explainers.SARFA,
    "perturbationSaliency": xrlbench.custom_explainers.PerturbationSaliency,
    "tabularLime": xrlbench.custom_explainers.TabularLime,
    "deepShap": xrlbench.custom_explainers.DeepSHAP,
    "gradientShap": xrlbench.custom_explainers.GradientSHAP,
    "integratedGradient": xrlbench.custom_explainers.IntegratedGradient,
    "imagePerturbationSaliency": xrlbench.custom_explainers.ImagePerturbationSaliency,
    "imageSarfa": xrlbench.custom_explainers.ImageSARFA,
    "imageDeepShap": xrlbench.custom_explainers.ImageDeepSHAP,
    "imageGradientShap": xrlbench.custom_explainers.ImageGradientSHAP,
    "imageIntegratedGradient": xrlbench.custom_explainers.ImageIntegratedGradient
}


class Explainer:
    def __init__(self, method, state, action, **kwargs):
        """
        Constructs an instance of an explainer for a given method.

        Parameters:
        -----------
        method : str
            The name of the explainer method to be used. Supported methods are:
            "tabularShap", "sarfa", "perturbationSaliency", "tabularLime", "deepShap", "gradientShap", "integratedGradient", "integratedGradient", "imageSarfa", imageDeepShap", "imageGradientShap", "imageIntegratedGradient"
        state : numpy.ndarray
            The state array.
        action : numpy.ndarray
            The action array.
        model : pytorch model, default=None
            The trained reinforcement learning model.
        categorical_names : list, optional (default=None)
            A list of names of categorical features in X.
        **kwargs :
            Keyword arguments to be passed to the explainer.
        """
        if method not in valid_explainers.keys():
            raise NotImplementedError(
                f"This explainer is not supported at the moment. Explainers supported are {list(valid_explainers.keys())}"
            )
        self.method = method
        self.state = state/255 if method.find("image") != -1 else state
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

