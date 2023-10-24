# -*- coding: utf-8 -*-

from xrlbench.explainers import Explainer
from xrlbench.evaluator import Evaluator
from xrlbench.environments import Environment


# Tabular数据测试
def tabular_input_test(environment, method, metric, k=3):
    environment = Environment(environment_name=environment)
    df = environment.get_dataset(generate=False)
    df_sample = df.sample(n=5000, random_state=42)
    action_sample = df_sample['action']
    state_sample = df_sample.drop(['action', 'reward'], axis=1)
    if method == "tabularShap":
        explainer = Explainer(method=method, state=state_sample, action=action_sample)
    else:
        explainer = Explainer(method=method, state=state_sample, action=action_sample, model=environment.model)
    importance = explainer.explain()
    evaluator = Evaluator(metric=metric, environment=environment)
    if metric == "RIS":
        performance = evaluator.evaluate(state_sample, action_sample, importance, explainer=explainer)
    else:
        performance = evaluator.evaluate(state_sample, action_sample, importance, k=k)
    return performance


# Image数据测试
def image_input_test(environment, method, metric, k=50):
    environment = Environment(environment_name=environment)
    dataset = environment.get_dataset(generate=False, data_format="h5")
    explainer = Explainer(method=method, state=dataset.observations, action=dataset.actions,
                          model=environment.model)
    importance = explainer.explain()
    evaluator = Evaluator(metric=metric, environment=environment)
    if metric == "RIS":
        performance = evaluator.evaluate(dataset.observations, dataset.actions, importance, explainer=explainer)
    else:
        performance = evaluator.evaluate(dataset.observations, dataset.actions, importance, k=k)
    return performance


if __name__ == "__main__":
    # Tabular数据测试
    environment = "lunarLander"
    method = "tabularShap"
    metric = "AIM"
    performance = tabular_input_test(environment, method, metric, k=3)

    # Image数据测试
    environment = "breakOut"
    method = "imageDeepShap"
    metric = "imageAIM"
    performance = image_input_test(environment, method, metric, k=50)













