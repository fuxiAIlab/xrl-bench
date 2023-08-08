# -*- coding: utf-8 -*-
"""
@Time ： 2023/5/30 19:08
@Auth ： Yu Xiong
@File ：test.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""

from xrlbench.explainers import Explainer
from xrlbench.evaluator import Evaluator
from xrlbench.environments import Environment
import pandas as pd


if __name__ == "__main__":
    # environment = Environment(environment_name="cartPole")
    # environment.train_model()
    # df = environment.get_dataset(generate=True)
    # print(df.info())

    # TarbularSHAP explain in LunarLander environment
    environment = Environment(environment_name="lunarLander")
    # environment.train_model()
    df = environment.get_dataset(generate=False)
    df_sample = df.sample(n=5000, random_state=42)
    action_sample = df_sample['action']
    state_sample = df_sample.drop(['action', 'reward'], axis=1)
    explainer = Explainer(method='tabularShap', state=state_sample, action=action_sample)
    saliency = explainer.explain()
    evaluator = Evaluator(metric="AUM", environment=environment)
    aum = evaluator.evaluate(state_sample, action_sample, saliency, k=4)
    print("accuracy:", aum)

    # SARFA explain in LunarLander environment
    # environment = Environment(environment_name="lunarLander")
    # df = environment.get_dataset(generate=False)
    # df_sample = df.sample(n=5000, random_state=42)
    # action_sample = df_sample['action']
    # state_sample = df_sample.drop(['action', 'reward'], axis=1)
    # explainer = Explainer(method='sarfa', state=state_sample, action=action_sample, model=environment.model)
    # saliency = explainer.explain()
    # print(saliency)
    # evaluator = Evaluator(metric="RIS", environment=environment)
    # aum = evaluator.evaluate(state_sample, action_sample, saliency, explainer=explainer)
    # print("accuracy:", aum)

    # VisualizeSaliency explain in LunarLander environment
    # environment = Environment(environment_name="lunarLander")
    # df = environment.get_dataset(generate=False)
    # df_sample = df.sample(n=5000, random_state=42)
    # action_sample = df_sample['action']
    # state_sample = df_sample.drop(['action', 'reward'], axis=1)
    # explainer = Explainer(method='visualizeSaliency', state=state_sample, action=action_sample, model=environment.model)
    # saliency = explainer.explain()
    # print(saliency)
    # evaluator = Evaluator(metric="AUM", environment=environment)
    # aum = evaluator.evaluate(state_sample, action_sample, saliency, k=1)
    # print("accuracy:", aum)

    # TabularLime explain in LunarLander environment
    # environment = Environment(environment_name="lunarLander")
    # df = environment.get_dataset(generate=False)
    # df_sample = df.sample(n=5000, random_state=42)
    # action_sample = df_sample['action']
    # state_sample = df_sample.drop(['action', 'reward'], axis=1)
    # explainer = Explainer(method='tabularLime', state=state_sample, action=action_sample,
    #                       model=environment.model)
    # saliency = explainer.explain()
    # print(saliency)
    # evaluator = Evaluator(metric="AUM", environment=environment)
    # aum = evaluator.evaluate(state_sample, action_sample, saliency, k=1)
    # print("accuracy:", aum)

    # GradientSHAP explain in LunarLander environment
    # environment = Environment(environment_name="lunarLander")
    # df = environment.get_dataset(generate=False)
    # df_sample = df.sample(n=5000, random_state=42)
    # action_sample = df_sample['action']
    # state_sample = df_sample.drop(['action', 'reward'], axis=1)
    # explainer = Explainer(method='gradientShap', state=state_sample, action=action_sample,
    #                       model=environment.model)
    # saliency = explainer.explain()
    # print(saliency)
    # evaluator = Evaluator(metric="AUM", environment=environment)
    # aum = evaluator.evaluate(state_sample, action_sample, saliency, k=1)
    # print("accuracy:", aum)
