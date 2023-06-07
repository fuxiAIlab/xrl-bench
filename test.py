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
from xrlbench.datasets import DataSet
import pandas as pd


if __name__ == "__main__":
    # SARFA explain in LunarLander environment
    dataset = DataSet(data_name="lunarLander")
    df = dataset.get_dataset(generate=False)
    df_sample = df.sample(n=5000, random_state=42)
    action_sample = list(df_sample['action'])
    state_sample = df_sample.drop(['action', 'reward'], axis=1)
    explainer = Explainer(method='sarfa', state=state_sample, action=action_sample, model=dataset.agent.qnetwork_local)
    saliency = explainer.explain()
    print(saliency)
    evaluator = Evaluator(metric="AUM", dataset=dataset)
    aum = evaluator.evaluate(state_sample.values, action_sample, saliency, k=5)
    print("accuracy:", aum)

    # TarbularSHAP explain in LunarLander environment
    # dataset = DataSet(data_name="lunarLander")
    # df = dataset.get_dataset(generate=False)
    # action = list(df['action'])
    # state = df.drop(['action', 'reward'], axis=1)
    # explainer = Explainer(method="tarbularShap", state=state, action=action)
    # df_sample = df.sample(n=5000, random_state=42)
    # action_sample = list(df_sample['action'])
    # state_sample = df_sample.drop(['action', 'reward'], axis=1)
    # shap_values = explainer.explain(state_sample)
    # print(shap_values.values)
    # evaluator = Evaluator(metric="PGI", dataset=dataset)
    # aum = evaluator.evaluate(state_sample, action_sample, shap_values.values, k=5)
    # print("accuracy:", aum)



    # LL = LunarLander(load_model=True)
    # df = LL.get_dataset()
    # action = list(df['action'])
    # state = df.drop(['action', 'reward'], axis=1)
    # explainer = TarbularSHAP(X=state, y=action)
    # print(explainer.report)
    # df_sample = df.sample(n=50000, random_state=42)
    # action_sample = list(df_sample['action'])
    # state_sample = df_sample.drop(['action', 'reward'], axis=1)
    # shap_values = explainer.explain(state_sample)
    # metric = Fidelity(dataset=LL)
    # fidelity = metric.evaluate(X=state_sample, y=action_sample, feature_weights=shap_values.values)
    # print("fidelity:", fidelity)

    # LL = LunarLander(load_model=True)
    # df = LL.get_dataset()
    # df_sample = df.sample(n=50000, random_state=42)
    # action_sample = list(df_sample['action'])
    # state_sample = df_sample.drop(['action', 'reward'], axis=1)
    #
    # explainer = SARFA(X=state_sample, y=action_sample, model=LL.agent.qnetwork_local)
    # saliency = explainer.explain()
    #
    # metric = Fidelity(dataset=LL)
    # fidelity = metric.evaluate(X=state_sample.values, y=action_sample, feature_weights=saliency)
    # print("fidelity:", fidelity)