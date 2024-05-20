![image](https://github.com/fuxiAIlab/xrl-bench/blob/main/docs/XRL-Bench.png)

# XRL-Bench: A benchmark for Explainable Reinforcement Learning

**XRL-Bench** is a comprehensive benchmark suite for evaluating eXplainable Reinforcement Learning (XRL) methods. It provides a standard and unified platform for researches and practitioners to develop, test, and compare their state importance-based XRL algorithms. XRL-Bench includes a wide range of environments, explainers, and evaluation methods to facilitate the development of state-of-the-art XRL techniques.

See our KDD [paper](https://arxiv.org/abs/2402.12685) for details and citations.

## Key Features
- **Environments**: XRL-Bench includes diverse open-source environments based on tabular state form, such as [Dunk City Dynasty](https://github.com/FuxiRL/DunkCityDynasty), Lunar Lander, CartPole, and Flappy Bird, and image state form, such as Breakout, Pong, covering a wide range of reinforcement learning tasks.
- **Explainers**: The benchmark supports various state importance-based explainers like TabularSHAP, TabularLIME, SARFA, Perturbation Saliency, Integrated Gradient, DeepSHAP and GradientSHAP, to understand the decision-making process of the RL agents.
- **Evaluation Methods**: XRL-Bench provides multiple evaluation methods, including Fidelity-focused methods such as AIM, AUM, PGI and PGU, and Stability-focused methods such as RIS. 

## Getting Started
To get started with XRL-Bench, follow the instruction below to set up the required dependencies 

```
git clone https://github.com/fuxiAIlab/xrl-bench.git
cd xrl-bench
pip install -r requirements.txt
```
and run the example code

```
from xrlbench.explainers import Explainer
from xrlbench.evaluator import Evaluator
from xrlbench.environments import Environment

env = Environment(environment_name="LunarLander")
dataset = env.get_dataset()
actions = dataset['action']
states = dataset.drop(['action', 'reward'], axis=1)
explainer = Explainer(method="tabularShap", state=states, action=actions)
shap_values = explainer.explain()
evalutor = Evaluator(metric="AIM", environment=env)
aim = evaluator.evaluate(states, actions, shap_values, k=1)
```

## LeaderBoards
We maintain two leaderboards to compare the AUC performance of explainers for both types of state form under various evaluation metrics.
#### Tabular states (DunkCityDynasty)

| Explainer | AIM | AUM | PGI | PGU | RIS |
| --- | --- | --- | --- | --- | --- |
| TabularSHAP | **0.214** | **0.894** | 0.905 | 0.662 | **1.023** |
| DeepSHAP | 0.337 | 0.493 | 0.790 | 0.712 | 1.261 |
| GradientSHAP | 0.326 | 0.523 | 0.766 | 0.690 | 1.419 |
| Integrated Gradient | 0.323 | 0.522 | 0.808 | 0.688 | 1.465 |
| SARFA | 0.361 | 0.709 | 0.952 | 0.665 | 1.071 |
| Perturbation Saliency | 0.364 | 0.687 | 0.951 | 0.680 | 1.357 |
| TabularLIME | 0.215 | 0.779 | **0.954** | **0.274** | 1.901 |

#### Tabular states (LunarLander)

| Explainer | AIM | AUM | PGI | PGU | RIS |
| --- | --- | --- | --- | --- | --- |
| TabularSHAP | **0.116** | **0.693** | 5.258 | 4.895 | 2.646 |
| DeepSHAP | 0.188 | 0.663 | 5.988 | 4.321 | **2.623** |
| GradientSHAP | 0.203 | 0.614 | 5.963 | 4.317 | 3.134 |
| Integrated Gradient | 0.203 | 0.618 | 5.930 | 4.375 | 2.800 |
| SARFA | 0.388 | 0.363 | 4.953 | 5.169 | 6.408 |
| Perturbation Saliency | 0.382 | 0.353 | 4.847 | 5.528 | 4.878 |
| TabularLIME | 0.323 | 0.472 | **6.179** | **3.755** | 2.871 |

#### Image states (Breakout)

| Explainer  | AIM | AUM | PGI | PGU | RIS |
| --- | --- | --- | --- | --- | --- |
| DeepSHAP | **0.162** | 0.630 | 1.748 | **0.347** | 0.375 |
| GradientSHAP | 0.260 | **0.655** | 1.755 | 0.384 | 0.659 |
| Integrated Gradient | 0.292 | 0.652 | **1.812** | 0.364 | **0.109** |
| SARFA | 0.253 | 0.270 | 1.225 | 0.991 | 0.653 |
| Perturbation Saliency | 0.258 | 0.387 | 1.370 | 0.621 | 0.649 |


## Contributing
We welcome contirbutions to XRL-Bench! If you'd like to contirbute, please follow these guidelines:

1. Fork the repository and create a new branch for your feature or bug fix.
2. Add your changes and include tests to ensure your changes work as expected.
3. Update the documentation as necessary.
4. Submit a pull request and wait for a review from the maintainers.

## License
XRL-Bench is under MIT license.


