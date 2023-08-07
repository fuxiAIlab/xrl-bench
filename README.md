![image](https://github.com/fuxiAIlab/xrl-bench/blob/main/docs/XRL-Bench.png)

# XRL-Bench: A benchmark for Explainable Reinforcement Learning

**XRL-Bench** is a comprehensive benchmark suite for evaluating eXplainable reinforcement learning (XRL) methods. It provides a unified platform for researches and practitioners to develop, test, and compare their feature importance-based XRL algorithms. XRL-Bench includes a wide range of environments, explainers, and evaluation methods to facilitate the development of state-of-the-art XRL techniques.

## Key Features
- **Environments**: XRL-Bench includes diverse open-source environments based on tabular numerical states and image data states, such as Lunar Lander, CartPole, and Flappy Bird, covering a wide range of reinforcement learning tasks.
- **Explainers**: The benchmark supports various feature importance-based explainers like TabularSHAP, TabularLIME, SARFA, VisualizeSaliency, DeepSHAP and GradientSHAP, to understand the decision-making process of the RL agents.
- **Evaluation Methods**: XRL-Bench provides multiple evaluation methods, including but not limited to Fidelity-focused methods such as AIM, AUM, PGI and PGU, and Stability-focused methods such as RIS. Additionally, XRL-Bench includes human evaluations for more intuitive and comprehensive evaluation results.

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

## Contributing
We welcome contirbutions to XRL-Bench! If you'd like to contirbute, please follow these guidelines:

1. Fork the repository and create a new branch for your feature or bug fix.
2. Add your changes and include tests to ensure your changes work as expected.
3. Update the documentation as necessary.
4. Submit a pull request and wait for a review from the maintainers.

## License
XRL-Bench is under MIT license.


