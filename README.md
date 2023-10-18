# Disentangling Epistemic and Aleatoric Uncertainty in Reinforcement Learning

This repository presents the experiments of the paper:

[Disentangling Epistemic and Aleatoric Uncertainty in Reinforcement Learning](https://arxiv.org/pdf/2206.01558.pdf)<br>
Bertrand Charpentier, Ransalu Senanayake, Mykel Kochenderfer, Stephan Günnemann<br>
Distribution-Free Uncertainty Quantification Workshop (DFUQ - ICML), 2022.

[Paper]((https://arxiv.org/pdf/2206.01558.pdf)

## Setup

To install requirements:
```
conda env create -f environment.yaml
conda activate aleatoric-epistemic-uncertainty-rl
conda env list

python setup.py develop
```
After the setup, it is possible to run the different models in the paper and reproduce the results by using the files `src/policies/dqn/run*.py`. The input parameters of the methods are self-explanatory.
