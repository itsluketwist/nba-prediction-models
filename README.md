# **nba-prediction-models** 🏀


![check code workflow](https://github.com/itsluketwist/python-template/actions/workflows/check.yaml/badge.svg)


<div>
    <!-- badges from : https://shields.io/ -->
    <!-- logos available : https://simpleicons.org/ -->
    <a href="https://creativecommons.org/licenses/by-sa/4.0/">
        <img alt="CC-BY-SA-4.0 License" src="https://img.shields.io/badge/Licence-CC_BY_SA_4.0-yellow?style=for-the-badge&logo=docs&logoColor=white" />
    </a>
    <a href="https://www.python.org/">
        <img alt="Python 3" src="https://img.shields.io/badge/Python_3-blue?style=for-the-badge&logo=python&logoColor=white" />
    </a>
    <a href="https://pytorch.org/">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-red?style=for-the-badge&logo=pytorch&logoColor=white" />
    </a>
    <a href="https://www.nba.com/">
        <img alt="NBA" src="https://img.shields.io/badge/NBA-black?style=for-the-badge&logo=nba&logoColor=white" />
    </a>
</div>

## *about*

This repository contains data and models to make predictions on upcoming NBA games, 
by modelling the recent form of the teams playing as a sequential dataset of game statistics.

## *structure*

The projects core modules and interfaces are:

- `data/*` module: This is where the sequential datasets for training and evaluation are constructed, from a publicly available database of NBA game statistics.
- `src/*` module: This contains the main code for training and evaluating the models.
- `output/*` module: Contains any output produced from training models.
- `train.ipynb` notebook: Code used to train the final models.
- `predict.ipynb` notebook: Interface to allow new result prediction.

## *installation*

Clone the repository code:

```shell
git clone https://github.com/itsluketwist/nba-prediction-models.git
```

Once cloned, install the requirements locally in a virtual environment:

```shell
pip install uv

uv venv

. venv/bin/activate

uv pip install -r requirements.lock
```

## *development*

Install and use pre-commit to ensure code is in a good state:

```shell
pre-commit install

pre-commit autoupdate

pre-commit run --all-files
```
