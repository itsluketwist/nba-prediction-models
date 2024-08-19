# **nba-prediction-models** 🏀

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

This repository contains historic NBA game data and PyTorch-based machine learning models to make predictions on upcoming NBA games, using the novel approach of modelling the recent form of the teams playing as a sequential dataset of game statistics.

## *structure*

The projects core modules and interfaces are as follows:

- `train.ipynb` notebook: Code used to train the models. The notebook has full instructions, and can be used to perform your own training on the implmented model architectures.
- `predict.ipynb` notebook: Interface to allow new result prediction from the pre-trained models. The notebook gives full instructions, and can be used to define your own dataset for a series of NBA game results, and then make a prediction on an upcoming game.
- `data/*` module: This is where the sequential datasets for training and evaluation are constructed, from a publicly available database of NBA game statistics. Checkout it's [readme](/data/README.md) for more information.
- `src/*` module: This contains the underlying code for training and evaluating the models.
- `output/*` module: Contains any output produced from training, including various pretrained models.

## *installation*

Clone the repository code:

```shell
git clone https://github.com/itsluketwist/nba-prediction-models.git
```

Once cloned, install the requirements locally in a virtual environment:

```shell
python -m venv .venv

. .venv/bin/activate

pip install -r requirements.lock
```

## *development*

Install and use pre-commit to ensure code is in a good state:

```shell
pre-commit install

pre-commit autoupdate

pre-commit run --all-files
```

Use `uv` for dependency management, first add to `requirements.txt`. Then install `uv` and version lock with:

```shell
pip install uv

uv pip compile requirements.txt -o requirements.lock
```
