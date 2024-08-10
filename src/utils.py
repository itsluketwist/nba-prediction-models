import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

import torch


def filename_datetime() -> str:
    """
    Generate a filename friendly datetime string.

    Returns
    -------
    str
        The filename string.
    """
    return datetime.now().strftime("%Y_%m_%d_T%H%M")


def get_device() -> torch.device:
    """
    Get the best available device for training.

    Returns
    -------
    torch.device
    """
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


device = get_device()


@dataclass
class History:
    """Class to hold training history data."""

    model_name: str
    sequence_len: int

    train_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    test_loss: List[float] = field(default_factory=list)
    test_accuracy: List[float] = field(default_factory=list)

    eval_accuracy_half: float = 0.0
    eval_accuracy_next: float = 0.0

    streak_accuracy_train_short: float = 0.0
    streak_accuracy_train_long: float = 0.0
    streak_accuracy_eval_short: float = 0.0
    streak_accuracy_eval_long: float = 0.0

    def update(
        self,
        train_loss: float,
        train_accuracy: float,
        test_loss: float,
        test_accuracy: float,
    ):
        """
        Update the training class with results from a training loop.

        Parameters
        ----------
        train_loss: float
        train_accuracy: float
        test_loss: float
        test_accuracy: float
        """
        self.train_loss.append(float(train_loss))
        self.train_accuracy.append(float(train_accuracy))
        self.test_loss.append(float(test_loss))
        self.test_accuracy.append(float(test_accuracy))


def check_dir(path: str, output: bool = True) -> str:
    """
    Utility to check the directory exists, and to create it if not.

    Parameters
    ----------
    path: str
    output: bool = True

    Returns
    -------
    str
        The given path, once it's sure to exist.
    """
    if output and not path.startswith("output/"):
        path = "output/" + path.lstrip("/")

    if not os.path.exists(path=path):
        os.mkdir(path=path)

    return path
