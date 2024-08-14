from typing import List

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, classification_report
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.utils import device


def training_loop(
    model: Module,
    loader: DataLoader,
    loss_func: _Loss,
    optimizer: Optimizer,
) -> tuple[float, float]:
    """
    Perform a training loop (forward and backward pass) on the given model.

    Parameters
    ----------
    model: Module
    loader: DataLoader
    loss_func: _Loss
    optimizer: Optimizer

    Returns
    -------
    tuple[float, float]
        The training statistics from the loop.
    """
    model.train().to(device)  # model needs to be in evaluation mode
    steps = len(loader)
    data_size = len(loader.dataset)
    total_loss, num_correct = 0, 0

    # loop over the training data set
    for x, y in loader:
        (x, y) = (x.to(device), y.to(device))

        # perform a forward pass and calculate the training loss
        pred = model(x)  # perform forward pass
        y = y.unsqueeze(1).float()
        loss = loss_func(pred, y)  # calculate the loss

        optimizer.zero_grad()  # zero out the gradients
        loss.backward()  # perform backpropagation
        optimizer.step()  # update the weights

        total_loss += loss  # track loss
        num_correct += (
            (pred.round() == y).type(torch.float).sum().item()
        )  # track correctness

    # return the training statistics
    average_loss = total_loss / steps
    accuracy = num_correct / data_size
    return average_loss, accuracy


def testing_loop(
    model: Module,
    loader: DataLoader,
    loss_func: _Loss,
):
    """
    Perform a testing loop (forward pass only) on the given model.

    Parameters
    ----------
    model: Module
    loader: DataLoader
    loss_func: _Loss

    Returns
    -------
    tuple[float, float]
        The testing statistics from the loop.
    """
    model.eval()  # model needs to be in evaluation mode
    steps = len(loader)
    data_size = len(loader.dataset)
    total_loss, num_correct = 0, 0

    # switch off autograd for validation
    with torch.no_grad():
        # loop over the testing data set
        for x, y in loader:
            (x, y) = (x.to(device), y.to(device))
            y = y.unsqueeze(1).float()

            # make the predictions and calculate the validation loss
            pred = model(x)
            total_loss += loss_func(pred, y)  # track loss
            num_correct += (
                (pred.round() == y).type(torch.float).sum().item()
            )  # track correctness

    # return the testing statistics
    average_loss = total_loss / steps
    accuracy = num_correct / data_size
    return average_loss, accuracy


def evaluation_loop(
    model: Module,
    loader: DataLoader,
    classes: List[str] = ["AWAY_WIN", "HOME_WIN"],
    print_report: bool = True,
) -> float:
    """
    Perform evaluation of a model, on the provided data.

    Parameters
    ----------
    model: Module
    loader: DataLoader
    classes: List[str] = ["AWAY_WIN", "HOME_WIN"]
    print_report: bool = True

    Returns
    -------
    float
        The balanced accuracy of predictions on the dataset.
    """
    model.eval().to(device)  # model needs to be in evaluation mode
    y_pred, y_true = [], []

    # turn off autograd for evaluation
    with torch.no_grad():
        # loop over the data
        for x, y in loader:
            (x, y) = (x.to(device), y.to(device))

            # make the predictions and add them to the list
            pred = model(x)
            y_pred.extend(pred.round().flatten().cpu())
            y_true.extend(y.cpu())

    y_pred_np = np.array(y_pred, dtype=np.integer)
    y_true_np = np.array(y_true, dtype=np.integer)

    if print_report:
        # generate a classification report
        print(
            classification_report(
                y_pred=y_pred_np,
                y_true=y_true_np,
                target_names=classes,
                zero_division=0,
            )
        )

    return balanced_accuracy_score(
        y_true=y_true,
        y_pred=y_pred,
    )
