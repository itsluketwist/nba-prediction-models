import csv
import logging

import torch
from torch import Tensor
from torch.nn import Module

from src.utils import device


logger = logging.getLogger(__name__)


def make_prediction(
    model: Module,
    data: Tensor,
) -> list[str]:
    """
    Use a pre-loaded model to make predictions on pre-loaded data.

    Parameters
    ----------
    model: Module
        The pre-loaded pytorch model for prediction.
    data: Tensor
        The pre-loaded sequential game data to predict.

    Returns
    -------
    float
        Predicted probability that the home team will win.
    """
    model.eval()
    model = model.to(device)
    data = data.to(device)

    # switch off autograd for
    with torch.no_grad():
        batch_pred = model(data)

    pred = batch_pred[0][0]

    if pred > 0.5:
        print("The HOME team will win.")
    else:
        print("The AWAY team will win.")

    return pred


def load_record_from_csv(
    file_path: str,
    has_header_row: bool = True,
) -> Tensor:
    """
    Utility function to load game result data from a csv into a Tensor, ready for predictions.

    Parameters
    ----------
    file_path: str
        Path to csv file to load.
    has_header_row: bool = True
        Whether the csv file has a header row.

    Returns
    -------
    Tensor
        PyTorch tensor of data ready to be used for predictions.
    """
    data = []
    with open(file=file_path, mode="r") as csvfile:
        reader = csv.reader(csvfile)

        if has_header_row:
            next(reader, None)  # skip headers

        for row in reader:
            data.append([float(x) for x in row])

    return Tensor([data])
