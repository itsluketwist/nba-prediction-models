import torch
import torch.nn as nn

from src import models
from src.utils import device


def save_model(file_path: str, model: nn.Module) -> None:
    """
    Save a model to file.

    Parameters
    ----------
    file_path: str
    model: nn.Module
    """
    torch.save(
        obj={
            "class": str(model),
            "args": model.init_args,
            "state_dict": model.state_dict(),
        },
        f=file_path,
    )


def load_model(file_path: str) -> nn.Module | None:
    """
    Load a model from file.

    Parameters
    ----------
    file_path: str

    Returns
    -------
    nn.Module | None
        The loaded model, or None invalid.
    """
    model_data = torch.load(file_path, map_location=device)

    model_class: nn.Module
    match model_data["class"]:
        case "RNN":
            model_class = models.RNN
        case "LSTM":
            model_class = models.LSTM
        case "GRU":
            model_class = models.GRU
        case "TCN":
            model_class = models.TCN
        case "TE":
            model_class = models.TE

        case _:
            print(f"Model class {model_data['class']} is not recogised.")
            return None

    model = model_class(**model_data["args"])
    model.load_state_dict(model_data["state_dict"])

    return model
