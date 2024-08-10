import logging

import torch
from jldc.core import save_jsonl
from torch import nn
from torch.optim import Adam

from src.loader import (
    ALL_GAME_DATA,
    EVALUATION_DATA_FULL_22_23,
    EVALUATION_DATA_HALF_21_22,
    STREAK_DATA_EVALUATION_LONG,
    STREAK_DATA_EVALUATION_SHORT,
    STREAK_DATA_TRAINING_LONG,
    STREAK_DATA_TRAINING_SHORT,
    TRAINING_DATA,
    get_eval_dataloader,
    get_train_dataloader,
)
from src.loop import evaluation_loop, testing_loop, training_loop
from src.plot import plot_history
from src.utils import History, device, filename_datetime


logger = logging.getLogger(__name__)


# default training hyperparameters
DEFAULT_INIT_LEARN_RATE = 1e-3
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_TRAIN_SPLIT = 0.8


def run_train(
    model: nn.Module,
    sequence_len: int,
    init_learning_rate: float = DEFAULT_INIT_LEARN_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    train_split: float = DEFAULT_TRAIN_SPLIT,
    output_path: str = "output",
    save_return: bool = False,
    weight_decay: float = 0.0,
    use_all_data: bool = False,
    data_as_sequence: bool = False,
):
    """
    Train the chosen model, given the hyperparameters.

    Parameters
    ----------
    model: nn.Module
        Model to use for predictions.
    sequence_len: int
        How many games to include in the sequence from the dataset.
    init_learning_rate: float = DEFAULT_INIT_LEARN_RATE
        Learning rate to use for training.
    batch_size: int = DEFAULT_BATCH_SIZE
        Batch size to use during training.
    epochs: int = DEFAULT_EPOCHS
        How many epochs to train for.
    train_split: float = DEFAULT_TRAIN_SPLIT
        What percentage of data to use for training vs. testing.
    output_path: str = "output"
        Location to save the model (.pth file) and learning curves (.png file) after training.
    save_return: bool = False
        Whether or not to save the returned model and history data to file.
    weight_decay: float = 0.0
        The rate of decay for regularization.
    use_all_data: bool = False
        Whether to use all data when training, ir just the training data.
    data_as_sequence: bool = True
        Whether to use the dataset as a sequence of vectors, or single vector.

    Returns
    -------
    tuple[nn.Module, History]:
        Tuple of the trained model, and the History class summarising training.
    """
    logger.info("Loading data...")
    train_loader, test_loader = get_train_dataloader(
        train_split=train_split,
        batch_size=batch_size,
        sequence_len=sequence_len,
        parquet_file=ALL_GAME_DATA if use_all_data else TRAINING_DATA,
        as_sequence=data_as_sequence,
    )

    loss_func = nn.BCELoss()  # init loss function
    opt = Adam(
        model.parameters(),
        lr=init_learning_rate,
        weight_decay=weight_decay,
    )  # init optimizer
    hist = History(
        model_name=str(model),
        sequence_len=sequence_len,
    )  # init history class

    model = model.to(device)

    logger.info("Beginning to train the network...")
    for e in range(0, epochs):
        train_loss, train_accuracy = training_loop(
            model=model,
            loader=train_loader,
            loss_func=loss_func,
            optimizer=opt,
        )
        test_loss, test_accuracy = testing_loop(
            model=model,
            loader=test_loader,
            loss_func=loss_func,
        )
        hist.update(
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            test_loss=test_loss,
            test_accuracy=test_accuracy,
        )

        # print this loops training and testing data
        logger.info("EPOCH: %s/%s", e + 1, epochs)
        logger.info(
            f"Train loss: {train_loss:.4f}, Train accuracy: {train_accuracy:.4f}",
        )
        logger.info(
            f"Val loss: {test_loss:.4f}, Val accuracy: {test_accuracy:.4f}\n",
        )

    model_name = f"{model}_seq{sequence_len}"

    # create a figure from the training data
    plot_history(
        history=hist,
        model_name=model_name,
        save_location=output_path,
    )

    # evaluate the model on the evaluation datasets
    logger.info("Evaluating network...")
    hist.eval_accuracy_half = evaluation_loop(
        model=model,
        loader=get_eval_dataloader(
            parquet_file=EVALUATION_DATA_HALF_21_22,
            sequence_len=sequence_len,
            as_sequence=data_as_sequence,
        ),
    )
    logger.info(f"Accuracy from season remainder: {hist.eval_accuracy_half:.4f}")

    hist.eval_accuracy_next = evaluation_loop(
        model=model,
        loader=get_eval_dataloader(
            parquet_file=EVALUATION_DATA_FULL_22_23,
            sequence_len=sequence_len,
            as_sequence=data_as_sequence,
        ),
    )
    logger.info(f"Accuracy from next season: {hist.eval_accuracy_next:.4f}")

    # evaluate the model on the streak datasets
    hist.streak_accuracy_train_short = evaluation_loop(
        model=model,
        loader=get_eval_dataloader(
            parquet_file=STREAK_DATA_TRAINING_SHORT,
            sequence_len=sequence_len,
            as_sequence=data_as_sequence,
        ),
        print_report=False,
    )
    logger.info(
        f"Accuracy on short streaks (training): {hist.streak_accuracy_train_short:.4f}"
    )

    hist.streak_accuracy_train_long = evaluation_loop(
        model=model,
        loader=get_eval_dataloader(
            parquet_file=STREAK_DATA_TRAINING_LONG,
            sequence_len=sequence_len,
            as_sequence=data_as_sequence,
        ),
        print_report=False,
    )
    logger.info(
        f"Accuracy on long streaks (training): {hist.streak_accuracy_train_long:.4f}"
    )

    hist.streak_accuracy_eval_short = evaluation_loop(
        model=model,
        loader=get_eval_dataloader(
            parquet_file=STREAK_DATA_EVALUATION_SHORT,
            sequence_len=sequence_len,
            as_sequence=data_as_sequence,
        ),
        print_report=False,
    )
    logger.info(
        f"Accuracy on short streaks (evaluation): {hist.streak_accuracy_eval_short:.4f}"
    )

    hist.streak_accuracy_eval_long = evaluation_loop(
        model=model,
        loader=get_eval_dataloader(
            parquet_file=STREAK_DATA_EVALUATION_LONG,
            sequence_len=sequence_len,
            as_sequence=data_as_sequence,
        ),
        print_report=False,
    )
    logger.info(
        f"Accuracy on long streaks (evaluation): {hist.streak_accuracy_eval_long:.4f}"
    )

    # training complete, save results to disk
    if save_return:
        file_prefix = f"{output_path.rstrip('/')}/{model_name}_{filename_datetime()}"
        torch.save(obj=model, f=f"{file_prefix}.pth")
        save_jsonl(file_path=f"{file_prefix}.json", data=hist)

    return model, hist
