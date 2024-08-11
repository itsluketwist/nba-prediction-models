import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import Generator, Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split


ALL_GAME_DATA = "data/parquet/complete_df.parquet"
TRAINING_DATA = "data/parquet/training_df.parquet"
EVALUATION_DATA = "data/parquet/evaluation_df_all.parquet"
EVALUATION_DATA_HALF_21_22 = "data/parquet/evaluation_df_half_21_22.parquet"
EVALUATION_DATA_FULL_22_23 = "data/parquet/evaluation_df_full_22_23.parquet"
FINAL_WEEK_DATA = "data/parquet/final_week_df.parquet"
STREAK_DATA_TRAINING_SHORT = "data/parquet/training_streaks_short_df.parquet"
STREAK_DATA_TRAINING_LONG = "data/parquet/training_streaks_long_df.parquet"
STREAK_DATA_EVALUATION_SHORT = "data/parquet/evaluation_streaks_short_df.parquet"
STREAK_DATA_EVALUATION_LONG = "data/parquet/evaluation_streaks_long_df.parquet"


class GameSequenceDataset(Dataset):
    """
    Dataset containing sequences of game data and their results.
    Per-record, later indexes get closer to the game date (0 is oldest game data).
    """

    def __init__(
        self,
        data_file: str,
        verbose: bool = False,
        sequence_len: int = 8,
        normalize: bool = False,
        as_sequence: bool = True,
        **kwargs,
    ):
        self.data = pd.read_parquet(data_file)
        self.verbose = verbose
        self.sequence_len = sequence_len
        self.normalize = normalize
        self.as_sequence = as_sequence
        self.vector_len = 116

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, str | int]:
        item = self.data.iloc[idx]
        sequence = Tensor(item["data"])[-self.sequence_len :]

        if self.normalize:
            sequence = F.normalize(sequence)

        if not self.as_sequence:
            sequence = sequence.reshape([self.sequence_len * self.vector_len])

        info = item["info"]
        result: str | int
        if self.verbose:
            key = "WIN" if info["home_win"] else "LOSS"
            _result = f"Home {key} for {info['home_vs_away']} ({info['game_date']})"
            info["result"] = _result
            result = info
        else:
            result = info["home_win"]

        return sequence, result


def get_train_dataloader(
    train_split: float = 0.8,
    batch_size: int = 64,
    parquet_file: str = TRAINING_DATA,
    dataset_class: Dataset = GameSequenceDataset,
    sequence_len: int = 10,
    as_sequence: bool = True,
    normalize: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Create dataloaders for training a model with NBA game data.

    Parameters
    ----------
    train_split: float = 0.8
    batch_size: int = 64
    parquet_file: str = TRAINING_DATA
    dataset_class: Dataset = GameSequenceDataset
    sequence_len: int = 10
    as_sequence: bool = True
    normalize: bool = True

    Returns
    -------
    tuple[DataLoader, DataLoader]
    """
    raw_data = dataset_class(
        data_file=parquet_file,
        sequence_len=sequence_len,
        as_sequence=as_sequence,
        normalize=normalize,
    )

    num_train = int(len(raw_data) * train_split)
    num_test = len(raw_data) - num_train

    (train_data, test_data) = random_split(
        dataset=raw_data,
        lengths=[num_train, num_test],
        generator=Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        dataset=train_data,
        shuffle=True,
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
    )

    return (train_loader, test_loader)


def get_eval_dataloader(
    parquet_file: str = EVALUATION_DATA,
    dataset_class: Dataset = GameSequenceDataset,
    sequence_len: int = 10,
    as_sequence: bool = True,
    normalize: bool = True,
) -> DataLoader:
    """
    Create a dataloader for evaluating a model with NBA game data.

    Parameters
    ----------
    parquet_file: str = EVALUATION_DATA
    dataset_class: Dataset = GameSequenceDataset
    sequence_len: int = 10
    as_sequence: bool = True
    normalize: bool = True

    Returns
    -------
    DataLoader
    """
    raw_data = dataset_class(
        data_file=parquet_file,
        sequence_len=sequence_len,
        as_sequence=as_sequence,
        normalize=normalize,
    )
    return DataLoader(
        dataset=raw_data,
        batch_size=len(raw_data),
    )


def get_sample_dataloader(
    count: int = 10,
    parquet_file: str = FINAL_WEEK_DATA,
    dataset_class: Dataset = GameSequenceDataset,
    sequence_len: int = 10,
    as_sequence: bool = True,
    normalize: bool = True,
) -> DataLoader:
    """
    Create a dataloader for providing sample NBA game data.

    Parameters
    ----------
    count: int = 10
    parquet_file: str = FINAL_WEEK_DATA
    dataset_class: Dataset = GameSequenceDataset
    sequence_len: int = 10
    as_sequence: bool = True
    normalize: bool = True

    Returns
    -------
    DataLoader
    """
    raw_data = dataset_class(
        data_file=parquet_file,
        verbose=True,
        sequence_len=sequence_len,
        as_sequence=as_sequence,
        normalize=normalize,
    )
    idxs = np.random.choice(range(0, len(raw_data)), size=(count,))
    _subset = Subset(raw_data, idxs)
    return DataLoader(
        dataset=_subset,
        batch_size=1,
    )
