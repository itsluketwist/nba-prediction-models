import torch
import torch.nn as nn


class TE(nn.Module):
    """Class for TE (transformer encoder) model to predict sequential NBA data."""

    def __init__(
        self,
        input_size: int = 116,
        sequence_len: int = 8,
        hidden_size: int = 512,
        **te_kwargs,
    ):
        super(TE, self).__init__()

        self._input_size = input_size * sequence_len
        self._sequence_len = sequence_len
        self._output_size = 1
        self._num_layers = 1

        self.te = nn.TransformerEncoderLayer(
            d_model=self._input_size,
            nhead=self._sequence_len,
            dim_feedforward=hidden_size,
            batch_first=True,
            **te_kwargs,
        )

        self.linear = nn.Linear(self._input_size, self._output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.te(input)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output

    def __repr__(self) -> str:
        return type(self).__name__
