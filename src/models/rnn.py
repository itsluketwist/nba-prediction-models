from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from src.utils import device


class _BaseRNN(nn.Module, ABC):
    """Base class for RNN models."""

    def __init__(self, hidden_size: int, input_size: int = 116, dropout: float = 0.0):
        super(_BaseRNN, self).__init__()

        self._hidden_size = hidden_size
        self._input_size = input_size
        self._dropout = dropout
        self._output_size = 1
        self._num_layers = 1

        self.rnn = self.get_rnn_layer()
        self.linear = nn.Linear(self._hidden_size, self._output_size)
        self.sigmoid = nn.Sigmoid()

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass

    def init_zeroes(self, batch_size: int):
        zeroes = torch.zeros(1, batch_size, self._hidden_size)
        zeroes = zeroes.to(device)
        return zeroes

    def __repr__(self) -> str:
        return type(self).__name__


class RNN(_BaseRNN):
    """Vanilla RNN class."""

    def get_rnn_layer(self) -> nn.Module:
        return nn.RNN(
            input_size=self._input_size,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            batch_first=True,
            dropout=self._dropout,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden = self.init_zeroes(batch_size=input.size(0))
        output, _ = self.rnn(input, hidden)
        output = output[:, -1, :]
        output = self.linear(output)
        output = self.sigmoid(output)
        return output


class LSTM(_BaseRNN):
    """LSTM class."""

    def get_rnn_layer(self) -> nn.Module:
        return nn.LSTM(
            input_size=self._input_size,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            batch_first=True,
            dropout=self._dropout,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden = self.init_zeroes(batch_size=input.size(0))
        state = self.init_zeroes(batch_size=input.size(0))
        output, _ = self.rnn(input, (hidden, state))
        output = output[:, -1, :]
        output = self.linear(output)
        output = self.sigmoid(output)
        return output


class GRU(_BaseRNN):
    """GRU class."""

    def get_rnn_layer(self) -> nn.Module:
        return nn.GRU(
            input_size=self._input_size,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            batch_first=True,
            dropout=self._dropout,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden = self.init_zeroes(batch_size=input.size(0))
        output, _ = self.rnn(input, hidden)
        output = output[:, -1, :]
        output = self.linear(output)
        output = self.sigmoid(output)
        return output
