import torch
import torch.nn as nn
from numpy.typing import ArrayLike
from pytorch_tcn import TCN as _TCN


class TCN(nn.Module):
    """Class for TCN (temporal convolutional network) model to predict sequential NBA data."""

    def __init__(
        self,
        channels: ArrayLike,
        input_size: int = 116,
        **tcn_kwargs,
    ):
        super(TCN, self).__init__()

        self._channels = channels
        self._input_size = input_size
        self._output_size = 1
        self._num_layers = 1

        self.tcn = _TCN(
            num_inputs=input_size,
            num_channels=channels,
            output_projection=self._output_size,
            output_activation="sigmoid",
            input_shape="NLC",
            **tcn_kwargs,
        )
        # self.linear = nn.Linear(self._hidden_size, self._output_size)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.tcn(input)
        output = output[:, -1, :]
        # output = self.linear(output)
        # output = self.sigmoid(output)
        return output

    def __repr__(self) -> str:
        return type(self).__name__
