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

        self.init_args = dict(
            channels=channels,
            input_size=input_size,
            **tcn_kwargs,
        )

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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.tcn(input)
        output = output[:, -1, :]
        return output

    def __repr__(self) -> str:
        return type(self).__name__
