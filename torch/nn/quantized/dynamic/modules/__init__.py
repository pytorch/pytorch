# flake8: noqa: F401
r"""Quantized Dynamic Modules.

This file is in the process of migration to `torch/ao/nn/quantized/dynamic`,
and is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/dynamic`,
while adding an import statement here.
"""

from torch.ao.nn.quantized.dynamic.modules import conv
from torch.ao.nn.quantized.dynamic.modules import linear
from torch.ao.nn.quantized.dynamic.modules import rnn

from torch.ao.nn.quantized.dynamic.modules.conv import Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
from torch.ao.nn.quantized.dynamic.modules.linear import Linear
from torch.ao.nn.quantized.dynamic.modules.rnn import LSTM, GRU, LSTMCell, RNNCell, GRUCell

__all__ = [
    'Linear',
    'LSTM',
    'GRU',
    'LSTMCell',
    'RNNCell',
    'GRUCell',
    'Conv1d',
    'Conv2d',
    'Conv3d',
    'ConvTranspose1d',
    'ConvTranspose2d',
    'ConvTranspose3d',
]
