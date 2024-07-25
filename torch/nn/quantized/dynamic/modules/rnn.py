# flake8: noqa: F401
r"""Quantized Dynamic Modules.

This file is in the process of migration to `torch/ao/nn/quantized/dynamic`,
and is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/dynamic/modules`,
while adding an import statement here.
"""

from torch.ao.nn.quantized.dynamic.modules.rnn import (
    GRU,
    GRUCell,
    LSTM,
    LSTMCell,
    pack_weight_bias,
    PackedParameter,
    RNNBase,
    RNNCell,
    RNNCellBase,
)


__all__ = [
    "pack_weight_bias",
    "PackedParameter",
    "RNNBase",
    "LSTM",
    "GRU",
    "RNNCellBase",
    "RNNCell",
    "LSTMCell",
    "GRUCell",
]
