# flake8: noqa: F401
r"""Quantized Dynamic Modules

This file is in the process of migration to `torch/ao/nn/quantized/dynamic`,
and is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/dynamic/modules`,
while adding an import statement here.
"""
from torch.ao.nn.quantized.dynamic.modules.rnn import PackedParameter
from torch.ao.nn.quantized.dynamic.modules.rnn import RNNBase
from torch.ao.nn.quantized.dynamic.modules.rnn import LSTM
from torch.ao.nn.quantized.dynamic.modules.rnn import GRU
from torch.ao.nn.quantized.dynamic.modules.rnn import RNNCellBase
from torch.ao.nn.quantized.dynamic.modules.rnn import RNNCell
from torch.ao.nn.quantized.dynamic.modules.rnn import LSTMCell
from torch.ao.nn.quantized.dynamic.modules.rnn import GRUCell
