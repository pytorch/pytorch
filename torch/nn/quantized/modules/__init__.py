r"""Quantized Modules

Note::
    The `torch.nn.quantized` namespace is in the process of being deprecated.
    Please, use `torch.ao.nn.quantized` instead.
"""

from torch.ao.nn.quantized.modules.activation import ReLU6, Hardswish, ELU, LeakyReLU, Sigmoid, Softmax, MultiheadAttention
from torch.ao.nn.quantized.modules.batchnorm import BatchNorm2d, BatchNorm3d
from torch.ao.nn.quantized.modules.conv import Conv1d, Conv2d, Conv3d
from torch.ao.nn.quantized.modules.conv import ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
from torch.ao.nn.quantized.modules.dropout import Dropout
from torch.ao.nn.quantized.modules.embedding_ops import Embedding, EmbeddingBag
from torch.ao.nn.quantized.modules.functional_modules import FloatFunctional, FXFloatFunctional, QFunctional
from torch.ao.nn.quantized.modules.linear import Linear
from torch.ao.nn.quantized.modules.normalization import LayerNorm, GroupNorm, InstanceNorm1d, InstanceNorm2d, InstanceNorm3d
from torch.ao.nn.quantized.modules.rnn import LSTM

from torch.ao.nn.quantized.modules import MaxPool2d
from torch.ao.nn.quantized.modules import Quantize, DeQuantize

# The following imports are needed in case the user decides
# to import the files directly,
# s.a. `from torch.nn.quantized.modules.conv import ...`.
# No need to add them to the `__all__`.
from . import activation
from . import batchnorm
from . import conv
from . import dropout
from . import embedding_ops
from . import functional_modules
from . import linear
from . import normalization

__all__ = [
    'BatchNorm2d',
    'BatchNorm3d',
    'Conv1d',
    'Conv2d',
    'Conv3d',
    'ConvTranspose1d',
    'ConvTranspose2d',
    'ConvTranspose3d',
    'DeQuantize',
    'ELU',
    'Embedding',
    'EmbeddingBag',
    'GroupNorm',
    'Hardswish',
    'InstanceNorm1d',
    'InstanceNorm2d',
    'InstanceNorm3d',
    'LayerNorm',
    'LeakyReLU',
    'Linear',
    'LSTM',
    'MaxPool2d',
    'MultiheadAttention',
    'Quantize',
    'ReLU6',
    'Sigmoid',
    'Softmax',
    'Dropout',
    # Wrapper modules
    'FloatFunctional',
    'FXFloatFunctional',
    'QFunctional',
]
