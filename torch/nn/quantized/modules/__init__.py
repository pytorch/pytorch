r"""Quantized Modules

Note::
    The `torch.nn.quantized` namespace is in the process of being deprecated.
    Please, use `torch.ao.nn.quantized` instead.
"""

from torch.nn.modules.pooling import MaxPool2d

from torch.ao.nn.quantized.modules import activation
from torch.ao.nn.quantized.modules.activation import ReLU6, Hardswish, ELU, LeakyReLU, Sigmoid, Softmax
from torch.ao.nn.quantized.modules import batchnorm
from torch.ao.nn.quantized.modules.batchnorm import BatchNorm2d, BatchNorm3d
from torch.ao.nn.quantized.modules import conv
from torch.ao.nn.quantized.modules.conv import Conv1d, Conv2d, Conv3d
from torch.ao.nn.quantized.modules.conv import ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
from torch.ao.nn.quantized.modules import dropout
from torch.ao.nn.quantized.modules.dropout import Dropout
from torch.ao.nn.quantized.modules import embedding_ops
from torch.ao.nn.quantized.modules.embedding_ops import Embedding, EmbeddingBag
from torch.ao.nn.quantized.modules import functional_modules
from torch.ao.nn.quantized.modules.functional_modules import FloatFunctional, FXFloatFunctional, QFunctional
from torch.ao.nn.quantized.modules import linear
from torch.ao.nn.quantized.modules.linear import Linear
from torch.ao.nn.quantized.modules import normalization
from torch.ao.nn.quantized.modules.normalization import LayerNorm, GroupNorm, InstanceNorm1d, InstanceNorm2d, InstanceNorm3d

from torch.ao.nn.quantized.modules import Quantize, DeQuantize


__all__ = [
    # Subpackages, in case the users import files directly
    # s.a. `from torch.nn.quantized.modules import conv`
    'activation',
    'batchnorm',
    'conv',
    'dropout',
    'embedding_ops',
    'functional_modules',
    'linear',
    'normalization',
    # Modules
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
    'MaxPool2d',
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
