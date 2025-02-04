r"""Quantized Modules.

Note::
    The `torch.nn.quantized` namespace is in the process of being deprecated.
    Please, use `torch.ao.nn.quantized` instead.
"""

# The following imports are needed in case the user decides
# to import the files directly,
# s.a. `from torch.nn.quantized.modules.conv import ...`.
# No need to add them to the `__all__`.
from torch.ao.nn.quantized.modules import (
    activation,
    batchnorm,
    conv,
    DeQuantize,
    dropout,
    embedding_ops,
    functional_modules,
    linear,
    MaxPool2d,
    normalization,
    Quantize,
    rnn,
    utils,
)
from torch.ao.nn.quantized.modules.activation import (
    ELU,
    Hardswish,
    LeakyReLU,
    MultiheadAttention,
    PReLU,
    ReLU6,
    Sigmoid,
    Softmax,
)
from torch.ao.nn.quantized.modules.batchnorm import BatchNorm2d, BatchNorm3d
from torch.ao.nn.quantized.modules.conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from torch.ao.nn.quantized.modules.dropout import Dropout
from torch.ao.nn.quantized.modules.embedding_ops import Embedding, EmbeddingBag
from torch.ao.nn.quantized.modules.functional_modules import (
    FloatFunctional,
    FXFloatFunctional,
    QFunctional,
)
from torch.ao.nn.quantized.modules.linear import Linear
from torch.ao.nn.quantized.modules.normalization import (
    GroupNorm,
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
    LayerNorm,
)
from torch.ao.nn.quantized.modules.rnn import LSTM


__all__ = [
    "BatchNorm2d",
    "BatchNorm3d",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "DeQuantize",
    "ELU",
    "Embedding",
    "EmbeddingBag",
    "GroupNorm",
    "Hardswish",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LayerNorm",
    "LeakyReLU",
    "Linear",
    "LSTM",
    "MultiheadAttention",
    "Quantize",
    "ReLU6",
    "Sigmoid",
    "Softmax",
    "Dropout",
    "PReLU",
    # Wrapper modules
    "FloatFunctional",
    "FXFloatFunctional",
    "QFunctional",
]
