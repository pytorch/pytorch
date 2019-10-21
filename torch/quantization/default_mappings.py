
from torch import nn

import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
import torch.nn.intrinsic.qat as nniqat
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
import torch.nn.qat as nnqat

from .stubs import QuantStub, DeQuantStub

# Map for swapping float module to quantized ones
DEFAULT_MODULE_MAPPING = {
    nn.Linear: nnq.Linear,
    nn.ReLU: nnq.ReLU,
    nn.Conv2d: nnq.Conv2d,
    QuantStub: nnq.Quantize,
    DeQuantStub: nnq.DeQuantize,
    # Wrapper Modules:
    nnq.FloatFunctional: nnq.QFunctional,
    # Intrinsic modules:
    nni.ConvReLU2d: nniq.ConvReLU2d,
    nni.LinearReLU: nniq.LinearReLU,
    nniqat.ConvReLU2d: nniq.ConvReLU2d,
    nniqat.LinearReLU: nniq.LinearReLU,
    nniqat.ConvBn2d: nnq.Conv2d,
    nniqat.ConvBnReLU2d: nniq.ConvReLU2d,
    # QAT modules:
    nnqat.Linear: nnq.Linear,
    nnqat.Conv2d: nnq.Conv2d,
}

# Map for swapping float module to qat modules
DEFAULT_QAT_MODULE_MAPPING = {
    nn.Linear: nnqat.Linear,
    nn.Conv2d: nnqat.Conv2d,
    # Intrinsic modules:
    nni.ConvBn2d: nniqat.ConvBn2d,
    nni.ConvBnReLU2d: nniqat.ConvBnReLU2d,
    nni.ConvReLU2d: nniqat.ConvReLU2d,
    nni.LinearReLU: nniqat.LinearReLU
}

# Map for swapping dynamic modules
DEFAULT_DYNAMIC_MODULE_MAPPING = {
    nn.Linear: nnqd.Linear,
    nn.LSTM: nnqd.LSTM,
}

# Whitelist for propagating the qconfig
_EXCLUDE_QCONFIG_PROPAGATE_LIST = {
    DeQuantStub,
}
_INCLUDE_QCONFIG_PROPAGATE_LIST = {
    nn.Sequential,
}

DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST = (
    set(DEFAULT_MODULE_MAPPING.keys()) |
    set(DEFAULT_QAT_MODULE_MAPPING.keys()) |
    set(DEFAULT_DYNAMIC_MODULE_MAPPING.keys()) |
    _INCLUDE_QCONFIG_PROPAGATE_LIST -
    _EXCLUDE_QCONFIG_PROPAGATE_LIST
)
