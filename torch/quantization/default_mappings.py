
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
    nn.ReLU6: nnq.ReLU6,
    nn.Hardswish: nnq.Hardswish,
    nn.ELU: nnq.ELU,
    nn.Conv1d: nnq.Conv1d,
    nn.Conv2d: nnq.Conv2d,
    nn.Conv3d: nnq.Conv3d,
    nn.BatchNorm2d: nnq.BatchNorm2d,
    nn.BatchNorm3d: nnq.BatchNorm3d,
    nn.LayerNorm: nnq.LayerNorm,
    nn.GroupNorm: nnq.GroupNorm,
    nn.InstanceNorm1d: nnq.InstanceNorm1d,
    nn.InstanceNorm2d: nnq.InstanceNorm2d,
    nn.InstanceNorm3d: nnq.InstanceNorm3d,
    QuantStub: nnq.Quantize,
    DeQuantStub: nnq.DeQuantize,
    # Wrapper Modules:
    nnq.FloatFunctional: nnq.QFunctional,
    # Intrinsic modules:
    nni.ConvReLU1d: nniq.ConvReLU1d,
    nni.ConvReLU2d: nniq.ConvReLU2d,
    nni.ConvReLU3d: nniq.ConvReLU3d,
    nni.LinearReLU: nniq.LinearReLU,
    nni.BNReLU2d: nniq.BNReLU2d,
    nni.BNReLU3d: nniq.BNReLU3d,
    nniqat.ConvReLU2d: nniq.ConvReLU2d,
    nniqat.LinearReLU: nniq.LinearReLU,
    nniqat.ConvBn2d: nnq.Conv2d,
    nniqat.ConvBnReLU2d: nniq.ConvReLU2d,
    # QAT modules:
    nnqat.Linear: nnq.Linear,
    nnqat.Conv2d: nnq.Conv2d,
    nnqat.Hardswish: nnq.Hardswish,
    nnqat.GroupNorm: nnq.GroupNorm,
    nnqat.InstanceNorm1d: nnq.InstanceNorm1d,
    nnqat.InstanceNorm2d: nnq.InstanceNorm2d,
    nnqat.InstanceNorm3d: nnq.InstanceNorm3d,
    nnqat.LayerNorm: nnq.LayerNorm,
}

# Map for swapping float module to qat modules
DEFAULT_QAT_MODULE_MAPPING = {
    nn.Linear: nnqat.Linear,
    nn.Conv2d: nnqat.Conv2d,
    nn.Hardswish: nnqat.Hardswish,
    nn.GroupNorm: nnqat.GroupNorm,
    nn.InstanceNorm1d: nnqat.InstanceNorm1d,
    nn.InstanceNorm2d: nnqat.InstanceNorm2d,
    nn.InstanceNorm3d: nnqat.InstanceNorm3d,
    nn.LayerNorm: nnqat.LayerNorm,
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
    nn.LSTMCell: nnqd.LSTMCell,
    nn.RNNCell: nnqd.RNNCell,
    nn.GRUCell: nnqd.GRUCell
}

# Whitelist for propagating the qconfig
_EXCLUDE_QCONFIG_PROPAGATE_LIST = {
    DeQuantStub,
}
_INCLUDE_QCONFIG_PROPAGATE_LIST = {
    nn.Sequential,
}

DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST = (
    (set(DEFAULT_MODULE_MAPPING.keys()) |
     set(DEFAULT_QAT_MODULE_MAPPING.keys()) |
     set(DEFAULT_DYNAMIC_MODULE_MAPPING.keys()) |
     _INCLUDE_QCONFIG_PROPAGATE_LIST) -
    _EXCLUDE_QCONFIG_PROPAGATE_LIST
)

DEFAULT_NUMERIC_SUITE_COMPARE_MODEL_OUTPUT_WHITE_LIST = (
    set(DEFAULT_MODULE_MAPPING.values())
    | set(DEFAULT_QAT_MODULE_MAPPING.values())
    | set(DEFAULT_DYNAMIC_MODULE_MAPPING.values())
    | set(DEFAULT_MODULE_MAPPING.keys())
    | set(DEFAULT_QAT_MODULE_MAPPING.keys())
    | set(DEFAULT_DYNAMIC_MODULE_MAPPING.keys())
    | _INCLUDE_QCONFIG_PROPAGATE_LIST
) - _EXCLUDE_QCONFIG_PROPAGATE_LIST
