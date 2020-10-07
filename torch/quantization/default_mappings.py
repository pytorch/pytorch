import torch
from torch import nn

import torch.nn.functional as F
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
    nn.ConvTranspose1d: nnq.ConvTranspose1d,
    nn.ConvTranspose2d: nnq.ConvTranspose2d,
    nn.BatchNorm2d: nnq.BatchNorm2d,
    nn.BatchNorm3d: nnq.BatchNorm3d,
    nn.LayerNorm: nnq.LayerNorm,
    nn.GroupNorm: nnq.GroupNorm,
    nn.InstanceNorm1d: nnq.InstanceNorm1d,
    nn.InstanceNorm2d: nnq.InstanceNorm2d,
    nn.InstanceNorm3d: nnq.InstanceNorm3d,
    nn.Embedding: nnq.Embedding,
    nn.EmbeddingBag: nnq.EmbeddingBag,
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
}

# mapping from floating point function or torch ops to quantized ops
DEFAULT_OPERATOR_MAPPING = {
    F.elu: torch._ops.ops.quantized.elu,
    F.hardswish: torch._ops.ops.quantized.hardswish,
    F.instance_norm: torch._ops.ops.quantized.instance_norm,
    F.layer_norm: torch._ops.ops.quantized.layer_norm,
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
    nn.LSTMCell: nnqd.LSTMCell,
    nn.RNNCell: nnqd.RNNCell,
    nn.GRUCell: nnqd.GRUCell,
}

# Allowed list for propagating the qconfig
_EXCLUDE_QCONFIG_PROPAGATE_LIST = {
    DeQuantStub,
}
_INCLUDE_QCONFIG_PROPAGATE_LIST = {
    nn.Sequential,
}

DEFAULT_QCONFIG_PROPAGATE_ALLOWED_LIST = (
    (set(DEFAULT_MODULE_MAPPING.keys()) |
     set(DEFAULT_QAT_MODULE_MAPPING.keys()) |
     set(DEFAULT_DYNAMIC_MODULE_MAPPING.keys()) |
     _INCLUDE_QCONFIG_PROPAGATE_LIST) -
    _EXCLUDE_QCONFIG_PROPAGATE_LIST
)

DEFAULT_NUMERIC_SUITE_COMPARE_MODEL_OUTPUT_ALLOWED_LIST = (
    set(DEFAULT_MODULE_MAPPING.values())
    | set(DEFAULT_QAT_MODULE_MAPPING.values())
    | set(DEFAULT_DYNAMIC_MODULE_MAPPING.values())
    | set(DEFAULT_MODULE_MAPPING.keys())
    | set(DEFAULT_QAT_MODULE_MAPPING.keys())
    | set(DEFAULT_DYNAMIC_MODULE_MAPPING.keys())
    | _INCLUDE_QCONFIG_PROPAGATE_LIST
) - _EXCLUDE_QCONFIG_PROPAGATE_LIST
