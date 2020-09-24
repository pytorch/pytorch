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
STATIC_QUANT_MODULE_MAPPINGS = {
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

# Map for swapping float module to qat modules
QAT_MODULE_MAPPINGS = {
    nn.Linear: nnqat.Linear,
    nn.Conv2d: nnqat.Conv2d,
    # Intrinsic modules:
    nni.ConvBn2d: nniqat.ConvBn2d,
    nni.ConvBnReLU2d: nniqat.ConvBnReLU2d,
    nni.ConvReLU2d: nniqat.ConvReLU2d,
    nni.LinearReLU: nniqat.LinearReLU
}

# Map for swapping dynamic modules
DYNAMIC_QUANT_MODULE_MAPPINGS = {
    nn.Linear: nnqd.Linear,
    nn.LSTM: nnqd.LSTM,
    nn.LSTMCell: nnqd.LSTMCell,
    nn.RNNCell: nnqd.RNNCell,
    nn.GRUCell: nnqd.GRUCell,
}

# Whitelist for propagating the qconfig
_EXCLUDE_QCONFIG_PROPAGATE_LIST = {
    DeQuantStub,
}
_INCLUDE_QCONFIG_PROPAGATE_LIST = {
    nn.Sequential,
}

# mapping from floating point function or torch ops to quantized ops
FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS = {
    F.elu: torch._ops.ops.quantized.elu,
    F.hardswish: torch._ops.ops.quantized.hardswish,
    F.instance_norm: torch._ops.ops.quantized.instance_norm,
    F.layer_norm: torch._ops.ops.quantized.layer_norm,
}

def register_static_quant_module_mapping(
        float_source_module_class, static_quant_target_module_class):
    ''' Register a mapping from `float_source__module_class` to `static_quant_target_module_class`
    `static_quant_target_module_class` must have from_float defined as a class method
    The mapping is used in the convert step of post training static quantization to
    convert a float module to a statically quantized module.
    '''
    assert hasattr(static_quant_target_module_class, 'from_float'), 'from_float must be defined' + \
        ' in quantized module class'
    STATIC_QUANT_MODULE_MAPPINGS[float_source_module_class] = static_quant_target_module_class

def get_static_quant_module_mappings():
    ''' Get module mapping for post training static quantization
    '''
    return STATIC_QUANT_MODULE_MAPPINGS

def get_static_quant_module_class(float_module_class):
    ''' Get the statically quantized module class corresponding to
    the floating point module class
    '''
    static_quant_module_class = STATIC_QUANT_MODULE_MAPPINGS.get(float_module_class, None)
    assert static_quant_module_class is not None, \
        'Floating point module class {}'.format(float_module_class) + \
        ' does not have a corresponding quantized module class'
    return static_quant_module_class

def register_qat_module_mapping(float_source_module_class, qat_target_module_class):
    '''Register a mapping from `float_source_module_class` to `qat_target_module_class`,
    `qat_target_module_class` must have from_float defined as a class method
    This mapping is used in prepare step of quantization aware training to swap
    a float module to a qat module.
    '''
    assert hasattr(qat_target_module_class, 'from_float'), 'from_float must be defined' + \
        ' in qat module class'
    QAT_MODULE_MAPPINGS[float_source_module_class] = qat_target_module_class

def get_qat_module_mappings():
    ''' Get module mapping for quantization aware training
    '''
    return QAT_MODULE_MAPPINGS

def register_dynamic_quant_module_class(float_source_module_class, dynamic_quant_target_module_class):
    ''' Register a mapping from `float_source_module_class` to `dynamic_quant_target_module_class`,
    `dynamic_quant_target_module_class` must have from_float defined as a class method
    This mapping is used in convert step of post training dynamic
    quantization to swap a float module to a dynamically quantized
    module.
    '''
    assert hasattr(dynamic_quant_target_module_class, 'from_float'), 'from_float must be defined' + \
        ' in dynamically quantized module type'
    DYNAMIC_QUANT_MODULE_MAPPINGS[float_source_module_class] = dynamic_quant_target_module_class

def get_dynamic_quant_module_mappings():
    ''' Get module mapping for post training dynamic quantization
    '''
    return DYNAMIC_QUANT_MODULE_MAPPINGS

def get_qconfig_propagation_list():
    ''' Get the list of module types that we'll attach qconfig
    attribute to in prepare
    '''
    QCONFIG_PROPAGATE_MODULE_CLASS_LIST = (
        (set(STATIC_QUANT_MODULE_MAPPINGS.keys()) |
         set(QAT_MODULE_MAPPINGS.keys()) |
         set(DYNAMIC_QUANT_MODULE_MAPPINGS.keys()) |
         _INCLUDE_QCONFIG_PROPAGATE_LIST) -
        _EXCLUDE_QCONFIG_PROPAGATE_LIST
    )
    return QCONFIG_PROPAGATE_MODULE_CLASS_LIST

def get_compare_output_module_list():
    ''' Get list of module class types that we will record output
    in numeric suite
    '''
    NUMERIC_SUITE_COMPARE_MODEL_OUTPUT_MODULE_LIST = (
        set(STATIC_QUANT_MODULE_MAPPINGS.values())
        | set(QAT_MODULE_MAPPINGS.values())
        | set(DYNAMIC_QUANT_MODULE_MAPPINGS.values())
        | set(STATIC_QUANT_MODULE_MAPPINGS.keys())
        | set(QAT_MODULE_MAPPINGS.keys())
        | set(DYNAMIC_QUANT_MODULE_MAPPINGS.keys())
        | _INCLUDE_QCONFIG_PROPAGATE_LIST
    ) - _EXCLUDE_QCONFIG_PROPAGATE_LIST
    return NUMERIC_SUITE_COMPARE_MODEL_OUTPUT_MODULE_LIST

def register_quantized_operator_mapping(float_op, quantized_op):
    ''' Register a mapping from `floating_point_op` (torch or functional) to `quantized_op`
    This is used in convert step of fx based graph mode quantization
    to convert a float op to quantized op.
    '''
    FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS[float_op] = quantized_op

def get_quantized_operator(float_op):
    ''' Get the quantized operator corresponding to the float operator
    '''
    quantized_op = FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS.get(float_op, None)
    assert quantized_op is not None, \
        'Operator {} does not have corresponding quantized op'.format(float_op)
    return quantized_op
