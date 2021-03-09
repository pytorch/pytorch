import copy

import torch
from torch import nn

import torch.nn.functional as F
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
import torch.nn.intrinsic.qat as nniqat
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
import torch.nn.qat as nnqat

from typing import Optional, Union, Dict, Set, Callable, Any

from .stubs import QuantStub, DeQuantStub
from .fake_quantize import (
    default_affine_fixed_qparams_fake_quant,
    default_symmetric_fixed_qparams_fake_quant,
)
from .utils import get_combined_dict

# Default map for swapping float module to quantized ones
DEFAULT_STATIC_QUANT_MODULE_MAPPINGS : Dict[Callable, Any] = {
    QuantStub: nnq.Quantize,
    DeQuantStub: nnq.DeQuantize,
    nn.BatchNorm2d: nnq.BatchNorm2d,
    nn.BatchNorm3d: nnq.BatchNorm3d,
    nn.Conv1d: nnq.Conv1d,
    nn.Conv2d: nnq.Conv2d,
    nn.Conv3d: nnq.Conv3d,
    nn.ConvTranspose1d: nnq.ConvTranspose1d,
    nn.ConvTranspose2d: nnq.ConvTranspose2d,
    nn.ELU: nnq.ELU,
    nn.Embedding: nnq.Embedding,
    nn.EmbeddingBag: nnq.EmbeddingBag,
    nn.GroupNorm: nnq.GroupNorm,
    nn.Hardswish: nnq.Hardswish,
    nn.InstanceNorm1d: nnq.InstanceNorm1d,
    nn.InstanceNorm2d: nnq.InstanceNorm2d,
    nn.InstanceNorm3d: nnq.InstanceNorm3d,
    nn.LayerNorm: nnq.LayerNorm,
    nn.LeakyReLU: nnq.LeakyReLU,
    nn.modules.linear._LinearWithBias: nnq.Linear,
    nn.Linear: nnq.Linear,
    nn.ReLU6: nnq.ReLU6,
    # Wrapper Modules:
    nnq.FloatFunctional: nnq.QFunctional,
    # Intrinsic modules:
    nni.BNReLU2d: nniq.BNReLU2d,
    nni.BNReLU3d: nniq.BNReLU3d,
    nni.ConvReLU1d: nniq.ConvReLU1d,
    nni.ConvReLU2d: nniq.ConvReLU2d,
    nni.ConvReLU3d: nniq.ConvReLU3d,
    nni.LinearReLU: nniq.LinearReLU,
    nniqat.ConvBn1d: nnq.Conv1d,
    nniqat.ConvBn2d: nnq.Conv2d,
    nniqat.ConvBnReLU1d: nniq.ConvReLU1d,
    nniqat.ConvBnReLU2d: nniq.ConvReLU2d,
    nniqat.ConvReLU2d: nniq.ConvReLU2d,
    nniqat.LinearReLU: nniq.LinearReLU,
    # QAT modules:
    nnqat.Linear: nnq.Linear,
    nnqat.Conv2d: nnq.Conv2d,
}

# Default map for swapping float module to qat modules
DEFAULT_QAT_MODULE_MAPPINGS : Dict[Callable, Any] = {
    nn.Conv2d: nnqat.Conv2d,
    nn.Linear: nnqat.Linear,
    nn.modules.linear._LinearWithBias: nnqat.Linear,
    # Intrinsic modules:
    nni.ConvBn1d: nniqat.ConvBn1d,
    nni.ConvBn2d: nniqat.ConvBn2d,
    nni.ConvBnReLU1d: nniqat.ConvBnReLU1d,
    nni.ConvBnReLU2d: nniqat.ConvBnReLU2d,
    nni.ConvReLU2d: nniqat.ConvReLU2d,
    nni.LinearReLU: nniqat.LinearReLU
}

# Default map for swapping dynamic modules
DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS : Dict[Callable, Any] = {
    nn.GRUCell: nnqd.GRUCell,
    nn.Linear: nnqd.Linear,
    nn.modules.linear._LinearWithBias: nnqd.Linear,
    nn.LSTM: nnqd.LSTM,
    nn.GRU: nnqd.GRU,
    nn.LSTMCell: nnqd.LSTMCell,
    nn.RNNCell: nnqd.RNNCell,
}

# Allowlist for propagating the qconfig
_INCLUDE_QCONFIG_PROPAGATE_LIST : Set[Callable] = {
    nn.Sequential,
}

# Default mapping from floating point function or torch ops to quantized ops
# TODO: merge with default static mapping
DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS : Dict[Union[Callable, str], Callable] = {
    F.elu: torch._ops.ops.quantized.elu,
    F.hardswish: torch._ops.ops.quantized.hardswish,
    F.instance_norm: torch._ops.ops.quantized.instance_norm,
    F.layer_norm: torch._ops.ops.quantized.layer_norm,
    F.leaky_relu: torch._ops.ops.quantized.leaky_relu,
}

# mapping from module to output activation post process class
DEFAULT_MODULE_TO_ACT_POST_PROCESS : Dict[Callable, Callable] = {
    nn.Hardsigmoid: default_affine_fixed_qparams_fake_quant,
    nn.Sigmoid: default_affine_fixed_qparams_fake_quant,
    nn.Tanh: default_symmetric_fixed_qparams_fake_quant,
}

def no_observer_set() -> Set[Any]:
    r"""These modules cannot have observers inserted by default."""
    no_observers = set([
        nn.quantizable.LSTM,
        nn.quantizable.MultiheadAttention
    ])
    return no_observers

def get_default_static_quant_module_mappings() -> Dict[Callable, Any]:
    ''' Get module mapping for post training static quantization
    '''
    return copy.deepcopy(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS)

def get_static_quant_module_class(
        float_module_class: Callable,
        additional_static_quant_mapping: Optional[Dict[Callable, Any]] = None) -> Any:
    r"""n Get the statically quantized module class corresponding to
    the floating point module class
    """
    if additional_static_quant_mapping is None:
        additional_static_quant_mapping = {}
    all_mappings = get_combined_dict(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS, additional_static_quant_mapping)
    static_quant_module_class = all_mappings.get(float_module_class, None)
    assert static_quant_module_class is not None, \
        "Floating point module class {}".format(str(float_module_class)) + \
        " does not have a corresponding quantized module class"
    return copy.deepcopy(static_quant_module_class)

def get_dynamic_quant_module_class(
        float_module_class: Callable,
        additional_dynamic_quant_mapping: Optional[Dict[Callable, Any]] = None) -> Any:
    r"""n Get the dynamically quantized module class corresponding to
    the floating point module class
    """
    if additional_dynamic_quant_mapping is None:
        additional_dynamic_quant_mapping = {}
    all_mappings = get_combined_dict(DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS, additional_dynamic_quant_mapping)
    dynamic_quant_module_class = all_mappings.get(float_module_class, None)
    assert dynamic_quant_module_class is not None, \
        "Floating point module class {}".format(str(float_module_class)) + \
        " does not have a corresponding quantized module class"
    return copy.deepcopy(dynamic_quant_module_class)

def get_default_qat_module_mappings() -> Dict[Callable, Any]:
    ''' Get default module mapping for quantization aware training
    '''
    return copy.deepcopy(DEFAULT_QAT_MODULE_MAPPINGS)

def get_default_dynamic_quant_module_mappings() -> Dict[Callable, Any]:
    ''' Get module mapping for post training dynamic quantization
    '''
    return DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS

def get_default_qconfig_propagation_list() -> Set[Callable]:
    ''' Get the default list of module types that we'll attach qconfig
    attribute to in prepare
    '''
    QCONFIG_PROPAGATE_MODULE_CLASS_LIST = (
        (set(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS.keys()) |
         set(DEFAULT_QAT_MODULE_MAPPINGS.keys()) |
         set(DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS.keys()) |
         _INCLUDE_QCONFIG_PROPAGATE_LIST)
    )
    return copy.deepcopy(QCONFIG_PROPAGATE_MODULE_CLASS_LIST)

def get_default_compare_output_module_list() -> Set[Callable]:
    ''' Get list of module class types that we will record output
    in numeric suite
    '''
    NUMERIC_SUITE_COMPARE_MODEL_OUTPUT_MODULE_LIST = (
        set(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS.values())
        | set(DEFAULT_QAT_MODULE_MAPPINGS.values())
        | set(DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS.values())
        | set(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS.keys())
        | set(DEFAULT_QAT_MODULE_MAPPINGS.keys())
        | set(DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS.keys())
        | _INCLUDE_QCONFIG_PROPAGATE_LIST
    )
    return copy.deepcopy(NUMERIC_SUITE_COMPARE_MODEL_OUTPUT_MODULE_LIST)

# TODO: merge with get_static_quant_module_class
def get_quantized_operator(float_op: Union[Callable, str]) -> Callable:
    ''' Get the quantized operator corresponding to the float operator
    '''
    quantized_op = DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS.get(float_op, None)
    assert quantized_op is not None, \
        'Operator {} does not have corresponding quantized op'.format(str(float_op))
    return quantized_op

def _get_special_act_post_process(module: torch.nn.Module) -> Optional[Callable]:
    r""" Get the special activation post process for `module`, this has
    higher priority than the activation post process in `qconfig`
    e.g.
    input: torch.nn.Sigmoid
    output: default_affine_fixed_qparam_fake_quant
    """
    return DEFAULT_MODULE_TO_ACT_POST_PROCESS.get(type(module), None)

def _has_special_act_post_process(module: torch.nn.Module) -> bool:
    return module.training and type(module) in DEFAULT_MODULE_TO_ACT_POST_PROCESS
