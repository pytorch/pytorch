import torch
from torch.fx.graph import (
    Node,
)
from ..observer import (
    default_affine_fixed_qparams_observer,
    default_symmetric_fixed_qparams_observer,
)

from ..utils import (
    get_qconfig_dtypes,
    activation_dtype,
)

from torch.ao.quantization.quantize import (
    is_activation_post_process,
)

from .pattern_utils import (
    register_quant_pattern,
    get_default_output_activation_post_process_map,
    Pattern,
)
from .utils import (
    all_node_args_have_no_tensors,
)

from ..qconfig import QConfigAny

from abc import ABC
import operator
from typing import Any, Callable, Dict, Union, Optional, Tuple, List

# -------------------------
# Pattern Registrations
# -------------------------

# 1. Post Training Static Quantization and Quantization Aware Training Patterns

# Base Pattern Handler
class QuantizeHandler(ABC):
    """ Base handler class for the quantizer patterns
    """
    def __init__(self, node: Node, modules: Dict[str, torch.nn.Module]):
        """ Records pattern information in __init__, which will be used
        in convert
        """
        # this is an indicator of whether all the inputs are Node or not
        # since some op might be quantized differently depending on whether
        # all inputs are tensors or not, e.g. add/mul
        self.num_tensor_args = len(node.args)
        self.all_node_args_are_tensors = True
        # the last node of the matched pattern
        self.last_node = node

    def _maybe_get_last_node_only_observer(
        self,
        modules: Dict[str, torch.nn.Module]
    ) -> Optional[torch.nn.Module]:
        """
        If the last node of the pattern is observed, return the observer
        instance. Otherwise, return None.
        """
        for maybe_obs_node, _ in self.last_node.users.items():
            if maybe_obs_node.op == 'call_module':
                maybe_obs = modules[str(maybe_obs_node.target)]
                if is_activation_post_process(maybe_obs):
                    return maybe_obs
        return None

    def input_output_observed(self) -> bool:
        """
        Returns True if the pattern matched to this qhandler could be
        be observed, and False it it should not be observed.
        """
        return True

    def is_general_tensor_value_op(self) -> bool:
        """
        Returns True if the operator works for both floating point and
        quantized input, and does some computation based on the input Tensor,
        so we need to insert observer/fake_quant for the output of the
        operator since the distribution of values is different for input and output
        Tensors (for HistogramObserver)
        while they share the same quantization parameters
        Example: avgpool2d
        """
        return False

    def is_general_tensor_shape_op(self) -> bool:
        """ Similar to is_general_tensor_value_op, this is a check
        for ops that works for both floating point and quantized input,
        that only re-arranges the Tensor values or query some metadata about the Tensor
        We don't insert observer/fake_quant for the output of these operators
        Example: reshape, transpose, maxpool2d
        """
        return False

    def should_insert_observer_for_output(
        self,
        qconfig: Any,
        model_is_training: bool,
    ) -> bool:
        """
        Returns true if an observer should be inserted for the output of
        the pattern matched to this QuantizeHandler instance during the
        prepare step.
        """
        # TODO(future PR): potentially clean up and deduplicate these
        # mappings.
        return self.all_node_args_are_tensors and self.input_output_observed()

    def should_mark_output_quantized_from_input_quantized_status(
        self,
        qconfig: QConfigAny
    ) -> bool:
        """
        Returns true if after convert, the output of the matched pattern is
        quantized iff the first input is also quantized.
        """
        return False

    def get_activation_ctr(
        self,
        qconfig: Any,
        pattern: Pattern,
        is_training: bool,
    ) -> Optional[Callable]:
        """
        Returns the constructor for the activation observer which should be
        used for the pattern matched to this handler. Some handlers override
        this to a different value than what is specified in the qconfig.
        """
        return qconfig.activation

    def is_output_quantized(self, qconfig):
        """ Returns true if the output node of convert is quantized
        when is_reference is False, we would return float node when a certain dtype
        combination is not supported (since fbgemm/qnnpack only support certain dtype
        combinations), so the output may be float, but when is_reference is True,
        we support all dtype combinations so the output will always be quantized.

        TODO: This is fragile, whether output is quantized should not depend on `is_reference` since
        we want to make sure whether a Tensor is quantized
        should be the same in prepare and convert and is_reference
        is only available in convert currently

        """
        return True

# Binary op configs

# Supported combinations are:
# quant_type | activation (compute_type) | weight
#  static       quint8                      qint8

# tuple (activation_dtype, weight_dtype, compute_dtype)
# these are supported types for common binary ops like add/mul etc.
all_dtypes = [
    (torch.qint8, torch.qint8, None),
    (torch.quint8, torch.qint8, None),
    (torch.float16, torch.float16, None),
]
fp16_dtypes = [
    (torch.float16, torch.float16, None)
]
int8_dtypes = [
    (torch.qint8, torch.qint8, None),
    (torch.quint8, torch.qint8, None),
]
binary_op_supported_dtypes : Dict[Union[Callable, str], List[Tuple[torch.dtype, torch.dtype, None]]] = {
    operator.add: all_dtypes,
    torch.add: all_dtypes,
    operator.mul: all_dtypes,
    torch.mul: all_dtypes,
    torch.bmm: fp16_dtypes,
    torch.sub: fp16_dtypes,
    operator.sub: fp16_dtypes,
    torch.div: fp16_dtypes,
    operator.truediv: fp16_dtypes,
    torch.matmul: int8_dtypes,
}

default_op_supported_dtypes = {
    torch.nn.ConvTranspose1d: int8_dtypes,
    torch.nn.ConvTranspose2d: int8_dtypes,
    torch.nn.ELU: int8_dtypes,
    torch.nn.LeakyReLU: int8_dtypes,
    torch.nn.Hardswish: int8_dtypes,
    torch.nn.InstanceNorm1d: int8_dtypes,
    torch.nn.InstanceNorm2d: int8_dtypes,
    torch.nn.InstanceNorm3d: int8_dtypes,
    torch.nn.LayerNorm: all_dtypes,
    torch.nn.SiLU: fp16_dtypes,
    torch.nn.Mish: fp16_dtypes,
    torch.nn.GELU: int8_dtypes,
    torch.nn.Dropout: int8_dtypes,
    torch.nn.Softmax: int8_dtypes,
    torch.nn.functional.elu: int8_dtypes,
    torch.nn.functional.hardswish: int8_dtypes,
    torch.nn.functional.instance_norm: int8_dtypes,
    torch.nn.functional.layer_norm: all_dtypes,
    torch.nn.functional.leaky_relu: int8_dtypes,
    torch.nn.functional.silu: fp16_dtypes,
    torch.nn.functional.mish: fp16_dtypes,
    torch.nn.functional.gelu: int8_dtypes,
    torch.nn.functional.softmax: int8_dtypes,
    torch.nn.functional.dropout: int8_dtypes,
    torch.sum: fp16_dtypes,
}

QAT_CONV_MODULE_CLASSES = \
    (torch.nn.qat.Conv2d,
     torch.nn.qat.Conv3d,
     torch.nn.intrinsic.qat.ConvBn1d,
     torch.nn.intrinsic.qat.ConvBn2d,
     torch.nn.intrinsic.qat.ConvBn3d,
     torch.nn.intrinsic.qat.ConvBnReLU1d,
     torch.nn.intrinsic.qat.ConvBnReLU2d,
     torch.nn.intrinsic.qat.ConvBnReLU3d,
     torch.nn.intrinsic.qat.ConvReLU2d,
     torch.nn.intrinsic.qat.ConvReLU3d)

@register_quant_pattern(operator.add)
@register_quant_pattern(operator.sub)
@register_quant_pattern(operator.mul)
@register_quant_pattern(operator.truediv)
@register_quant_pattern(torch.add)
@register_quant_pattern(torch.sub)
@register_quant_pattern(torch.mul)
@register_quant_pattern(torch.div)
@register_quant_pattern(torch.bmm)
@register_quant_pattern((torch.nn.ReLU, operator.add))
@register_quant_pattern((torch.nn.ReLU, operator.mul))
@register_quant_pattern((torch.nn.ReLU, torch.add))
@register_quant_pattern((torch.nn.ReLU, torch.mul))
@register_quant_pattern((torch.nn.functional.relu, operator.add))
@register_quant_pattern((torch.nn.functional.relu, operator.mul))
@register_quant_pattern((torch.nn.functional.relu, torch.add))
@register_quant_pattern((torch.nn.functional.relu, torch.mul))
@register_quant_pattern((torch.relu, operator.add))
@register_quant_pattern((torch.relu, operator.mul))
@register_quant_pattern(torch.matmul)
class BinaryOpQuantizeHandler(QuantizeHandler):
    def __init__(
            self,
            node: Node,
            modules: Dict[str, torch.nn.Module]):
        super().__init__(node, modules)
        self.relu_node = None
        if (
            node.op == 'call_function' and
                node.target in (torch.nn.functional.relu, torch.relu)
        ) or (
            node.op == 'call_module' and
                isinstance(modules[str(node.target)], torch.nn.ReLU)
        ):
            self.relu_node = node
            node = node.args[0]  # type: ignore[assignment]
        self.binary_op_node = node
        self.binary_op = node.target

        # determine how many of the first two args are Tensors (versus scalars)
        # this distinguishes things like "x + y" from "x + 2" or "2 + x"
        self.num_tensor_args = 0
        cache_for_no_tensor_check: Dict[Node, bool] = dict()
        for arg_idx in range(len(self.binary_op_node.args)):
            arg = self.binary_op_node.args[arg_idx]
            if isinstance(arg, Node) and (not all_node_args_have_no_tensors(arg, modules, cache_for_no_tensor_check)):
                self.num_tensor_args += 1
        self.all_node_args_are_tensors = \
            (self.num_tensor_args == len(self.binary_op_node.args))

    def should_insert_observer_for_output(
        self,
        qconfig: Any,
        model_is_training: bool,
    ) -> bool:
        """
        Returns true if an observer should be inserted for the output of
        the pattern matched to this QuantizeHandler instance during the
        prepare step.
        """
        dtypes = get_qconfig_dtypes(qconfig)
        if not (self.binary_op in binary_op_supported_dtypes and dtypes in binary_op_supported_dtypes[self.binary_op]):
            return False
        if self.num_tensor_args == 1:
            return True
        elif self.all_node_args_are_tensors and self.input_output_observed():
            return True
        else:
            return False

    def is_general_tensor_value_op(self) -> bool:
        return self.num_tensor_args == 1

    def input_output_observed(self):
        # for x + y where x and y are scalars, we do not observe anything
        return self.num_tensor_args > 0

    def is_output_quantized(self, qconfig):
        dtypes = get_qconfig_dtypes(qconfig)
        return self.binary_op in binary_op_supported_dtypes and \
            dtypes in binary_op_supported_dtypes[self.binary_op]

@register_quant_pattern(torch.cat)
class CatQuantizeHandler(QuantizeHandler):
    def is_general_tensor_value_op(self) -> bool:
        return True

# handle conv, maybe followed by relu
# NB: matching order is reversed, that is we match from the bottom of this list to the beginning
@register_quant_pattern(torch.nn.Conv1d)
@register_quant_pattern(torch.nn.Conv2d)
@register_quant_pattern(torch.nn.Conv3d)
@register_quant_pattern(torch.nn.functional.conv1d)
@register_quant_pattern(torch.nn.functional.conv2d)
@register_quant_pattern(torch.nn.functional.conv3d)
# TODO: add qat.Conv1d
@register_quant_pattern(torch.nn.qat.Conv2d)
@register_quant_pattern(torch.nn.qat.Conv3d)
@register_quant_pattern(torch.nn.intrinsic.ConvReLU1d)
@register_quant_pattern(torch.nn.intrinsic.ConvReLU2d)
@register_quant_pattern(torch.nn.intrinsic.ConvReLU3d)
@register_quant_pattern(torch.nn.intrinsic.qat.ConvBn1d)
@register_quant_pattern(torch.nn.intrinsic.qat.ConvBn2d)
@register_quant_pattern(torch.nn.intrinsic.qat.ConvBn3d)
@register_quant_pattern(torch.nn.intrinsic.qat.ConvBnReLU1d)
@register_quant_pattern(torch.nn.intrinsic.qat.ConvBnReLU2d)
@register_quant_pattern(torch.nn.intrinsic.qat.ConvBnReLU3d)
@register_quant_pattern(torch.nn.intrinsic.qat.ConvReLU2d)
@register_quant_pattern(torch.nn.intrinsic.qat.ConvReLU3d)
@register_quant_pattern((torch.nn.functional.relu, torch.nn.functional.conv1d))
@register_quant_pattern((torch.nn.functional.relu, torch.nn.functional.conv2d))
@register_quant_pattern((torch.nn.functional.relu, torch.nn.functional.conv3d))
@register_quant_pattern((torch.nn.ReLU, torch.nn.functional.conv1d))
@register_quant_pattern((torch.nn.ReLU, torch.nn.functional.conv2d))
@register_quant_pattern((torch.nn.ReLU, torch.nn.functional.conv3d))
# just for error checks
@register_quant_pattern((torch.nn.ReLU, torch.nn.Conv1d))
@register_quant_pattern((torch.nn.ReLU, torch.nn.Conv2d))
@register_quant_pattern((torch.nn.ReLU, torch.nn.Conv3d))
@register_quant_pattern((torch.nn.functional.relu, torch.nn.Conv2d))
@register_quant_pattern((torch.nn.functional.relu, torch.nn.Conv3d))
# TODO: rename Relu -> ReLU to be more consistent with other classes
class ConvReluQuantizeHandler(QuantizeHandler):
    def __init__(self, node: Node, modules: Dict[str, torch.nn.Module]):
        super().__init__(node, modules)
        self.relu_node = None
        if (node.op == 'call_function' and node.target is torch.nn.functional.relu) or \
           (node.op == 'call_module' and isinstance(modules[str(node.target)], torch.nn.ReLU)):
            self.relu_node = node
            node = node.args[0]  # type: ignore[assignment]
        self.conv_node = node
        if node.op == "call_module":
            self.conv = modules[str(self.conv_node.target)]
        elif node.op == "call_function":
            self.conv = node.target  # type: ignore[assignment]

@register_quant_pattern(torch.nn.functional.linear)
@register_quant_pattern(torch.nn.qat.Linear)
@register_quant_pattern(torch.nn.intrinsic.LinearReLU)
@register_quant_pattern(torch.nn.intrinsic.qat.LinearReLU)
@register_quant_pattern((torch.nn.functional.relu, torch.nn.functional.linear))
@register_quant_pattern((torch.nn.ReLU, torch.nn.functional.linear))
@register_quant_pattern(torch.nn.intrinsic.LinearBn1d)
@register_quant_pattern(torch.nn.intrinsic.qat.LinearBn1d)
# for error checks
@register_quant_pattern((torch.nn.ReLU, torch.nn.Linear))
@register_quant_pattern((torch.nn.functional.relu, torch.nn.Linear))
class LinearReLUQuantizeHandler(QuantizeHandler):
    def __init__(
            self,
            node: Node,
            modules: Dict[str, torch.nn.Module]):
        super().__init__(node, modules)
        self.relu_node = None
        if (node.op == 'call_function' and node.target is torch.nn.functional.relu) or \
           (node.op == 'call_module' and isinstance(modules[str(node.target)], torch.nn.ReLU)):
            self.relu_node = node
            node = node.args[0]  # type: ignore[assignment]
        self.linear_node = node
        if node.op == 'call_module':
            self.linear = modules[str(self.linear_node.target)]

@register_quant_pattern(torch.nn.BatchNorm2d)
@register_quant_pattern(torch.nn.BatchNorm3d)
@register_quant_pattern(torch.nn.intrinsic.BNReLU2d)
@register_quant_pattern(torch.nn.intrinsic.BNReLU3d)
class BatchNormQuantizeHandler(QuantizeHandler):
    def __init__(
            self,
            node: Node,
            modules: Dict[str, torch.nn.Module]):
        super().__init__(node, modules)
        assert node.op == 'call_module'
        self.bn_node = node
        self.bn = modules[str(self.bn_node.target)]

@register_quant_pattern(torch.nn.qat.Embedding)
@register_quant_pattern(torch.nn.qat.EmbeddingBag)
@register_quant_pattern(torch.nn.Embedding)
@register_quant_pattern(torch.nn.EmbeddingBag)
class EmbeddingQuantizeHandler(QuantizeHandler):
    def __init__(
            self,
            node: Node,
            modules: Dict[str, torch.nn.Module]):
        super().__init__(node, modules)

    def input_output_observed(self) -> bool:
        return False

# TODO (maybe): merge with embedding quantize handler
@register_quant_pattern(torch.nn.GRUCell)
@register_quant_pattern(torch.nn.LSTMCell)
@register_quant_pattern(torch.nn.RNNCell)
@register_quant_pattern(torch.nn.LSTM)
class RNNDynamicQuantizeHandler(QuantizeHandler):
    def __init__(
            self,
            node: Node,
            modules: Dict[str, torch.nn.Module]):
        super().__init__(node, modules)

@register_quant_pattern(torch.nn.ConvTranspose1d)
@register_quant_pattern(torch.nn.ConvTranspose2d)
@register_quant_pattern(torch.nn.ELU)
@register_quant_pattern(torch.nn.LeakyReLU)
@register_quant_pattern(torch.nn.Hardswish)
@register_quant_pattern(torch.nn.InstanceNorm1d)
@register_quant_pattern(torch.nn.InstanceNorm2d)
@register_quant_pattern(torch.nn.InstanceNorm3d)
@register_quant_pattern(torch.nn.LayerNorm)
@register_quant_pattern(torch.nn.SiLU)
@register_quant_pattern(torch.nn.Mish)
@register_quant_pattern(torch.nn.Dropout)
# we currently only support reference patterns for these ops so they have been removed
# until they receive a proper fp16 kernel. To use the reference pattern, use a custom qconfig
# @register_quant_pattern(torch.nn.GELU)
# @register_quant_pattern(torch.nn.Softmax)
@register_quant_pattern(torch.nn.functional.elu)
@register_quant_pattern(torch.nn.functional.hardswish)
@register_quant_pattern(torch.nn.functional.instance_norm)
@register_quant_pattern(torch.nn.functional.layer_norm)
@register_quant_pattern(torch.nn.functional.leaky_relu)
@register_quant_pattern(torch.nn.functional.silu)
@register_quant_pattern(torch.nn.functional.mish)
@register_quant_pattern(torch.nn.functional.dropout)
# we currently only support reference patterns for these ops so they have been removed
# until they receive a proper fp16 kernel. To use the reference pattern, use a custom qconfig
# @register_quant_pattern(torch.nn.functional.gelu)
# @register_quant_pattern(torch.nn.functional.softmax)
@register_quant_pattern(torch.sum)
class DefaultNodeQuantizeHandler(QuantizeHandler):
    """ Common quantized op, first input and first output will be quantized
    """
    def __init__(
            self,
            node: Node,
            modules: Dict[str, torch.nn.Module]):
        super().__init__(node, modules)
        if node.op == "call_function" or node.op == "call_method":
            self.op = node.target
        elif node.op == "call_module":
            self.op = type(modules[str(node.target)])

    def is_output_quantized(self, qconfig):
        dtypes = get_qconfig_dtypes(qconfig)
        return self.op in default_op_supported_dtypes and \
            dtypes in default_op_supported_dtypes[self.op]

@register_quant_pattern(torch.nn.Hardsigmoid, default_affine_fixed_qparams_observer)
@register_quant_pattern(torch.nn.functional.hardsigmoid, default_affine_fixed_qparams_observer)
@register_quant_pattern('hardsigmoid', default_affine_fixed_qparams_observer)
@register_quant_pattern('hardsigmoid_', default_affine_fixed_qparams_observer)
@register_quant_pattern(torch.nn.Sigmoid, default_affine_fixed_qparams_observer)
@register_quant_pattern(torch.sigmoid, default_affine_fixed_qparams_observer)
@register_quant_pattern('sigmoid', default_affine_fixed_qparams_observer)
@register_quant_pattern('sigmoid_', default_affine_fixed_qparams_observer)
@register_quant_pattern(torch.nn.Tanh, default_symmetric_fixed_qparams_observer)
@register_quant_pattern(torch.tanh, default_symmetric_fixed_qparams_observer)
@register_quant_pattern('tanh', default_symmetric_fixed_qparams_observer)
@register_quant_pattern('tanh_', default_symmetric_fixed_qparams_observer)
class FixedQParamsOpQuantizeHandler(QuantizeHandler):
    def __init__(self,
                 node: Node,
                 modules: Dict[str, torch.nn.Module]):
        super().__init__(node, modules)
        self.node = node

    def should_mark_output_quantized_from_input_quantized_status(
        self,
        qconfig: QConfigAny
    ) -> bool:
        # FixQParamOps are the same as CopyNode in int8 quantization
        return activation_dtype(qconfig) in [torch.quint8, torch.qint8]

    # some qhandlers override the activations constructor
    def get_activation_ctr(self, qconfig, pattern, is_training) -> Optional[Callable]:
        act_dtype = activation_dtype(qconfig)
        if act_dtype == torch.quint8:
            return get_default_output_activation_post_process_map(is_training).get(
                pattern, qconfig.activation)
        else:
            return qconfig.activation

@register_quant_pattern(torch.nn.AdaptiveAvgPool1d)
@register_quant_pattern(torch.nn.AdaptiveAvgPool2d)
@register_quant_pattern(torch.nn.AdaptiveAvgPool3d)
@register_quant_pattern(torch.nn.AvgPool1d)
@register_quant_pattern(torch.nn.AvgPool2d)
@register_quant_pattern(torch.nn.AvgPool3d)
@register_quant_pattern(torch.nn.Hardtanh)
@register_quant_pattern(torch.nn.MaxPool1d)
@register_quant_pattern(torch.nn.MaxPool2d)
@register_quant_pattern(torch.nn.MaxPool3d)
@register_quant_pattern(torch.nn.ReLU)
@register_quant_pattern(torch.nn.ReLU6)
@register_quant_pattern(torch.adaptive_avg_pool1d)
@register_quant_pattern(torch.nn.functional.adaptive_avg_pool2d)
@register_quant_pattern(torch.nn.functional.adaptive_avg_pool3d)
@register_quant_pattern(torch.nn.functional.hardtanh)
@register_quant_pattern(torch.nn.functional.hardtanh_)
@register_quant_pattern(torch.nn.functional.interpolate)
@register_quant_pattern(torch.nn.functional.max_pool1d)
@register_quant_pattern(torch.nn.functional.max_pool2d)
@register_quant_pattern(torch.nn.functional.max_pool3d)
@register_quant_pattern(torch.nn.functional.relu)
@register_quant_pattern(torch.nn.functional.relu6)
@register_quant_pattern(torch.avg_pool1d)
@register_quant_pattern(torch._C._nn.avg_pool2d)
@register_quant_pattern(torch._C._nn.avg_pool3d)
@register_quant_pattern(torch.clamp)
@register_quant_pattern(torch.flatten)
@register_quant_pattern(torch.mean)
@register_quant_pattern(operator.floordiv)
@register_quant_pattern('clamp')
@register_quant_pattern('mean')
@register_quant_pattern('relu')
@register_quant_pattern('relu_')
class CopyNodeQuantizeHandler(QuantizeHandler):
    """ Operators that works on both float and quantized input
    if input is quantized, the output Tensor shares
    the same quantization parameter with input.
    These ops will do computation on the input Tensor, e.g. average pool, so we will
    insert extra observer/fake_quant for the output of these operators.
    TODO: maybe rename this to TensorValueOpQuantizeHandler
    """
    def should_mark_output_quantized_from_input_quantized_status(
        self,
        qconfig: QConfigAny
    ) -> bool:
        return True

    def is_general_tensor_value_op(self) -> bool:
        return True

class CustomModuleQuantizeHandler(QuantizeHandler):
    pass

@register_quant_pattern(torch.nn.Identity)
@register_quant_pattern(torch.transpose)
@register_quant_pattern(torch.repeat_interleave)
@register_quant_pattern(torch.squeeze)
@register_quant_pattern(torch.stack)
@register_quant_pattern(torch.unsqueeze)
@register_quant_pattern('contiguous')
@register_quant_pattern('detach')
@register_quant_pattern('detach_')
@register_quant_pattern('permute')
@register_quant_pattern('repeat')
@register_quant_pattern('repeat_interleave')
@register_quant_pattern('reshape')
@register_quant_pattern('resize_')
@register_quant_pattern('shape')
@register_quant_pattern('size')
@register_quant_pattern('squeeze')
@register_quant_pattern('squeeze_')
@register_quant_pattern('transpose')
@register_quant_pattern('unsqueeze')
@register_quant_pattern('unsqueeze_')
@register_quant_pattern('view')
class GeneralTensorShapeOpQuantizeHandler(QuantizeHandler):
    """ Operators that works on both float and quantized input
    if input is quantized, the output Tensor shares
    the same quantization parameter with input.
    These ops only do rearrangement of Tensor values, for
    example reshape, or just query the information about Tensor
    e.g. size, and we do not insert extra observer/fake_quant
    for the output of the operator.
    """
    def is_general_tensor_shape_op(self) -> bool:
        return True

    def should_mark_output_quantized_from_input_quantized_status(
        self,
        qconfig: QConfigAny
    ) -> bool:
        return True

class StandaloneModuleQuantizeHandler(QuantizeHandler):
    """ Converts an observed standalone module to quantized standalone module
    by calling convert_fx on the observed standalone module.
    """
    pass
