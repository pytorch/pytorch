import torch
from torch.fx.graph import (
    Node,
)
from ..observer import (
    default_affine_fixed_qparams_observer,
    default_symmetric_fixed_qparams_observer,
)

from ..utils import (
    activation_dtype,
)

from .pattern_utils import (
    register_quant_pattern,
    get_default_output_activation_post_process_map,
    Pattern,
)
from .utils import (
    all_node_args_have_no_tensors,
)

from abc import ABC
import operator
from typing import Any, Callable, Dict, Optional

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
        if isinstance(node, Node):
            self.num_tensor_args = len(node.args)
        else:
            self.num_tensor_args = 0
        self.all_node_args_are_tensors = True

    # TODO: can remove after the is_dynamic flag is defined, so that we can
    # move embedding op to backend_config_dict
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
        or the ops that only re-arranges the Tensor values or query some metadata
        about the Tensor
        so we need to insert observer/fake_quant for the output of the
        operator (same observer instance as input)
        since the distribution of values is different for input and output
        Tensors (for HistogramObserver) while they share the same quantization
        parameters
        Example operator: avgpool2d, reshape, transpose, maxpool2d
        Example observed operator:
        observer_0 - avgpool2d - observer_0 (same observer instance as input)
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

    def is_general_tensor_value_op(self) -> bool:
        return self.num_tensor_args == 1

@register_quant_pattern(torch.cat)
class CatQuantizeHandler(QuantizeHandler):
    def is_general_tensor_value_op(self) -> bool:
        return True

# TODO: remove this class
class ConvReluQuantizeHandler(QuantizeHandler):
    pass

# TODO: remove this class
class LinearReLUQuantizeHandler(QuantizeHandler):
    pass

@register_quant_pattern(torch.nn.BatchNorm2d)
@register_quant_pattern(torch.nn.BatchNorm3d)
@register_quant_pattern(torch.nn.intrinsic.BNReLU2d)
@register_quant_pattern(torch.nn.intrinsic.BNReLU3d)
class BatchNormQuantizeHandler(QuantizeHandler):
    pass

@register_quant_pattern(torch.nn.qat.Embedding)
@register_quant_pattern(torch.nn.qat.EmbeddingBag)
@register_quant_pattern(torch.nn.Embedding)
@register_quant_pattern(torch.nn.EmbeddingBag)
class EmbeddingQuantizeHandler(QuantizeHandler):
    def input_output_observed(self) -> bool:
        return False

# TODO (maybe): merge with embedding quantize handler
@register_quant_pattern(torch.nn.GRUCell)
@register_quant_pattern(torch.nn.LSTMCell)
@register_quant_pattern(torch.nn.RNNCell)
@register_quant_pattern(torch.nn.LSTM)
class RNNDynamicQuantizeHandler(QuantizeHandler):
    pass

# we currently only support reference patterns for these ops so they have been removed
# until they receive a proper fp16 kernel. To use the reference pattern, use a custom qconfig
# @register_quant_pattern(torch.nn.GELU)
# @register_quant_pattern(torch.nn.Softmax)
# we currently only support reference patterns for these ops so they have been removed
# until they receive a proper fp16 kernel. To use the reference pattern, use a custom qconfig
# @register_quant_pattern(torch.nn.functional.gelu)
# @register_quant_pattern(torch.nn.functional.softmax)
class DefaultNodeQuantizeHandler(QuantizeHandler):
    """ Common quantized op, first input and first output will be quantized
    """
    pass

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
    def is_general_tensor_value_op(self) -> bool:
        return True

class StandaloneModuleQuantizeHandler(QuantizeHandler):
    """ Converts an observed standalone module to quantized standalone module
    by calling convert_fx on the observed standalone module.
    """
    pass
