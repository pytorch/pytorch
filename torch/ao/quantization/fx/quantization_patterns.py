import torch
from torch.fx import GraphModule
from torch.fx.graph import (
    Node,
    Graph,
)
from ..observer import (
    default_affine_fixed_qparams_observer,
    default_symmetric_fixed_qparams_observer,
)

from ..quantization_mappings import (
    get_static_quant_module_class,
    get_dynamic_quant_module_class,
    get_quantized_operator,
)
from ..utils import (
    get_swapped_custom_module_class,
    activation_is_statically_quantized,
    activation_is_int8_quantized,
    weight_is_statically_quantized,
    get_qconfig_dtypes,
    activation_dtype,
    get_qparam_dict,
    check_node,
)

from torch.ao.quantization.quantize import (
    is_activation_post_process,
)

from .pattern_utils import (
    register_quant_pattern,
    get_default_output_activation_post_process_map,
    Pattern,
)
from ..utils import _parent_name
from .utils import (
    all_node_args_have_no_tensors,
    quantize_node,
    get_per_tensor_qparams,
    get_linear_prepack_op_for_dtype,
    create_qparam_nodes,
    get_qconv_prepack_op,
    get_qconv_op,
    create_node_from_old_node_preserve_meta,
)

from ..qconfig import QConfigAny

from abc import ABC
import operator
import warnings

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

    def convert(self,
                node: Node,
                qconfig: QConfigAny,
                modules: Dict[str, torch.nn.Module],
                quantized_graph: Graph,
                node_name_to_scope: Dict[str, Tuple[str, type]],
                load_arg: Callable,
                is_reference: bool = False,
                convert_custom_config_dict: Dict[str, Any] = None) -> Node:
        """ Convert the given node to a quantized node and insert
        it to the quantized graph
        """
        return NotImplemented


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
     torch.nn.intrinsic.qat.ConvBn2d,
     torch.nn.intrinsic.qat.ConvBnReLU2d,
     torch.nn.intrinsic.qat.ConvReLU2d,
     torch.nn.intrinsic.qat.ConvBn3d,
     torch.nn.intrinsic.qat.ConvBnReLU3d,
     torch.nn.intrinsic.qat.ConvReLU3d)


##########################
# Helper Functions
##########################

def _load_weight_qparams(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs):
    key = prefix + "_weight_qparams"
    if key in state_dict:
        self._weight_qparams = state_dict[key]
        state_dict.pop(key)

def _save_weight_qparams(self, destination, prefix, keep_vars):
    for attr_name in dir(self):
        if "_weight_qparams" == attr_name and \
           isinstance(getattr(self, attr_name), dict):
            weight_qparams = getattr(self, attr_name)
            destination[prefix + attr_name] = weight_qparams


def _to_reference(float_module, weight_qparams):
    """ Make a weighted float module (e.g. conv and linear )a reference module by
    attaching _weight_qparams that records the qparams for weight
    and change the name for the module so that it's recognized
    when people print the model
    """
    float_module._weight_qparams = weight_qparams
    float_module._register_state_dict_hook(_save_weight_qparams)
    float_module._register_load_state_dict_pre_hook(_load_weight_qparams, with_module=True)

    float_module_name = float_module._get_name()

    def _get_name():
        return float_module_name + "(Reference)"

    float_module._get_name = _get_name

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
class BinaryOpQuantizeHandler(QuantizeHandler):
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

        qbin_op_mapping: Dict[Union[Callable, str], Callable] = {
            operator.add: torch.ops.quantized.add,
            torch.add: torch.ops.quantized.add,
            operator.mul: torch.ops.quantized.mul,
            torch.mul: torch.ops.quantized.mul,
        }
        qbin_relu_op_mapping: Dict[Union[Callable, str], Callable] = {
            operator.add: torch.ops.quantized.add_relu,
            torch.add: torch.ops.quantized.add_relu,
            operator.mul: torch.ops.quantized.mul_relu,
            torch.mul: torch.ops.quantized.mul_relu,
        }
        # corresponding quantized op
        self.quantized_binary_op: Optional[Callable] = None
        if self.binary_op in qbin_op_mapping:
            self.quantized_binary_op = qbin_relu_op_mapping[self.binary_op] \
                if self.relu_node is not None \
                else qbin_op_mapping[self.binary_op]

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

    def convert(self,
                node: Node,
                qconfig: QConfigAny,
                modules: Dict[str, torch.nn.Module],
                quantized_graph: Graph,
                node_name_to_scope: Dict[str, Tuple[str, type]],
                load_arg: Callable,
                is_reference: bool = False,
                convert_custom_config_dict: Dict[str, Any] = None) -> Node:

        if self.num_tensor_args == 0:
            # example: x + y, when x and y are scalars
            return quantized_graph.node_copy(
                node, load_arg(quantized=None))

        dtypes = get_qconfig_dtypes(qconfig)

        if is_reference:
            act_dtype = activation_dtype(qconfig)
            dtypes = get_qconfig_dtypes(qconfig)
            if act_dtype == torch.float or \
               not (self.binary_op in binary_op_supported_dtypes and dtypes in binary_op_supported_dtypes[self.binary_op]):
                return quantized_graph.node_copy(node, load_arg(quantized=torch.float))
            else:
                if self.num_tensor_args == 2:
                    # make sure both inputs are quantized to act_dtype
                    load_arg(quantized={0: act_dtype, 1: act_dtype})(self.binary_op_node.args)
                args = load_arg(quantized=torch.float)(self.binary_op_node.args)
                kwargs = load_arg(quantized=torch.float)(self.binary_op_node.kwargs)
                op_out = quantized_graph.node_copy(self.binary_op_node, load_arg(quantized=torch.float))

                def modified_load_arg(n: Node):
                    if n.name == self.binary_op_node.name:
                        return op_out
                    else:
                        return load_arg(quantized=torch.float)(n)

                if self.relu_node:
                    op_out = quantized_graph.node_copy(self.relu_node, modified_load_arg)
                activation_post_process = \
                    self._maybe_get_last_node_only_observer(modules)
                assert activation_post_process is not None
                return quantize_node(
                    op_out, activation_post_process,
                    node, modules, quantized_graph, node_name_to_scope, is_input=False)
        elif not is_reference and self.binary_op in binary_op_supported_dtypes and \
                dtypes in binary_op_supported_dtypes[self.binary_op]:
            if dtypes in [(torch.quint8, torch.qint8, None)]:
                assert self.quantized_binary_op is not None
                if self.num_tensor_args == 1:
                    # add/mul scalar
                    first_arg = self.binary_op_node.args[0]
                    cache_for_no_tensor_check: Dict[Node, bool] = dict()
                    if isinstance(first_arg, Node) and (
                            not all_node_args_have_no_tensors(
                                first_arg, modules, cache_for_no_tensor_check)):
                        quantized_index = 0
                    else:
                        quantized_index = 1

                    return create_node_from_old_node_preserve_meta(
                        quantized_graph,
                        (
                            'call_function', self.quantized_binary_op,
                            load_arg(quantized=[quantized_index])(self.binary_op_node.args),
                            self.binary_op_node.kwargs
                        ),
                        self.binary_op_node)
                else:
                    activation_post_process = \
                        self._maybe_get_last_node_only_observer(modules)
                    assert activation_post_process is not None
                    scale, zero_point = activation_post_process.calculate_qparams()  # type: ignore[operator]
                    scale = float(scale)
                    zero_point = int(zero_point)
                    scale_arg, zero_point_arg = \
                        create_qparam_nodes(
                            node.name, scale, zero_point, modules,
                            quantized_graph, node_name_to_scope)
                    kwargs = {**self.binary_op_node.kwargs}
                    add_args = (*load_arg(quantized=activation_dtype(qconfig))(self.binary_op_node.args), scale_arg, zero_point_arg)
                    op = create_node_from_old_node_preserve_meta(
                        quantized_graph,
                        ('call_function', self.quantized_binary_op, add_args, kwargs),
                        self.binary_op_node)
                    return op
            else:
                assert dtypes == (torch.float16, torch.float16, None)
                # TODO (refactor) this is duplicated, maybe have a helper function
                if self.relu_node:
                    op_out = quantized_graph.node_copy(self.binary_op_node, load_arg(quantized=torch.float))
                    relu_args = [op_out]
                    relu_args.extend(load_arg(quantized=torch.float)(self.relu_node.args[1:]))
                    relu_kwargs = load_arg(quantized=torch.float)(self.relu_node.kwargs)
                    op_out = create_node_from_old_node_preserve_meta(
                        quantized_graph,
                        ("call_function", torch.nn.functional.relu, tuple(relu_args), relu_kwargs),
                        self.relu_node)
                else:
                    op_out = quantized_graph.node_copy(node, load_arg(quantized=torch.float))
                return quantized_graph.create_node(
                    "call_method", "to", (op_out, torch.float16,), {}
                )
        else:
            # leave the op unquantized if the dtype,reference combination is not supported
            warnings.warn(
                "dtype combination: {} is not "
                "supported by {} for is_reference={}. "
                "Supported non-reference dtype combinations are: {} "
                "".format(dtypes,
                          self.binary_op,
                          is_reference,
                          binary_op_supported_dtypes[self.binary_op]
                          )
            )
            if self.relu_node:
                op_out = quantized_graph.node_copy(self.binary_op_node, load_arg(quantized=torch.float))
                relu_args = [op_out]
                relu_args.extend(load_arg(quantized=torch.float)(self.relu_node.args[1:]))
                relu_kwargs = load_arg(quantized=torch.float)(self.relu_node.kwargs)
                return create_node_from_old_node_preserve_meta(
                    quantized_graph,
                    ("call_function", torch.nn.functional.relu, tuple(relu_args), relu_kwargs),
                    self.relu_node)
            else:
                return quantized_graph.node_copy(node, load_arg(quantized=torch.float))


@register_quant_pattern(torch.cat)
class CatQuantizeHandler(QuantizeHandler):
    def is_general_tensor_value_op(self) -> bool:
        return True

    def convert(self,
                node: Node,
                qconfig: QConfigAny,
                modules: Dict[str, torch.nn.Module],
                quantized_graph: Graph,
                node_name_to_scope: Dict[str, Tuple[str, type]],
                load_arg: Callable,
                is_reference: bool = False,
                convert_custom_config_dict: Dict[str, Any] = None) -> Node:
        if not self.all_node_args_are_tensors:
            return NotImplemented
        act_dtype = activation_dtype(qconfig)
        if act_dtype == torch.float:
            op_out = quantized_graph.node_copy(node, load_arg(quantized=torch.float))
            return op_out
        else:
            activation_post_process = \
                self._maybe_get_last_node_only_observer(modules)
            assert activation_post_process is not None
            # make sure the first argument is quantized to act_dtype
            load_arg(quantized={0: act_dtype})(node.args)
            args = list(load_arg(quantized=torch.float)(node.args))
            kwargs = load_arg(quantized=torch.float)(node.kwargs)
            op_out = quantized_graph.node_copy(node, load_arg(quantized=torch.float))
            return quantize_node(
                op_out,
                activation_post_process,
                node,
                modules,
                quantized_graph,
                node_name_to_scope,
                is_input=False)

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

    def convert(self,
                node: Node,
                qconfig: QConfigAny,
                modules: Dict[str, torch.nn.Module],
                quantized_graph: Graph,
                node_name_to_scope: Dict[str, Tuple[str, type]],
                load_arg: Callable,
                is_reference: bool = False,
                convert_custom_config_dict: Dict[str, Any] = None) -> Node:
        # Supported combinations are:
        # quant_type | activation (compute_type) | weight
        #  static       quint8                      qint8

        # tuple (activation_dtype, weight_dtype, compute_dtype)
        supported_dtypes = [
            (torch.quint8, torch.qint8, None),
        ]

        # TODO: is_reference option for conv module
        dtypes = get_qconfig_dtypes(qconfig)
        # leave the op unquantized if the dtype combination is not supported
        if not is_reference and dtypes not in supported_dtypes:
            warnings.warn(
                "dtype combination: {} is not "
                "supported by Conv "
                "supported dtype combinations are: {}".format(dtypes, supported_dtypes))
            if self.relu_node:
                conv_out = quantized_graph.node_copy(self.conv_node, load_arg(quantized=torch.float))
                relu_args = [conv_out]
                relu_args.extend(load_arg(quantized=torch.float)(self.relu_node.args[1:]))
                relu_kwargs = load_arg(quantized=torch.float)(self.relu_node.kwargs)
                return create_node_from_old_node_preserve_meta(
                    quantized_graph,
                    ("call_function", torch.nn.functional.relu, tuple(relu_args), relu_kwargs),
                    self.relu_node)
            else:
                return quantized_graph.node_copy(node, load_arg(quantized=torch.float))

        activation_int8_quantized = activation_is_int8_quantized(qconfig)

        if self.conv_node.op == 'call_module':
            # note that relu should already be fused into conv module in the fusion step
            assert self.relu_node is None, 'conv module and relu fusion is not executed, ' \
                'please make sure to run fusion before prepare'
            output_activation_post_process = \
                self._maybe_get_last_node_only_observer(modules)
            assert output_activation_post_process is not None

            # We'll always produce reference pattern for torch.nn.Conv*d,
            # will remove the else branch after we migrated all use cases
            if is_reference or \
                    type(self.conv) in [torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d] and \
                    dtypes in [(torch.quint8, torch.qint8, None)]:
                # produce dequant - float_op - quant pattern
                dtype = torch.float
                if activation_int8_quantized:
                    dtype = activation_dtype(qconfig)
                activation = load_arg(quantized=dtype)(self.conv_node.args[0])
                args = load_arg(quantized=torch.float)(self.conv_node.args)
                # Get the float conv and attach quantization scheme and quantization
                # parameters of weight to the module
                # and qparam is a dictionary of
                # {"qscheme": ..., "scale": ..., "zero_point": ...} for per tensor quantization or
                # {"qscheme": ..., "scale": ..., "zero_point": ..., "axis": ...} for per channel quantization
                float_conv = self.conv
                fused_conv = None
                if isinstance(
                        float_conv,
                        QAT_CONV_MODULE_CLASSES):
                    # case 1. converting qat conv module to
                    # a float conv module, we need to attch
                    # weight fake_quant to the conv module,
                    # weight fake_quant is assumed to be run during
                    # QAT so we don't need to run it again here
                    float_conv = self.conv.to_float()  # type: ignore[operator]
                    # change qat conv to conv
                    parent_name, name = _parent_name(self.conv_node.target)
                    setattr(modules[parent_name], name, float_conv)
                    if isinstance(float_conv, torch.nn.intrinsic._FusedModule):
                        fused_conv = float_conv
                        float_conv = float_conv[0]
                    weight_post_process = self.conv.weight_fake_quant
                else:
                    # case 2. converting a conv module/fused conv module
                    # to float conv module, we need to attach
                    # weight observer to the conv module and run it
                    # with conv weight
                    if isinstance(float_conv, torch.nn.intrinsic._FusedModule):
                        fused_conv = float_conv
                        float_conv = float_conv[0]  # type: ignore[index]
                    assert qconfig is not None
                    weight_post_process = qconfig.weight()
                    # run weight observer
                    weight_post_process(float_conv.weight)  # type: ignore[operator]
                weight_qparams = get_qparam_dict(weight_post_process)
                # hardcoded for now, TODO: expose the api to user,
                # we can have a map from module to reference module
                # and allow user to register new ones
                qconv_cls = get_static_quant_module_class(
                    type(float_conv), is_reference=True)
                ref_conv = qconv_cls.from_float(float_conv, weight_qparams)  # type: ignore[attr-defined]
                # if the parent is a fused conv (Sequential), we can replace the first
                # item to ref conv, otherwise we can update
                # the conv instance in the module tree
                if fused_conv is not None:
                    fused_conv[0] = ref_conv
                else:
                    parent_name, name = _parent_name(self.conv_node.target)
                    setattr(modules[parent_name], name, ref_conv)
                op_out = create_node_from_old_node_preserve_meta(
                    quantized_graph,
                    ('call_module', self.conv_node.target, args, {}),
                    self.conv_node)
                if output_activation_post_process:
                    op_out = quantize_node(
                        op_out,
                        output_activation_post_process,
                        node,
                        modules,
                        quantized_graph,
                        node_name_to_scope,
                        is_input=False)
                return op_out
            else:
                if convert_custom_config_dict is None:
                    convert_custom_config_dict = {}
                additional_static_quant_mapping = convert_custom_config_dict.get("static", {})
                # 1. attach activation post process to module
                self.conv.activation_post_process = output_activation_post_process
                # 2. select quantized class
                qconv_cls = get_static_quant_module_class(
                    type(self.conv), additional_static_quant_mapping, is_reference=is_reference)
                quantized = qconv_cls.from_float(self.conv)
                parent_name, name = _parent_name(self.conv_node.target)
                setattr(modules[parent_name], name, quantized)
                return create_node_from_old_node_preserve_meta(
                    quantized_graph,
                    (
                        'call_module',
                        self.conv_node.target,
                        (load_arg(quantized=torch.quint8)(self.conv_node.args[0]),),
                        {},
                    ),
                    self.conv_node)
        else:  # call_function
            assert self.conv_node.op == "call_function"
            if is_reference:
                # make sure the input and weight are quantized to torch.quint8, torch.qint8, respectively
                load_arg(quantized={0: torch.quint8, 1: torch.qint8})(self.conv_node.args)
                args = load_arg(quantized=torch.float)(self.conv_node.args)
                kwargs = load_arg(quantized=torch.float)(self.conv_node.kwargs)
                op_out = create_node_from_old_node_preserve_meta(
                    quantized_graph,
                    ("call_function", self.conv, args, kwargs),
                    self.conv_node)
                if self.relu_node:
                    relu_args = [op_out]
                    relu_args.extend(load_arg(quantized=torch.float)(self.relu_node.args[1:]))
                    relu_kwargs = load_arg(quantized=torch.float)(self.relu_node.kwargs)
                    op_out = create_node_from_old_node_preserve_meta(
                        quantized_graph,
                        ("call_function", torch.nn.functional.relu, tuple(relu_args), relu_kwargs),
                        self.relu_node)

                if activation_int8_quantized:
                    root_module = modules['']
                    act_post_process_name = self.relu_node.name if self.relu_node else self.conv_node.name
                    act_post_process_node = self.relu_node if self.relu_node else self.conv_node
                    activation_post_process = \
                        self._maybe_get_last_node_only_observer(modules)
                    assert activation_post_process is not None
                    return quantize_node(
                        op_out,
                        activation_post_process,
                        act_post_process_node,
                        modules,
                        quantized_graph,
                        node_name_to_scope,
                        is_input=False)
                else:
                    # output for dynamically quantized conv op is not quantized
                    return op_out
            else:
                assert len(self.conv_node.args) >= 7, \
                    "only conv2d calls with all arguments specified is supported right now in is_reference=False option"
                # make sure the input and weight are quantized to torch.quint8, torch.qint8, respectively
                args = load_arg(quantized={0: torch.quint8, 1: torch.qint8})(self.conv_node.args)
                # pack weight
                weight = load_arg(quantized=torch.qint8)(self.conv_node.args[1])
                other_args = load_arg(quantized=torch.float)(self.conv_node.args[2:])
                bias, stride, padding, dilation, groups = other_args
                if self.conv == torch.nn.functional.conv1d:
                    # F.conv1d can take `int` as well as `list[int]` for stride,
                    # padding, dilation, but the prepack op cannot. Convert
                    # these to lists if needed.
                    stride = [stride] if isinstance(stride, int) else stride
                    padding = [padding] if isinstance(padding, int) else padding
                    dilation = [dilation] if isinstance(dilation, int) else dilation
                prepack_args = (weight, bias, stride, padding, dilation, groups)
                prepack_op = get_qconv_prepack_op(self.conv)
                packed_weight = quantized_graph.create_node(
                    "call_function", prepack_op, prepack_args, {})
                assert activation_int8_quantized, \
                    "currently only static quantization is supported for conv"
                # construct conv input
                if activation_int8_quantized:
                    qconv_op = get_qconv_op(self.conv, self.relu_node is not None)
                    conv_input = load_arg(quantized=torch.quint8)(self.conv_node.args[0])

                    activation_post_process = \
                        self._maybe_get_last_node_only_observer(modules)
                    assert activation_post_process is not None

                    scale, zero_point, _ = get_per_tensor_qparams(activation_post_process)
                    scale_node, zero_point_node = \
                        create_qparam_nodes(
                            self.conv_node.name, scale, zero_point, modules,
                            quantized_graph, node_name_to_scope)
                    qconv_args = (conv_input, packed_weight, scale_node, zero_point_node)
                    kwargs = load_arg(quantized=torch.float)(self.conv_node.kwargs)
                    op = create_node_from_old_node_preserve_meta(
                        quantized_graph,
                        ('call_function', qconv_op, qconv_args, kwargs),
                        self.conv_node)
                    # Store the name of the fused op to get the path of node after fusion as well.
                    # TODO: may need to change the key to Node regenerate the map in each transformation,
                    # since we might not be able to rely on the name
                    node_name_to_scope[op.name] = node_name_to_scope[self.conv_node.name]
                    return op
                else:
                    # conv2d_dyanmic branch
                    raise Exception("Only static quant is supported for conv")

@register_quant_pattern(torch.nn.Linear)
@register_quant_pattern(torch.nn.functional.linear)
@register_quant_pattern(torch.nn.qat.Linear)
@register_quant_pattern(torch.nn.intrinsic.LinearReLU)
@register_quant_pattern(torch.nn.intrinsic.qat.LinearReLU)
@register_quant_pattern((torch.nn.functional.relu, torch.nn.functional.linear))
@register_quant_pattern((torch.nn.ReLU, torch.nn.functional.linear))
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

    def convert(self,
                node: Node,
                qconfig: QConfigAny,
                modules: Dict[str, torch.nn.Module],
                quantized_graph: Graph,
                node_name_to_scope: Dict[str, Tuple[str, type]],
                load_arg: Callable,
                is_reference: bool = False,
                convert_custom_config_dict: Dict[str, Any] = None) -> Node:
        if convert_custom_config_dict is None:
            convert_custom_config_dict = {}
        # Supported combinations are:
        # quant_type | activation (compute_type) | weight
        #  static       quint8                      qint8
        #  dynamic      float32 (quint8)            qint8
        #  weight_only  float32                    float16
        # tuple (activation_dtype, weight_dtype, compute_dtype)
        supported_dtypes = [
            (torch.quint8, torch.qint8, None),
            (torch.float32, torch.qint8, torch.quint8),
            (torch.float32, torch.float16, None),
            # static float16 quantization
            (torch.float16, torch.float16, None),
        ]
        dtypes = get_qconfig_dtypes(qconfig)
        # leave the op unquantized if the dtype combination is not supported
        if not is_reference and dtypes not in supported_dtypes:
            warnings.warn(
                "dtype combination: {} is not "
                "supported by Linear "
                "supported dtype combinations are: {}".format(dtypes, supported_dtypes))
            if self.relu_node:
                op_out = quantized_graph.node_copy(self.linear_node, load_arg(quantized=torch.float))
                relu_args = [op_out]
                relu_args.extend(load_arg(quantized=torch.float)(self.relu_node.args[1:]))
                relu_kwargs = load_arg(quantized=torch.float)(self.relu_node.kwargs)
                return create_node_from_old_node_preserve_meta(
                    quantized_graph,
                    ("call_function", torch.nn.functional.relu, tuple(relu_args), relu_kwargs),
                    self.relu_node)
            else:
                return quantized_graph.node_copy(node, load_arg(quantized=None))

        activation_int8_quantized = activation_is_int8_quantized(qconfig)
        activation_statically_quantized = activation_is_statically_quantized(qconfig)
        weight_dtype = dtypes[1]
        if self.linear_node.op == 'call_module':

            output_activation_post_process = \
                self._maybe_get_last_node_only_observer(modules)

            # note that relu should already be fused into linear modul in the fusion step
            assert self.relu_node is None, 'linear module and relu fusion is not executed, ' \
                'please make sure to run fusion before prepare'
            # we'll always produce reference pattern for the following modules
            # will remove the else branch after we migrated all use cases
            module_allowlist = [
                torch.nn.Linear,
                torch.nn.qat.Linear,
                torch.nn.intrinsic.modules.fused.LinearReLU,
                torch.nn.intrinsic.qat.modules.linear_relu.LinearReLU
            ]
            if is_reference or type(self.linear) in module_allowlist and dtypes in [(torch.quint8, torch.qint8, None)]:
                # produce dequant - float_op - quant pattern
                dtype = torch.float
                if activation_int8_quantized:
                    dtype = activation_dtype(qconfig)
                activation = load_arg(quantized=dtype)(self.linear_node.args[0])
                args = load_arg(quantized=torch.float)(self.linear_node.args)

                # Get the float linear and attach qscheme and qparams the the module
                float_linear = self.linear
                fused_linear = None
                if isinstance(float_linear, (torch.nn.qat.Linear, torch.nn.intrinsic.qat.LinearReLU)):
                    float_linear = float_linear.to_float()
                    # change qat linear to linear
                    parent_name, name = _parent_name(self.linear_node.target)
                    setattr(modules[parent_name], name, float_linear)
                    # Attach weight fake quant to the linear module
                    if isinstance(float_linear, torch.nn.intrinsic.LinearReLU):
                        fused_linear = float_linear
                        float_linear = float_linear[0]
                    weight_post_process = self.linear.weight_fake_quant
                else:
                    if isinstance(float_linear, torch.nn.intrinsic.LinearReLU):
                        fused_linear = float_linear
                        float_linear = self.linear[0]  # type: ignore[index]
                    # Attach the weight observer to the module
                    weight_post_process = qconfig.weight()  # type: ignore[union-attr]

                # Run weight observer
                # TODO: This is currently a hack for QAT to get the right shapes for scale and zero point.
                # In the future, we should require the user to calibrate the model after calling prepare
                weight_post_process(float_linear.weight)  # type: ignore[operator]

                weight_qparams = get_qparam_dict(weight_post_process)
                # TODO: include the configuration in backend_config_dict
                # we can have a map from module to reference module
                # and allow user to register new ones
                qlinear_cls = get_static_quant_module_class(
                    type(float_linear), is_reference=True)
                ref_linear = qlinear_cls.from_float(float_linear, weight_qparams)

                # if the parent is a fused linear (Sequential), we can replace the first
                # item to ref linear, otherwise we can update
                # the linear instance in the module tree
                if fused_linear is not None:
                    fused_linear[0] = ref_linear
                else:
                    parent_name, name = _parent_name(self.linear_node.target)
                    setattr(modules[parent_name], name, ref_linear)
                op_out = create_node_from_old_node_preserve_meta(
                    quantized_graph,
                    ('call_module', self.linear_node.target, args, {}),
                    self.linear_node)
                if output_activation_post_process:
                    op_out = quantize_node(
                        op_out,
                        output_activation_post_process,
                        node,
                        modules,
                        quantized_graph,
                        node_name_to_scope,
                        is_input=False)
                return op_out
            # non-reference option
            else:
                # 1. attach output activation post process to linear module
                if output_activation_post_process:
                    self.linear.activation_post_process = output_activation_post_process

                # 2. select corresponding quantized linear class for the float linear class
                if activation_int8_quantized:
                    additional_static_quant_mapping = convert_custom_config_dict.get("static", {})
                    qlinear = get_static_quant_module_class(
                        type(self.linear), additional_static_quant_mapping)
                else:
                    assert dtypes in [
                        (torch.float32, torch.qint8, torch.quint8),
                        (torch.float32, torch.float16, None),
                    ], f"dtype {dtypes} not supported yet"
                    additional_dynamic_quant_mapping = convert_custom_config_dict.get("dynamic", {})
                    qlinear = get_dynamic_quant_module_class(type(self.linear), additional_dynamic_quant_mapping)

                quantized = qlinear.from_float(self.linear)
                parent_name, name = _parent_name(self.linear_node.target)
                setattr(modules[parent_name], name, quantized)
                # activation needs to be quantized for static quantization
                dtype = torch.float
                if activation_int8_quantized:
                    dtype = activation_dtype(qconfig)
                return create_node_from_old_node_preserve_meta(
                    quantized_graph,
                    (
                        'call_module',
                        self.linear_node.target,
                        (load_arg(quantized=dtype)(self.linear_node.args[0]),), {},
                    ),
                    self.linear_node)
        else:  # call_function
            assert self.linear_node.op == 'call_function'
            if is_reference:
                quantized_input_dtypes = [torch.float, torch.float]
                if activation_int8_quantized:
                    quantized_input_dtypes[0] = torch.quint8
                if weight_is_statically_quantized(qconfig):
                    quantized_input_dtypes[1] = torch.qint8
                args = load_arg(quantized=quantized_input_dtypes)(self.linear_node.args)
                args = load_arg(quantized=torch.float)(self.linear_node.args)
                kwargs = load_arg(quantized=torch.float)(self.linear_node.kwargs)
                op_out = create_node_from_old_node_preserve_meta(
                    quantized_graph,
                    ("call_function", torch.nn.functional.linear, args, kwargs),
                    self.linear_node)
                if self.relu_node:
                    relu_args = [op_out]
                    relu_args.extend(load_arg(quantized=torch.float)(self.relu_node.args[1:]))
                    relu_kwargs = load_arg(quantized=torch.float)(self.relu_node.kwargs)
                    op_out = create_node_from_old_node_preserve_meta(
                        quantized_graph,
                        ("call_function", torch.nn.functional.relu, tuple(relu_args), relu_kwargs),
                        self.relu_node)

                if activation_statically_quantized:
                    # quantize output for statically quantized linear op
                    root_module = modules['']
                    act_post_process_name = self.relu_node.name if self.relu_node else self.linear_node.name
                    act_post_process_node = self.relu_node if self.relu_node else self.linear_node
                    activation_post_process = \
                        self._maybe_get_last_node_only_observer(modules)
                    assert activation_post_process is not None
                    return quantize_node(
                        op_out,
                        activation_post_process,
                        act_post_process_node,
                        modules,
                        quantized_graph,
                        node_name_to_scope,
                        is_input=False)
                else:
                    # output for dynamically quantized linear op is not quantized
                    return op_out
            else:  # non-reference option
                # prepacking weights for static int8 quant and dynamic quant
                if dtypes != (torch.float16, torch.float16, None):
                    # linear args
                    # (x, weight, bias, ...)
                    # TODO: the name should be weight is int8 quantized
                    weight_quantized = weight_is_statically_quantized(qconfig)
                    dtype = weight_dtype if weight_quantized else torch.float
                    linear_weight = load_arg(quantized=dtype)(self.linear_node.args[1])

                    # get other arguments
                    kwargs = {**load_arg(quantized=torch.float)(self.linear_node.kwargs)}
                    # all args after bias, including bias
                    other_args = load_arg(quantized=torch.float)(self.linear_node.args[2:])
                    # bias might be either positional, or a keyword argument
                    if len(self.linear_node.args) > 2:
                        bias = load_arg(quantized=torch.float)(self.linear_node.args[2])
                        other_args = other_args[1:]  # remove the bias argument
                    else:
                        bias = kwargs.pop('bias', None)

                    prepack_args = (linear_weight, bias)
                    prepack_op = get_linear_prepack_op_for_dtype(weight_dtype)
                    packed_weight = quantized_graph.create_node(
                        'call_function', prepack_op, prepack_args, {})
                # construct linear input
                if activation_int8_quantized:
                    qlinear_op = torch.ops.quantized.linear_relu if self.relu_node else torch.ops.quantized.linear
                    linear_input = load_arg(quantized=torch.quint8)(self.linear_node.args[0])
                    activation_post_process = \
                        self._maybe_get_last_node_only_observer(modules)
                    assert activation_post_process is not None
                    scale, zero_point, _ = get_per_tensor_qparams(activation_post_process)
                    scale_node, zero_point_node = \
                        create_qparam_nodes(
                            self.linear_node.name, scale, zero_point, modules,
                            quantized_graph, node_name_to_scope)

                    qlinear_args = (linear_input, packed_weight, scale_node, zero_point_node)
                    op = create_node_from_old_node_preserve_meta(
                        quantized_graph,
                        ("call_function", qlinear_op, qlinear_args, kwargs),
                        self.linear_node)
                    # Store the name of the fused op to get the path of node after fusion as well.
                    # TODO: may need to change the key to Node regenerate the map in each transformation,
                    # since we might not be able to rely on the name
                    node_name_to_scope[op.name] = node_name_to_scope[self.linear_node.name]
                    return op
                elif dtypes in [(torch.float32, torch.qint8, torch.quint8),
                                (torch.float32, torch.float16, None)]:
                    # choose linear dynamic or linear dynamic fp16 op based on weight dtype
                    if weight_dtype == torch.qint8:
                        if self.relu_node:
                            qlinear_op = torch.ops.quantized.linear_relu_dynamic
                        else:
                            qlinear_op = torch.ops.quantized.linear_dynamic
                    else:
                        if self.relu_node:
                            qlinear_op = torch.ops.quantized.linear_relu_dynamic_fp16
                        else:
                            qlinear_op = torch.ops.quantized.linear_dynamic_fp16

                    linear_input = load_arg(quantized=torch.float)(self.linear_node.args[0])
                    qlinear_args = (linear_input, packed_weight)  # type: ignore[assignment]
                    op_out = create_node_from_old_node_preserve_meta(
                        quantized_graph,
                        ("call_function", qlinear_op, qlinear_args, kwargs),
                        self.linear_node)
                    # Store the name of the dynamic op to get the path of node after replacement as well.
                    # TODO: may need to change the key to Node regenerate the map in each transformation,
                    # since we might not be able to rely on the name
                    node_name_to_scope[op_out.name] = node_name_to_scope[self.linear_node.name]
                    return op_out
                else:
                    assert dtypes == (torch.float16, torch.float16, None)
                    # TODO (refactor) this is duplicated, maybe have a helper function
                    if self.relu_node:
                        op_out = quantized_graph.node_copy(self.linear_node, load_arg(quantized=torch.float))
                        relu_args = [op_out]
                        relu_args.extend(load_arg(quantized=torch.float)(self.relu_node.args[1:]))
                        relu_kwargs = load_arg(quantized=torch.float)(self.relu_node.kwargs)
                        op_out = create_node_from_old_node_preserve_meta(
                            quantized_graph,
                            ("call_function", torch.nn.functional.relu, tuple(relu_args), relu_kwargs),
                            self.relu_node)
                    else:
                        op_out = quantized_graph.node_copy(node, load_arg(quantized=torch.float))
                    return quantized_graph.create_node(
                        "call_method", "to", (op_out, torch.float16), {})

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

    def convert(self,
                node: Node,
                qconfig: QConfigAny,
                modules: Dict[str, torch.nn.Module],
                quantized_graph: Graph,
                node_name_to_scope: Dict[str, Tuple[str, type]],
                load_arg: Callable,
                is_reference: bool = False,
                convert_custom_config_dict: Dict[str, Any] = None) -> Node:
        if convert_custom_config_dict is None:
            convert_custom_config_dict = {}
        additional_static_quant_mapping = convert_custom_config_dict.get("static", {})
        # 1. attach activation post process to module
        output_activation_post_process = \
            self._maybe_get_last_node_only_observer(modules)
        assert output_activation_post_process is not None
        if is_reference:
            # produce dequant - float_op - quant pattern
            dtype = activation_dtype(qconfig)
            activation = load_arg(quantized=dtype)(self.bn_node.args[0])
            args = load_arg(quantized=torch.float)(self.bn_node.args)
            op_out = create_node_from_old_node_preserve_meta(
                quantized_graph,
                ("call_module", self.bn_node.target, args, {}),
                self.bn_node)
            if output_activation_post_process:
                op_out = quantize_node(
                    op_out,
                    output_activation_post_process,
                    node,
                    modules,
                    quantized_graph,
                    node_name_to_scope,
                    is_input=False)
            return op_out
        else:
            self.bn.activation_post_process = output_activation_post_process
            qbn_cls = get_static_quant_module_class(type(self.bn), additional_static_quant_mapping)
            quantized = qbn_cls.from_float(self.bn)
            parent_name, name = _parent_name(self.bn_node.target)
            setattr(modules[parent_name], name, quantized)
            return create_node_from_old_node_preserve_meta(
                quantized_graph,
                (
                    'call_module',
                    self.bn_node.target,
                    load_arg(quantized=[0])(self.bn_node.args),
                    load_arg(quantized=torch.float)(self.bn_node.kwargs),
                ),
                self.bn_node)

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

    def convert(self,
                node: Node,
                qconfig: QConfigAny,
                modules: Dict[str, torch.nn.Module],
                quantized_graph: Graph,
                node_name_to_scope: Dict[str, Tuple[str, type]],
                load_arg: Callable,
                is_reference: bool = False,
                convert_custom_config_dict: Dict[str, Any] = None) -> Node:
        # Supported combinations are:
        # quant_type  | activation | weight | activation_compute_type
        # weight_only |  float32   | quint8 | None
        # weight_only |  float32   | quint4x2 | None
        # tuple (activation_dtype, weight_dtype, compute_dtype)
        supported_dtypes = [
            (torch.float32, torch.quint8, None),
            (torch.float32, torch.quint4x2, None),
        ]
        assert node.op == 'call_module'
        emb_node = node
        dtypes = get_qconfig_dtypes(qconfig)
        # leave the op unquantized if the dtype combination is not supported
        if dtypes not in supported_dtypes:
            warnings.warn(
                "dtype combination: {} is not "
                "supported by Embedding/EmbeddingBag, "
                "supported dtype combinations are: {}".format(dtypes, supported_dtypes))
            return quantized_graph.node_copy(node, load_arg(quantized=None))

        emb = modules[str(emb_node.target)]
        qemb = get_static_quant_module_class(type(emb))
        quantized = qemb.from_float(emb)
        parent_name, name = _parent_name(emb_node.target)
        setattr(modules[parent_name], name, quantized)
        return create_node_from_old_node_preserve_meta(
            quantized_graph,
            (
                'call_module',
                emb_node.target,
                load_arg(quantized=torch.float)(emb_node.args),
                load_arg(quantized=torch.float)(emb_node.kwargs),
            ),
            emb_node)

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

    def input_output_observed(self) -> bool:
        return False

    def convert(self,
                node: Node,
                qconfig: QConfigAny,
                modules: Dict[str, torch.nn.Module],
                quantized_graph: Graph,
                node_name_to_scope: Dict[str, Tuple[str, type]],
                load_arg: Callable,
                is_reference: bool = False,
                convert_custom_config_dict: Dict[str, Any] = None) -> Node:
        # Supported combinations are:
        # quant_type  | activation | weight | activation_compute_type
        # dynamic |  float32   | qint8 | quint8
        # dynamic |  float32   | float16 | None
        # tuple (activation_dtype, weight_dtype, compute_dtype)
        supported_dtypes = [
            (torch.float32, torch.qint8, torch.quint8),
            (torch.float32, torch.float16, None),
        ]
        assert node.op == 'call_module'
        dtypes = get_qconfig_dtypes(qconfig)
        # leave the op unquantized if the dtype combination is not supported
        if dtypes not in supported_dtypes:
            warnings.warn(
                "dtype combination: {} is not "
                "supported by Embedding/EmbeddingBag, "
                "supported dtype combinations are: {}".format(dtypes, supported_dtypes))
            return quantized_graph.node_copy(node, load_arg(quantized=None))

        module = modules[str(node.target)]
        qmodule_cls = get_dynamic_quant_module_class(type(module))
        qmodule = qmodule_cls.from_float(module)
        parent_name, name = _parent_name(node.target)
        setattr(modules[parent_name], name, qmodule)
        return create_node_from_old_node_preserve_meta(
            quantized_graph,
            (
                'call_module',
                node.target,
                load_arg(quantized=torch.float)(node.args),
                load_arg(quantized=torch.float)(node.kwargs),
            ),
            node)

ARGS_TO_SKIP = {
    torch._ops.ops.quantized.hardswish: ['inplace'],
    torch._ops.ops.quantized.elu: ['inplace'],
    torch._ops.ops.quantized.dropout: ['inplace'],
    torch._ops.ops.quantized.instance_norm:
    ['running_mean', 'running_var', 'use_input_stats', 'momentum'],
}
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

    def convert(self,
                node: Node,
                qconfig: QConfigAny,
                modules: Dict[str, torch.nn.Module],
                quantized_graph: Graph,
                node_name_to_scope: Dict[str, Tuple[str, type]],
                load_arg: Callable,
                is_reference: bool = False,
                convert_custom_config_dict: Dict[str, Any] = None) -> Node:
        if not self.all_node_args_are_tensors:
            return NotImplemented
        assert node.op in ['call_module', 'call_function'], 'Only call_module and ' + \
            'call_function are handled in DefaultNode'
        if convert_custom_config_dict is None:
            convert_custom_config_dict = {}
        additional_static_quant_mapping = convert_custom_config_dict.get("static", {})

        dtypes = get_qconfig_dtypes(qconfig)
        if not is_reference and dtypes not in default_op_supported_dtypes[self.op]:
            warnings.warn(
                "dtype combination: {} is not "
                "supported by {} "
                "supported dtype combinations are: {}".format(dtypes, self.op, default_op_supported_dtypes[self.op]))
            return quantized_graph.node_copy(node, load_arg(quantized=torch.float))
        # TODO: make helper functions for (torch.quint8, torch.qint8, None)
        if not is_reference:
            if dtypes in [(torch.quint8, torch.qint8, None)]:
                activation_post_process = \
                    self._maybe_get_last_node_only_observer(modules)
                assert activation_post_process is not None
                if node.op == 'call_module':
                    module = modules[str(node.target)]
                    module.activation_post_process = activation_post_process
                    quantized_module_cls = get_static_quant_module_class(
                        type(module), additional_static_quant_mapping)
                    quantized_module = quantized_module_cls.from_float(module)
                    parent_name, name = _parent_name(node.target)
                    setattr(modules[parent_name], name, quantized_module)
                    return create_node_from_old_node_preserve_meta(
                        quantized_graph,
                        (
                            'call_module',
                            node.target,
                            load_arg(quantized=[0])(node.args),
                            load_arg(quantized=torch.float)(node.kwargs),
                        ),
                        node)
                else:
                    assert node.op == "call_function"
                    # call_function
                    scale, zero_point = activation_post_process.calculate_qparams()  # type: ignore[operator]
                    scale = float(scale)
                    zero_point = int(zero_point)
                    scale_arg, zero_point_arg = \
                        create_qparam_nodes(
                            node.name, scale, zero_point, modules,
                            quantized_graph, node_name_to_scope)

                    assert not isinstance(node.target, str), "Expecting node.target for "
                    "call_function to be a function instead of a string"
                    quantized_op = get_quantized_operator(node.target)
                    args = load_arg(quantized=[0])(node.args)
                    kwargs = {**load_arg(quantized=torch.float)(node.kwargs), "output_scale": scale_arg,
                              "output_zero_point": zero_point_arg}
                    if quantized_op in ARGS_TO_SKIP:
                        args_to_skip = ARGS_TO_SKIP[quantized_op]
                        for arg in args_to_skip:
                            if arg in kwargs:
                                kwargs.pop(arg)
                    return create_node_from_old_node_preserve_meta(
                        quantized_graph,
                        ("call_function", quantized_op, args, kwargs),  # type: ignore[arg-type]
                        node)
            else:
                assert dtypes in [(torch.float16, torch.float16, None)]
                # Generally fp16 kernels don't exist for fp16 ops
                warnings.warn(
                    "Only reference patterns are currently supported for {dtype} dtype with {op} op"
                    "".format(dtype=dtypes, op=self.op))
                op_out = quantized_graph.node_copy(node, load_arg(quantized=torch.float))
                return quantized_graph.create_node(
                    "call_method", "to", (op_out, torch.float16), {})
        else:
            assert is_reference
            # We can produce reference for a dtypes including
            # (torch.quint8, torch.qint8, torch.qint32, torch.float16)
            act_dtype = activation_dtype(qconfig)
            if act_dtype == torch.float:
                op_out = quantized_graph.node_copy(node, load_arg(quantized=torch.float))
                return op_out
            else:
                activation_post_process = \
                    self._maybe_get_last_node_only_observer(modules)
                assert activation_post_process is not None
                # make sure the input is quantized to act_dtype
                load_arg(quantized={0: act_dtype})(node.args)
                args = load_arg(quantized=torch.float)(node.args)
                kwargs = load_arg(quantized=torch.float)(node.kwargs)
                op_out = quantized_graph.node_copy(node, load_arg(quantized=torch.float))
                return quantize_node(
                    op_out, activation_post_process,
                    node, modules, quantized_graph, node_name_to_scope, is_input=False)

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

    def convert(self,
                node: Node,
                qconfig: QConfigAny,
                modules: Dict[str, torch.nn.Module],
                quantized_graph: Graph,
                node_name_to_scope: Dict[str, Tuple[str, type]],
                load_arg: Callable,
                is_reference: bool = False,
                convert_custom_config_dict: Dict[str, Any] = None) -> Node:
        act_dtype = activation_dtype(qconfig)
        if act_dtype == torch.float:
            op_out = quantized_graph.node_copy(node, load_arg(quantized=torch.float))
            return op_out
        else:
            activation_post_process = \
                self._maybe_get_last_node_only_observer(modules)
            assert activation_post_process is not None
            # make sure the input is quantized to act_dtype
            load_arg(quantized={0: act_dtype})(node.args)
            args = load_arg(quantized=torch.float)(node.args)
            kwargs = load_arg(quantized=torch.float)(node.kwargs)
            op_out = quantized_graph.node_copy(node, load_arg(quantized=torch.float))
            return quantize_node(
                op_out, activation_post_process,
                node, modules, quantized_graph, node_name_to_scope, is_input=False)

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

    def convert(self,
                node: Node,
                qconfig: QConfigAny,
                modules: Dict[str, torch.nn.Module],
                quantized_graph: Graph,
                node_name_to_scope: Dict[str, Tuple[str, type]],
                load_arg: Callable,
                is_reference: bool = False,
                convert_custom_config_dict: Dict[str, Any] = None) -> Node:

        is_call_function, is_call_method, is_call_module = check_node(node, modules)
        if is_reference or (is_call_function or is_call_method or is_call_module):
            # when activation dtype is torch.float, the node does not require
            # observation
            # e.g. dynamic quantization or weight_only quantization
            act_dtype = activation_dtype(qconfig)
            if act_dtype == torch.float:
                op_out = quantized_graph.node_copy(node, load_arg(quantized=torch.float))
                return op_out
            else:
                activation_post_process = \
                    self._maybe_get_last_node_only_observer(modules)
                assert activation_post_process is not None
                # make sure the input is quantized to act_dtype
                load_arg(quantized={0: act_dtype})(node.args)
                args = list(load_arg(quantized=torch.float)(node.args))
                kwargs = load_arg(quantized=torch.float)(node.kwargs)
                op_out = quantized_graph.node_copy(node, load_arg(quantized=torch.float))
                return quantize_node(
                    op_out,
                    activation_post_process,
                    node, modules, quantized_graph, node_name_to_scope, is_input=False)
        else:
            return quantized_graph.node_copy(node, load_arg(quantized=None))

class CustomModuleQuantizeHandler(QuantizeHandler):
    def convert(self,
                node: Node,
                qconfig: QConfigAny,
                modules: Dict[str, torch.nn.Module],
                quantized_graph: Graph,
                node_name_to_scope: Dict[str, Tuple[str, type]],
                load_arg: Callable,
                is_reference: bool = False,
                convert_custom_config_dict: Dict[str, Any] = None) -> Node:
        """ Convert a float custom module to quantized custom module
        """
        assert node.op == 'call_module'
        assert convert_custom_config_dict is not None
        custom_module_class_mapping = convert_custom_config_dict.get("observed_to_quantized_custom_module_class", None)
        assert custom_module_class_mapping is not None
        observed_custom_module = modules[str(node.target)]
        if activation_is_statically_quantized(qconfig):
            activation_post_process = \
                self._maybe_get_last_node_only_observer(modules)
            assert activation_post_process is not None
            observed_custom_module.activation_post_process = activation_post_process
        quantized_custom_module_class = get_swapped_custom_module_class(
            observed_custom_module, custom_module_class_mapping, qconfig)
        quantized_custom_module = \
            quantized_custom_module_class.from_observed(observed_custom_module)
        parent_name, name = _parent_name(node.target)
        setattr(modules[parent_name], name, quantized_custom_module)
        # hardcoded the quntized input to be None (take whatever is in the environemnt),
        # we can extend this
        # if there is a need, e.g. get the indexes of quantized inputs from some
        # module attribute like module._QUANTIZED_INPUT_INDEXES
        return quantized_graph.node_copy(node, load_arg(quantized=None))

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

    def convert(self,
                node: Node,
                qconfig: QConfigAny,
                modules: Dict[str, torch.nn.Module],
                quantized_graph: Graph,
                node_name_to_scope: Dict[str, Tuple[str, type]],
                load_arg: Callable,
                is_reference: bool = False,
                convert_custom_config_dict: Dict[str, Any] = None) -> Node:
        # when activation dtype is torch.float, the node does not require
        # observation
        # e.g. dynamic quantization or weight_only quantization
        act_dtype = activation_dtype(qconfig)
        if act_dtype == torch.float:
            op_out = quantized_graph.node_copy(node, load_arg(quantized=torch.float))
            return op_out
        else:
            activation_post_process = \
                self._maybe_get_last_node_only_observer(modules)
            if activation_post_process is not None:
                args = list(load_arg(quantized=torch.float)(node.args))
                kwargs = load_arg(quantized=torch.float)(node.kwargs)
                op_out = quantized_graph.node_copy(node, load_arg(quantized=torch.float))
                return quantize_node(
                    op_out,
                    activation_post_process,
                    node, modules, quantized_graph, node_name_to_scope, is_input=False)
            else:
                return quantized_graph.node_copy(node, load_arg(quantized=torch.float))

class StandaloneModuleQuantizeHandler(QuantizeHandler):
    """ Converts an observed standalone module to quantized standalone module
    by calling convert_fx on the observed standalone module.
    """
    def convert(self,
                node: Node,
                qconfig: QConfigAny,
                modules: Dict[str, torch.nn.Module],
                quantized_graph: Graph,
                node_name_to_scope: Dict[str, Tuple[str, type]],
                load_arg: Callable,
                is_reference: bool = False,
                convert_custom_config_dict: Dict[str, Any] = None) -> Node:
        assert node.op == 'call_module'
        convert = torch.ao.quantization.quantize_fx._convert_standalone_module_fx  # type: ignore[attr-defined]
        # We know that observed standalone module is a GraphModule since
        # it's produced by us
        observed_standalone_module : GraphModule = modules[str(node.target)]  # type: ignore[assignment]
        input_quantized_idxs = observed_standalone_module._standalone_module_input_quantized_idxs.tolist()  # type: ignore[operator]
        quantized_standalone_module = convert(observed_standalone_module, is_reference=is_reference)
        parent_name, name = _parent_name(node.target)
        # update the modules dict
        setattr(modules[parent_name], name, quantized_standalone_module)
        modules[str(node.target)] = quantized_standalone_module
        return quantized_graph.node_copy(node, load_arg(quantized=input_quantized_idxs))
