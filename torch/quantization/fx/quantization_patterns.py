import torch
from torch.fx import GraphModule
from torch.fx.graph import (
    Node,
    Graph,
)
from torch.quantization import (
    default_affine_fixed_qparams_fake_quant,
    default_symmetric_fixed_qparams_fake_quant,
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
)

from ..quantize import (
    is_activation_post_process,
)

from .pattern_utils import (
    register_quant_pattern,
    get_default_output_activation_post_process_map,
    Pattern,
)

from .utils import (
    _parent_name,
    all_node_args_have_no_tensors,
    quantize_node,
    get_per_tensor_qparams,
    get_linear_prepack_op_for_dtype,
    create_qparam_nodes,
    get_qconv_prepack_op,
    get_qconv_op,
)

from .qconfig_utils import QConfigAny

from abc import ABC, abstractmethod
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
    ) -> Optional[Callable]:
        """
        Returns the constructor for the activation observer which should be
        used for the pattern matched to this handler. Some handlers override
        this to a different value than what is specified in the qconfig.
        """
        return qconfig.activation


    @abstractmethod
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
binary_op_all_dtypes = [
    (torch.quint8, torch.qint8, None),
    (torch.float16, torch.float16, None),
]
binary_op_float16_dtypes = [
    (torch.float16, torch.float16, None)
]
binary_op_int8_dtypes = [
    (torch.quint8, torch.qint8, None),
]
binary_op_supported_dtypes : Dict[Union[Callable, str], List[Tuple[torch.dtype, torch.dtype, None]]] = {
    operator.add: binary_op_all_dtypes,
    torch.add: binary_op_all_dtypes,
    operator.mul: binary_op_all_dtypes,
    torch.mul: binary_op_all_dtypes,
    torch.bmm: binary_op_float16_dtypes,
    torch.sub: binary_op_float16_dtypes,
    operator.sub: binary_op_float16_dtypes,
    torch.div: binary_op_float16_dtypes,
    operator.truediv: binary_op_float16_dtypes,
    torch.sum: binary_op_float16_dtypes
}
binary_reference_op_supported_dtypes : Dict[Union[Callable, str], List[Tuple[torch.dtype, torch.dtype, None]]] = {
    torch.bmm: binary_op_int8_dtypes,
}


@register_quant_pattern(operator.add)
@register_quant_pattern(operator.sub)
@register_quant_pattern(operator.mul)
@register_quant_pattern(operator.truediv)
@register_quant_pattern(torch.add)
@register_quant_pattern(torch.sub)
@register_quant_pattern(torch.mul)
@register_quant_pattern(torch.div)
@register_quant_pattern(torch.sum)
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
        if self.num_tensor_args == 1:
            return activation_dtype(qconfig) == torch.float16
        elif self.all_node_args_are_tensors and self.input_output_observed():
            return True
        else:
            return False

    def input_output_observed(self):
        # for x + y where x and y are scalars, we do not observe anything
        return self.num_tensor_args > 0

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

        if is_reference and self.binary_op in binary_reference_op_supported_dtypes and \
                dtypes in binary_reference_op_supported_dtypes[self.binary_op]:
            if dtypes in binary_op_int8_dtypes:
                args = load_arg(quantized=[torch.quint8, torch.qint8])(node.args)
                args = load_arg(quantized=torch.float)(node.args)
                kwargs = load_arg(quantized=torch.float)(node.kwargs)
                op_out = quantized_graph.node_copy(node, load_arg(quantized=torch.float))
                activation_post_process = \
                    self._maybe_get_last_node_only_observer(modules)
                assert activation_post_process is not None
                return quantize_node(
                    op_out, activation_post_process,
                    node, modules, quantized_graph, node_name_to_scope, is_input=False)
            else:
                warnings.warn(
                    "No implementation found for dtype combination: {}"
                    "for op {} with is_reference={} despite it being listed as supported"
                    "this should not happen".format(dtypes, self.binary_op, is_reference))
                return quantized_graph.node_copy(node, load_arg(quantized=torch.float))
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

                    return quantized_graph.create_node(
                        'call_function', self.quantized_binary_op,
                        load_arg(quantized=[quantized_index])(self.binary_op_node.args), self.binary_op_node.kwargs)
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
                    op = quantized_graph.create_node(
                        'call_function', self.quantized_binary_op, add_args, kwargs)
                    return op
            else:
                assert dtypes == (torch.float16, torch.float16, None)
                # TODO (refactor) this is duplicated, maybe have a helper function
                if self.relu_node:
                    op_out = quantized_graph.node_copy(self.binary_op_node, load_arg(quantized=torch.float))
                    relu_args = [op_out]
                    relu_args.extend(load_arg(quantized=torch.float)(self.relu_node.args[1:]))
                    relu_kwargs = load_arg(quantized=torch.float)(self.relu_node.kwargs)
                    op_out = quantized_graph.create_node(
                        "call_function", torch.nn.functional.relu, tuple(relu_args), relu_kwargs)
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
                "Supported reference dtype combinations are: {}"
                "".format(dtypes,
                          self.binary_op,
                          is_reference,
                          binary_op_supported_dtypes[self.binary_op],
                          (
                              [] if self.binary_op not in binary_reference_op_supported_dtypes.keys()
                              else binary_reference_op_supported_dtypes[self.binary_op]
                          )
                          )
            )
            if self.relu_node:
                op_out = quantized_graph.node_copy(self.binary_op_node, load_arg(quantized=torch.float))
                relu_args = [op_out]
                relu_args.extend(load_arg(quantized=torch.float)(self.relu_node.args[1:]))
                relu_kwargs = load_arg(quantized=torch.float)(self.relu_node.kwargs)
                return quantized_graph.create_node(
                    "call_function", torch.nn.functional.relu, tuple(relu_args), relu_kwargs)
            else:
                return quantized_graph.node_copy(node, load_arg(quantized=torch.float))


@register_quant_pattern(torch.cat)
class CatQuantizeHandler(QuantizeHandler):
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
        return quantized_graph.node_copy(node, load_arg(quantized=torch.quint8))

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
@register_quant_pattern((torch.nn.ReLU, torch.nn.Conv2d))
@register_quant_pattern((torch.nn.ReLU, torch.nn.Conv3d))
@register_quant_pattern((torch.nn.functional.relu, torch.nn.Conv2d))
@register_quant_pattern((torch.nn.functional.relu, torch.nn.Conv3d))
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
        if dtypes not in supported_dtypes:
            warnings.warn(
                "dtype combination: {} is not "
                "supported by Conv "
                "supported dtype combinations are: {}".format(dtypes, supported_dtypes))
            if self.relu_node:
                conv_out = quantized_graph.node_copy(self.conv_node, load_arg(quantized=torch.float))
                relu_args = [conv_out]
                relu_args.extend(load_arg(quantized=torch.float)(self.relu_node.args[1:]))
                relu_kwargs = load_arg(quantized=torch.float)(self.relu_node.kwargs)
                return quantized_graph.create_node(
                    "call_function", torch.nn.functional.relu, tuple(relu_args), relu_kwargs)
            else:
                return quantized_graph.node_copy(node, load_arg(quantized=torch.float))

        activation_int8_quantized = activation_is_int8_quantized(qconfig)

        if self.conv_node.op == 'call_module':
            # note that relu should already be fused into conv module in the fusion step
            assert self.relu_node is None, 'conv module and relu fusion is not executed, ' \
                'please make sure to run fusion before prepare'
            if convert_custom_config_dict is None:
                convert_custom_config_dict = {}
            additional_static_quant_mapping = convert_custom_config_dict.get("static", {})
            # 1. attach activation post process to module
            output_activation_post_process = \
                self._maybe_get_last_node_only_observer(modules)
            assert output_activation_post_process is not None
            self.conv.activation_post_process = output_activation_post_process
            # 2. select quantized class
            qconv_cls = get_static_quant_module_class(
                type(self.conv), additional_static_quant_mapping, is_reference=is_reference)
            quantized = qconv_cls.from_float(self.conv)
            parent_name, name = _parent_name(self.conv_node.target)
            setattr(modules[parent_name], name, quantized)
            return quantized_graph.create_node(
                'call_module',
                self.conv_node.target,
                (load_arg(quantized=torch.quint8)(self.conv_node.args[0]),),
                {})
        else:  # call_function
            assert self.conv_node.op == "call_function"
            if is_reference:
                args = load_arg(quantized=[torch.quint8, torch.qint8])(self.conv_node.args)
                args = load_arg(quantized=torch.float)(self.conv_node.args)
                kwargs = load_arg(quantized=torch.float)(self.conv_node.kwargs)
                op_out = quantized_graph.create_node(
                    "call_function", self.conv, args, kwargs)
                if self.relu_node:
                    relu_args = [op_out]
                    relu_args.extend(load_arg(quantized=torch.float)(self.relu_node.args[1:]))
                    relu_kwargs = load_arg(quantized=torch.float)(self.relu_node.kwargs)
                    op_out = quantized_graph.create_node(
                        "call_function", torch.nn.functional.relu, tuple(relu_args), relu_kwargs)

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
                    op = quantized_graph.create_node(
                        'call_function', qconv_op, qconv_args, kwargs)
                    # Store the name of the fused op to get the path of node after fusion as well.
                    # TODO: may need to change the key to Node regenerate the map in each transformation,
                    # since we might not be able to rely on the name
                    node_name_to_scope[op.name] = node_name_to_scope[self.conv_node.name]
                    return op
                else:
                    # conv2d_dyanmic branch
                    raise Exception("Only static quant is supported for conv")


# handle linear, maybe followed by relu
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
        if dtypes not in supported_dtypes:
            warnings.warn(
                "dtype combination: {} is not "
                "supported by Linear "
                "supported dtype combinations are: {}".format(dtypes, supported_dtypes))
            if self.relu_node:
                op_out = quantized_graph.node_copy(self.linear_node, load_arg(quantized=torch.float))
                relu_args = [op_out]
                relu_args.extend(load_arg(quantized=torch.float)(self.relu_node.args[1:]))
                relu_kwargs = load_arg(quantized=torch.float)(self.relu_node.kwargs)
                return quantized_graph.create_node(
                    "call_function", torch.nn.functional.relu, tuple(relu_args), relu_kwargs)
            else:
                return quantized_graph.node_copy(node, load_arg(quantized=None))

        activation_int8_quantized = activation_is_int8_quantized(qconfig)
        activation_statically_quantized = activation_is_statically_quantized(qconfig)
        weight_dtype = dtypes[1]
        # TODO: reference_model option for linear module
        if self.linear_node.op == 'call_module':

            output_activation_post_process = \
                self._maybe_get_last_node_only_observer(modules)

            # note that relu should already be fused into conv module in the fusion step
            assert self.relu_node is None, 'linear module and relu fusion is not executed, ' \
                'please make sure to run fusion before prepare'
            # 1. attach output activation post process to linear module

            if output_activation_post_process:
                self.linear.activation_post_process = output_activation_post_process

            # 2. select corresponding quantized linear class for the float linear class
            if activation_int8_quantized:
                additional_static_quant_mapping = convert_custom_config_dict.get("static", {})
                qlinear = get_static_quant_module_class(
                    type(self.linear), additional_static_quant_mapping,
                    is_reference=is_reference)
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
            return quantized_graph.create_node(
                'call_module',
                self.linear_node.target,
                (load_arg(quantized=dtype)(self.linear_node.args[0]),), {})
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
                op_out = quantized_graph.create_node(
                    "call_function", torch.nn.functional.linear, args, kwargs)
                if self.relu_node:
                    relu_args = [op_out]
                    relu_args.extend(load_arg(quantized=torch.float)(self.relu_node.args[1:]))
                    relu_kwargs = load_arg(quantized=torch.float)(self.relu_node.kwargs)
                    op_out = quantized_graph.create_node(
                        "call_function", torch.nn.functional.relu, tuple(relu_args), relu_kwargs)

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
                    # pack weight
                    bias = None
                    # all args after bias, including bias
                    other_args = load_arg(quantized=torch.float)(self.linear_node.args[2:])
                    if len(self.linear_node.args) > 2:
                        bias = load_arg(quantized=torch.float)(self.linear_node.args[2])
                        other_args = other_args[1:]  # remove the bias argument
                    else:
                        assert 'bias' in kwargs, \
                            'expect bias provided as a keyword argument when it is not a positional argument'
                        bias = kwargs['bias']
                        kwargs.pop('bias')
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
                    op = quantized_graph.create_node(
                        "call_function", qlinear_op, qlinear_args, kwargs)
                    # Store the name of the fused op to get the path of node after fusion as well.
                    # TODO: may need to change the key to Node regenerate the map in each transformation,
                    # since we might not be able to rely on the name
                    node_name_to_scope[op.name] = node_name_to_scope[self.linear_node.name]
                    return op
                elif dtypes in [(torch.float32, torch.qint8, torch.quint8),
                                (torch.float32, torch.float16, None)]:
                    # choose linear dynamic or linear dynamic fp16 op based on weight dtype
                    qlinear_op = torch.ops.quantized.linear_dynamic \
                        if weight_dtype == torch.qint8 \
                        else torch.ops.quantized.linear_dynamic_fp16
                    linear_input = load_arg(quantized=torch.float)(self.linear_node.args[0])
                    qlinear_args = (linear_input, packed_weight)  # type: ignore[assignment]
                    op_out = quantized_graph.create_node(
                        "call_function", qlinear_op, qlinear_args, kwargs)
                    # Store the name of the dynamic op to get the path of node after replacement as well.
                    # TODO: may need to change the key to Node regenerate the map in each transformation,
                    # since we might not be able to rely on the name
                    node_name_to_scope[op_out.name] = node_name_to_scope[self.linear_node.name]
                    if self.relu_node:
                        op_out = quantized_graph.create_node("call_function", torch.nn.functional.relu, (op_out,), {})
                    return op_out
                else:
                    assert dtypes == (torch.float16, torch.float16, None)
                    # TODO (refactor) this is duplicated, maybe have a helper function
                    if self.relu_node:
                        op_out = quantized_graph.node_copy(self.linear_node, load_arg(quantized=torch.float))
                        relu_args = [op_out]
                        relu_args.extend(load_arg(quantized=torch.float)(self.relu_node.args[1:]))
                        relu_kwargs = load_arg(quantized=torch.float)(self.relu_node.kwargs)
                        op_out = quantized_graph.create_node(
                            "call_function", torch.nn.functional.relu, tuple(relu_args), relu_kwargs)
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
        activation_post_process = \
            self._maybe_get_last_node_only_observer(modules)
        assert activation_post_process is not None
        self.bn.activation_post_process = activation_post_process
        qbn_cls = get_static_quant_module_class(type(self.bn), additional_static_quant_mapping)
        quantized = qbn_cls.from_float(self.bn)
        parent_name, name = _parent_name(self.bn_node.target)
        setattr(modules[parent_name], name, quantized)
        return quantized_graph.create_node(
            'call_module',
            self.bn_node.target,
            load_arg(quantized=[0])(self.bn_node.args),
            load_arg(quantized=torch.float)(self.bn_node.kwargs))

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
        return quantized_graph.create_node(
            'call_module',
            emb_node.target,
            load_arg(quantized=torch.float)(emb_node.args),
            load_arg(quantized=torch.float)(emb_node.kwargs))

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
        return quantized_graph.create_node(
            'call_module',
            node.target,
            load_arg(quantized=torch.float)(node.args),
            load_arg(quantized=torch.float)(node.kwargs))

ARGS_TO_SKIP = {
    torch._ops.ops.quantized.hardswish: ['inplace'],
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
# we currently only support reference patterns for these ops so they have been removed
# until they receive a proper fp16 kernel. To use the reference pattern, use a custom qconfig
# @register_quant_pattern(torch.nn.GELU)
# @register_quant_pattern(torch.nn.Softmax)
@register_quant_pattern(torch.nn.functional.hardswish)
@register_quant_pattern(torch.nn.functional.instance_norm)
@register_quant_pattern(torch.nn.functional.layer_norm)
@register_quant_pattern(torch.nn.functional.leaky_relu)
@register_quant_pattern(torch.nn.functional.silu)
@register_quant_pattern(torch.nn.functional.mish)
# we currently only support reference patterns for these ops so they have been removed
# until they receive a proper fp16 kernel. To use the reference pattern, use a custom qconfig
# @register_quant_pattern(torch.nn.functional.gelu)
# @register_quant_pattern(torch.nn.functional.softmax)
class DefaultNodeQuantizeHandler(QuantizeHandler):
    ''' Common quantized op, first input and first output will be quantized
    '''

    def __init__(
            self,
            node: Node,
            modules: Dict[str, torch.nn.Module]):
        super().__init__(node, modules)
        if node.op == "call_function" or node.op == "call_method":
            self.op = node.target
        elif node.op == "call_module":
            self.op = type(modules[str(node.target)])

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

        all_dtypes = [
            (torch.quint8, torch.qint8, None),
            (torch.float16, torch.float16, None)
        ]
        int8_dtypes = [
            (torch.quint8, torch.qint8, None)
        ]
        fp16_dtypes = [
            (torch.float16, torch.float16, None)
        ]
        supported_dtypes = {
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
            torch.nn.Softmax: int8_dtypes,
            torch.nn.functional.hardswish: int8_dtypes,
            torch.nn.functional.instance_norm: int8_dtypes,
            torch.nn.functional.layer_norm: all_dtypes,
            torch.nn.functional.leaky_relu: int8_dtypes,
            torch.nn.functional.silu: fp16_dtypes,
            torch.nn.functional.mish: fp16_dtypes,
            torch.nn.functional.gelu: int8_dtypes,
            torch.nn.functional.softmax: int8_dtypes,
        }
        dtypes = get_qconfig_dtypes(qconfig)
        if dtypes not in supported_dtypes[self.op]:
            warnings.warn(
                "dtype combination: {} is not "
                "supported by {} "
                "supported dtype combinations are: {}".format(dtypes, self.op, supported_dtypes[self.op]))
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
                    return quantized_graph.create_node(
                        'call_module',
                        node.target,
                        load_arg(quantized=[0])(node.args),
                        load_arg(quantized=torch.float)(node.kwargs))
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
                    return quantized_graph.create_node(
                        "call_function", quantized_op, args, kwargs)  # type: ignore[arg-type]
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
            if dtypes in [(torch.quint8, torch.qint8, None)]:
                load_arg(quantized=[0])(node.args)
                args = load_arg(quantized=torch.float)(node.args)
                kwargs = load_arg(quantized=torch.float)(node.kwargs)
                op_out = quantized_graph.node_copy(node, load_arg(quantized=torch.float))
                activation_post_process = \
                    self._maybe_get_last_node_only_observer(modules)
                assert activation_post_process is not None
                return quantize_node(
                    op_out, activation_post_process,
                    node, modules, quantized_graph, node_name_to_scope, is_input=False)
            else:
                assert dtypes in [(torch.float16, torch.float16, None)]
                op_out = quantized_graph.node_copy(node, load_arg(quantized=torch.float))
                return quantized_graph.create_node(
                    "call_method", "to", (op_out, torch.float16), {})


# TODO: elu is using scale/zero_point instead of output_scale, output_zero_point
@register_quant_pattern(torch.nn.functional.elu)
class ELUQuantizeHandler(QuantizeHandler):
    def convert(self,
                node: Node,
                qconfig: QConfigAny,
                modules: Dict[str, torch.nn.Module],
                quantized_graph: Graph,
                node_name_to_scope: Dict[str, Tuple[str, type]],
                load_arg: Callable,
                is_reference: bool = False,
                convert_custom_config_dict: Dict[str, Any] = None) -> Node:
        activation_post_process = \
            self._maybe_get_last_node_only_observer(modules)
        assert activation_post_process is not None
        scale, zero_point = activation_post_process.calculate_qparams()  # type: ignore[operator]
        scale = float(scale)
        zero_point = int(zero_point)
        scale_arg, zero_point_arg = create_qparam_nodes(
            node.name, scale, zero_point, modules, quantized_graph, node_name_to_scope)

        quantized_op = get_quantized_operator(node.target)
        args = load_arg(quantized=[0])(node.args)
        kwargs = {**load_arg(quantized=torch.float)(node.kwargs), 'output_scale': scale_arg, 'output_zero_point': zero_point_arg}
        kwargs.pop('inplace')
        return quantized_graph.create_node(
            'call_function', quantized_op, args, kwargs)  # type: ignore[arg-type]

@register_quant_pattern(torch.nn.Hardsigmoid, default_affine_fixed_qparams_fake_quant)
@register_quant_pattern(torch.nn.functional.hardsigmoid, default_affine_fixed_qparams_fake_quant)
@register_quant_pattern('hardsigmoid', default_affine_fixed_qparams_fake_quant)
@register_quant_pattern('hardsigmoid_', default_affine_fixed_qparams_fake_quant)
@register_quant_pattern(torch.nn.Sigmoid, default_affine_fixed_qparams_fake_quant)
@register_quant_pattern(torch.sigmoid, default_affine_fixed_qparams_fake_quant)
@register_quant_pattern('sigmoid', default_affine_fixed_qparams_fake_quant)
@register_quant_pattern('sigmoid_', default_affine_fixed_qparams_fake_quant)
@register_quant_pattern(torch.nn.Tanh, default_symmetric_fixed_qparams_fake_quant)
@register_quant_pattern(torch.tanh, default_symmetric_fixed_qparams_fake_quant)
@register_quant_pattern('tanh', default_symmetric_fixed_qparams_fake_quant)
@register_quant_pattern('tanh_', default_symmetric_fixed_qparams_fake_quant)
class FixedQParamsOpQuantizeHandler(QuantizeHandler):
    def __init__(self,
                 node: Node,
                 modules: Dict[str, torch.nn.Module]):
        super().__init__(node, modules)
        self.node = node

    def should_insert_observer_for_output(
        self,
        qconfig: Any,
        model_is_training: bool,
    ) -> bool:
        if model_is_training:
            # in QAT, always insert fake_quants
            return True
        else:
            # in PTQ, only insert observers when emulating fp16
            return activation_dtype(qconfig) == torch.float16

    def should_mark_output_quantized_from_input_quantized_status(
        self,
        qconfig: QConfigAny
    ) -> bool:
        # FixQParamOps are the same as CopyNode in int8 quantization
        return activation_dtype(qconfig) in [torch.quint8, torch.qint8]

    # some qhandlers override the activations constructor
    def get_activation_ctr(self, qconfig, pattern) -> Optional[Callable]:
        if activation_dtype(qconfig) == torch.float16:
            return qconfig.activation
        else:
            return get_default_output_activation_post_process_map().get(
                pattern, None)

    def convert(self,
                node: Node,
                qconfig: QConfigAny,
                modules: Dict[str, torch.nn.Module],
                quantized_graph: Graph,
                node_name_to_scope: Dict[str, Tuple[str, type]],
                load_arg: Callable,
                is_reference: bool = False,
                convert_custom_config_dict: Dict[str, Any] = None) -> Node:
        dtypes = get_qconfig_dtypes(qconfig)
        if dtypes == (torch.float16, torch.float16, None):
            op_out = quantized_graph.node_copy(node, load_arg(quantized=torch.float))
            return quantized_graph.create_node(
                "call_method", "to", (op_out, torch.float16,), {}
            )
        else:
            return quantized_graph.node_copy(node, load_arg(quantized=None))


# these ops have quantized equivalents that do not need any extra information
@register_quant_pattern(torch.nn.AdaptiveAvgPool1d)
@register_quant_pattern(torch.nn.AdaptiveAvgPool2d)
@register_quant_pattern(torch.nn.AdaptiveAvgPool3d)
@register_quant_pattern(torch.nn.AvgPool1d)
@register_quant_pattern(torch.nn.AvgPool2d)
@register_quant_pattern(torch.nn.AvgPool3d)
@register_quant_pattern(torch.nn.Dropout)
@register_quant_pattern(torch.nn.Hardtanh)
@register_quant_pattern(torch.nn.Identity)
@register_quant_pattern(torch.nn.MaxPool1d)
@register_quant_pattern(torch.nn.MaxPool2d)
@register_quant_pattern(torch.nn.MaxPool3d)
@register_quant_pattern(torch.nn.ReLU)
@register_quant_pattern(torch.nn.ReLU6)
@register_quant_pattern(torch.adaptive_avg_pool1d)
@register_quant_pattern(torch.nn.functional.adaptive_avg_pool2d)
@register_quant_pattern(torch.nn.functional.adaptive_avg_pool3d)
@register_quant_pattern(torch.nn.functional.dropout)
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
@register_quant_pattern(torch.chunk)
@register_quant_pattern(torch.clamp)
@register_quant_pattern(torch.flatten)
@register_quant_pattern(torch.transpose)
@register_quant_pattern(torch.max)
@register_quant_pattern(torch.mean)
@register_quant_pattern(torch.min)
@register_quant_pattern(torch.repeat_interleave)
@register_quant_pattern(torch.sort)
@register_quant_pattern(torch.squeeze)
@register_quant_pattern(torch.stack)
@register_quant_pattern(torch.unsqueeze)
@register_quant_pattern(operator.floordiv)
@register_quant_pattern(operator.getitem)
@register_quant_pattern('chunk')
@register_quant_pattern('clamp')
@register_quant_pattern('contiguous')
@register_quant_pattern('detach')
@register_quant_pattern('detach_')
@register_quant_pattern('mean')
@register_quant_pattern('numel')
@register_quant_pattern('permute')
@register_quant_pattern('relu')
@register_quant_pattern('relu_')
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
class CopyNodeQuantizeHandler(QuantizeHandler):
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
        convert = torch.quantization.quantize_fx._convert_standalone_module_fx  # type: ignore[attr-defined]
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
