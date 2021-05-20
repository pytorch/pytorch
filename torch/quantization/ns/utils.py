import enum
import operator

import torch
import torch.nn as nn
toq = torch.ops.quantized
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.quantization.fx.quantize import is_activation_post_process

from .ns_types import NSNodeTargetType

from typing import Any, Tuple, Callable, Dict, Set, List

def getattr_from_fqn(gm: GraphModule, fqn: str) -> Any:
    """
    Given a gm and a fqn such as "foo.bar.baz", returns gm.foo.bar.baz.
    """
    fqn_parts = fqn.split(".")
    cur_val = gm
    for part in fqn_parts:
        cur_val = getattr(cur_val, part)
    return cur_val

# TODO(future PR): consider deleting this enum and using the torch types
# directly.  This might be tricky because it is not a one to one mapping.
class NodeInputOrOutputType(enum.Enum):
    FP32 = enum.auto()  # torch.float
    INT8 = enum.auto()  # torch.qint8 or torch.quint8
    FP16 = enum.auto()  # torch.float16
    UNKNOWN = enum.auto()  # we cannot determine input/output dtype
    # TODO(future PR): while these functions can support multiple dtypes,
    #   for the purposes of numerical debugging we want to get the actual
    #   dtype used in the model. We will likely need some kind of dtype
    #   propagation to estimate this.
    FP32_OR_INT8 = enum.auto()  # either torch.float or torch.quint8 or torch.qint8
    # TODO(future PRs): dynamic quant, fake quant, etc


def get_node_first_input_and_output_type(
    node: Node,
    gm: GraphModule,
    logger_cls: Callable,
    node_type_to_io_type_map: Dict[str, Set[NSNodeTargetType]],
) -> Tuple[NodeInputOrOutputType, NodeInputOrOutputType]:

    # TODO(future PR): clean this up
    FUNS_IO_TYPE_FP32 = node_type_to_io_type_map['funs_io_type_fp32']
    FUNS_IO_TYPE_FP16 = node_type_to_io_type_map['funs_io_type_fp16']
    FUNS_IO_TYPE_INT8 = node_type_to_io_type_map['funs_io_type_int8']
    FUNS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map['funs_io_type_fp32_or_int8']
    MODS_IO_TYPE_FP32 = node_type_to_io_type_map['mods_io_type_fp32']
    MODS_IO_TYPE_INT8 = node_type_to_io_type_map['mods_io_type_int8']
    MODS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map['mods_io_type_fp32_or_int8']
    METHS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map['meths_io_type_fp32_or_int8']

    if node.op == 'call_function':
        if node.target in FUNS_IO_TYPE_FP32:
            return (NodeInputOrOutputType.FP32, NodeInputOrOutputType.FP32)
        if node.target in FUNS_IO_TYPE_FP16:
            return (NodeInputOrOutputType.FP16, NodeInputOrOutputType.FP16)
        elif node.target in FUNS_IO_TYPE_INT8:
            return (NodeInputOrOutputType.INT8, NodeInputOrOutputType.INT8)
        elif node.target in FUNS_IO_TYPE_FP32_OR_INT8:
            return (NodeInputOrOutputType.FP32_OR_INT8, NodeInputOrOutputType.FP32_OR_INT8)
        else:
            return (NodeInputOrOutputType.UNKNOWN, NodeInputOrOutputType.UNKNOWN)

    elif node.op == 'call_module':
        assert node.op == 'call_module'
        assert isinstance(node.target, str)
        mod = getattr_from_fqn(gm, node.target)
        if isinstance(mod, logger_cls):  # type: ignore[arg-type]
            # A logger's input and output type is the output type of
            # the preceding node.
            first_arg = node.args[0]
            assert isinstance(first_arg, Node)
            _prev_node_input_type, prev_node_output_type = \
                get_node_first_input_and_output_type(
                    first_arg, gm, logger_cls, node_type_to_io_type_map)
            return (prev_node_output_type, prev_node_output_type)
        is_known_fp32_input_module = any(
            isinstance(mod, target_type) for target_type in MODS_IO_TYPE_FP32  # type: ignore[arg-type]
        )
        is_known_int8_input_module = any(
            isinstance(mod, target_type) for target_type in MODS_IO_TYPE_INT8  # type: ignore[arg-type]
        )
        is_known_fp32_or_int8_input_module = any(
            isinstance(mod, target_type) for target_type in MODS_IO_TYPE_FP32_OR_INT8  # type: ignore[arg-type]
        )
        if is_known_fp32_input_module:
            return (NodeInputOrOutputType.FP32, NodeInputOrOutputType.FP32)
        elif is_known_int8_input_module:
            return (NodeInputOrOutputType.INT8, NodeInputOrOutputType.INT8)
        elif is_known_fp32_or_int8_input_module:
            return (NodeInputOrOutputType.FP32_OR_INT8, NodeInputOrOutputType.FP32_OR_INT8)
        else:
            return (NodeInputOrOutputType.UNKNOWN, NodeInputOrOutputType.UNKNOWN)

    elif node.op == 'call_method':
        if node.target == 'dequantize':
            # Dequantize is a special node because it allows multiple input types.
            # So, we look up the output type of the previous node and return that
            # as the input type of this node instance.
            prev_node = node.args[0]
            assert isinstance(prev_node, Node)
            _prev_node_input_type, prev_node_output_type = \
                get_node_first_input_and_output_type(
                    prev_node, gm, logger_cls, node_type_to_io_type_map)
            return (prev_node_output_type, NodeInputOrOutputType.FP32)

        elif node.target == 'to':
            # to is a special node because it allows multiple input types.
            # So, we look up the output type of the previous node and return that
            # as the input type of this node instance. We also look up the target
            # of to and return the correct output type.
            prev_node = node.args[0]
            assert isinstance(prev_node, Node)
            _prev_node_input_type, prev_node_output_type = \
                get_node_first_input_and_output_type(
                    prev_node, gm, logger_cls, node_type_to_io_type_map)

            cur_node_dtype_target = node.args[1]
            assert cur_node_dtype_target is torch.float16, \
                f"{cur_node_dtype_target} handling needs to be added"

            return (prev_node_output_type, NodeInputOrOutputType.FP16)

        elif node.target in METHS_IO_TYPE_FP32_OR_INT8:
            return (NodeInputOrOutputType.FP32_OR_INT8, NodeInputOrOutputType.FP32_OR_INT8)

        return (NodeInputOrOutputType.UNKNOWN, NodeInputOrOutputType.UNKNOWN)
    else:
        return (NodeInputOrOutputType.UNKNOWN, NodeInputOrOutputType.UNKNOWN)

def return_first_non_observer_node(
    node: Node,
    gm: GraphModule,
) -> Node:
    """
    If node is not an observer, returns it.  If node is an observer,
    navigates up the graph and returns the first parent which is not an
    observer.  For example,

    graph: (node_non_obs), node = node_non_obs : returns node_non_obs
    graph: (node_non_obs -> obs0), node = obs0 : returns node_non_obs
    graph: (node_non_obs -> obs0 -> fq0), node = fq0 : returns node_non_obs
    """
    if node.op == 'call_module':
        node_obj = getattr_from_fqn(gm, node.target)  # type: ignore[arg-type]
        if is_activation_post_process(node_obj):
            assert len(node.args) == 1
            assert isinstance(node.args[0], Node)
            node = node.args[0]
            # code duplication intended, not worth refactoring
            assert isinstance(node.target, str)
            node_obj = getattr_from_fqn(gm, node.target)
            if is_activation_post_process(node_obj):
                assert len(node.args) == 1
                assert isinstance(node.args[0], Node)
                node = node.args[0]
    return node

def get_number_of_non_param_args(
    node: Node,
    gm: GraphModule,
) -> int:
    """
    Assumes that all non-param args occur first. Returns the number of
    non-param args expected for a node.  For example, for

      F.linear(x, weight, bias)

    Returns 1, because x is a non-param arg and weight and bias are params.
    For

      lstm_mod(x, hid)

    Returns 2, because both x and hid are non-param args.
    """
    if node.op == 'call_module':
        node_obj = getattr_from_fqn(gm, node.target)  # type: ignore[arg-type]
        if isinstance(node_obj, nn.LSTM):
            return 2

    # default is 1
    return 1

def get_arg_indices_of_inputs_to_log(node: Node) -> List[int]:
    """
    Returns the indices of args of the node which we should attach
    loggers to, if input logging is enabled.

    For example,
    * for (x + y), returns [0, 1]
    * for (1 + y), returns [1]
    * for (x + 1), returns [0]
    * for (linear(x, w, b)) returns [0]
    * by default, returns [0]
    """
    if len(node.args) == 0:
        return []
    if (
        node.op == 'call_function' and (
            # TODO(future PR): use relationship map instead of hardcoding
            node.target in (torch.add, torch.ops.quantized.add, operator.add) or
            node.target in (torch.mul, torch.ops.quantized.mul, operator.mul)
        )
    ):
        result = []
        for i in range(2):
            if type(node.args[i]) == Node:
                result.append(i)
        return result
    return [0]

def get_target_type_str(node: Node, gm: GraphModule) -> str:
    """
    Returns a string representation of the type of the function or module
    pointed to by this node, or '' for other op types.
    """
    target_type = ''
    if node.op in ('call_function', 'call_method'):
        target_type = str(node.target)
    elif node.op == 'call_module':
        assert isinstance(node.target, str)
        target_mod = getattr_from_fqn(gm, node.target)
        target_type = str(type(target_mod))
    return target_type
