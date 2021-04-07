import enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
toq = torch.ops.quantized
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.quantization.fx.quantize import is_activation_post_process

from typing import Any, Tuple, Callable

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
    # TODO(future PRs): dynamic quant, fake quant, etc


# TODO(future PR): make configurable
# TODO(future PR): fill out coverage
fp32_funs = set([
    F.linear,
    F.conv1d,
    F.conv2d,
    F.conv3d,
    # TODO(future PR): move this to a new category, since
    # i/o can be fp32 or int8
    torch.cat,
    F.relu,
])

# TODO(future PR): make configurable
# TODO(future PR): fill out coverage
int8_funs = set([
    toq.linear,
    toq.linear_relu,
    toq.conv1d,
    toq.conv1d_relu,
    toq.conv2d,
    toq.conv2d_relu,
    toq.conv3d,
    toq.conv3d_relu,
    toq.cat,
])

# TODO(future PR): make configurable
# TODO(future PR): fill out coverage
fp32_mods = set([
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.LSTM,
    # note: nnqd.Linear is an instance of nnq.Linear, so this
    # check has to happen before the int8 module check
    nnqd.Linear,
    nnqd.LSTM,
])

# TODO(future PR): make configurable
# TODO(future PR): fill out coverage
int8_mods = set([
    nnq.Linear,
    nnq.Conv1d,
    nnq.Conv2d,
    nnq.Conv3d,
])


def get_node_first_input_and_output_type(
    node: Node,
    gm: GraphModule,
    logger_cls: Callable,
) -> Tuple[NodeInputOrOutputType, NodeInputOrOutputType]:
    if node.op == 'call_function':
        if node.target in fp32_funs:
            return (NodeInputOrOutputType.FP32, NodeInputOrOutputType.FP32)
        elif node.target in int8_funs:
            return (NodeInputOrOutputType.INT8, NodeInputOrOutputType.INT8)
        else:
            return (NodeInputOrOutputType.UNKNOWN, NodeInputOrOutputType.UNKNOWN)

    elif node.op == 'call_module':
        assert node.op == 'call_module'
        assert isinstance(node.target, str)
        mod = getattr_from_fqn(gm, node.target)
        if isinstance(mod, logger_cls):  # type: ignore
            # A logger's input and output type is the output type of
            # the preceding node.
            first_arg = node.args[0]
            assert isinstance(first_arg, Node)
            _prev_node_input_type, prev_node_output_type = \
                get_node_first_input_and_output_type(
                    first_arg, gm, logger_cls)
            return (prev_node_output_type, prev_node_output_type)
        # For now, hacky check to see which mod is in which namespace
        is_known_fp32_input_module = any(
            isinstance(mod, target_type) for target_type in fp32_mods
        )
        is_known_int8_input_module = any(
            isinstance(mod, target_type) for target_type in int8_mods
        )
        if is_known_fp32_input_module:
            return (NodeInputOrOutputType.FP32, NodeInputOrOutputType.FP32)
        elif is_known_int8_input_module:
            return (NodeInputOrOutputType.INT8, NodeInputOrOutputType.INT8)
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
                get_node_first_input_and_output_type(prev_node, gm, logger_cls)
            return (prev_node_output_type, NodeInputOrOutputType.FP32)

        elif node.target == 'to':
            # to is a special node because it allows multiple input types.
            # So, we look up the output type of the previous node and return that
            # as the input type of this node instance. We also look up the target
            # of to and return the correct output type.
            prev_node = node.args[0]
            assert isinstance(prev_node, Node)
            _prev_node_input_type, prev_node_output_type = \
                get_node_first_input_and_output_type(prev_node, gm, logger_cls)

            cur_node_dtype_target = node.args[1]
            assert cur_node_dtype_target is torch.float16, \
                f"{cur_node_dtype_target} handling needs to be added"

            return (prev_node_output_type, NodeInputOrOutputType.FP16)

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
        node_obj = getattr_from_fqn(gm, node.target)  # type: ignore
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
        node_obj = getattr_from_fqn(gm, node.target)  # type: ignore
        if isinstance(node_obj, nn.LSTM):
            return 2

    # default is 1
    return 1
