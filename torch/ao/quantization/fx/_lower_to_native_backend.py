# mypy: allow-untyped-defs
import operator
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.quantized.reference as nnqr
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.nn.quantized.modules.utils import WeightedQuantizedModule
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.quantization_mappings import get_quantized_operator
from torch.ao.quantization.utils import _parent_name
from torch.fx import GraphModule, map_arg, Node
from torch.fx.graph import Graph

from .utils import (
    collect_producer_nodes,
    create_node_from_old_node_preserve_meta,
    get_linear_prepack_op_for_dtype,
    get_new_attr_name_with_prefix,
    get_qconv_prepack_op,
    graph_module_from_producer_nodes,
)


QOP_TO_ARG_NAMES_TO_SKIP = {
    torch._ops.ops.quantized.hardswish: ["inplace"],
    torch._ops.ops.quantized.elu: ["inplace"],
    torch._ops.ops.quantized.dropout: ["inplace"],
    torch._ops.ops.quantized.instance_norm: [
        "running_mean",
        "running_var",
        "use_input_stats",
        "momentum",
    ],
}


def _is_node_in_list(node, modules, func_list, method_list, module_type_list):
    is_call_function = node.op == "call_function" and node.target in func_list
    is_call_method = node.op == "call_method" and node.target in method_list
    is_call_module = (
        node.op == "call_module" and type(modules[str(node.target)]) in module_type_list
    )
    return is_call_function, is_call_method, is_call_module


def is_fixed_qparams_node(node, modules):
    func_list = [
        torch.nn.functional.hardsigmoid,
        torch.nn.functional.sigmoid,
        torch.sigmoid,
        torch.tanh,
    ]
    method_list = [
        "hardsigmoid",
        "hardsigmoid_",
        "sigmoid",
        "sigmoid_",
        "tanh",
        "tanh_",
    ]
    module_type_list = [
        torch.nn.Hardsigmoid,
        torch.nn.Sigmoid,
        torch.nn.Tanh,
        torch.nn.Softmax,
    ]
    return _is_node_in_list(node, modules, func_list, method_list, module_type_list)


def is_default_node(node, modules):
    func_list = [
        torch.nn.functional.elu,
        torch.nn.functional.hardswish,
        torch.nn.functional.instance_norm,
        torch.nn.functional.layer_norm,
        torch.nn.functional.leaky_relu,
        torch.nn.functional.dropout,
    ]
    method_list: List[Any] = []
    module_type_list = [
        nnqr.ConvTranspose1d,
        nnqr.ConvTranspose2d,
        nnqr.ConvTranspose3d,
        torch.nn.ELU,
        torch.nn.LeakyReLU,
        torch.nn.Hardswish,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.Dropout,
        torch.nn.PReLU,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.ao.nn.intrinsic.BNReLU2d,
        torch.ao.nn.intrinsic.BNReLU3d,
    ]
    return _is_node_in_list(node, modules, func_list, method_list, module_type_list)


def is_copy_node(node, modules):
    func_list = [
        torch.adaptive_avg_pool1d,
        torch.nn.functional.adaptive_avg_pool2d,
        torch.nn.functional.adaptive_avg_pool3d,
        torch.nn.functional.hardtanh,
        torch.nn.functional.hardtanh_,
        torch.nn.functional.interpolate,
        torch.nn.functional.max_pool1d,
        torch.nn.functional.max_pool2d,
        torch.nn.functional.max_pool3d,
        torch.nn.functional.relu,
        torch.nn.functional.relu6,
        torch.avg_pool1d,
        torch._C._nn.avg_pool2d,
        torch._C._nn.avg_pool3d,
        torch.clamp,
        torch.flatten,
        torch.mean,
        operator.floordiv,
        # F.channel_shuffle and torch.channel_shuffle are essentially the same thing
        # so we only need to put one of them here
        torch.channel_shuffle,
    ]
    method_list = [
        "clamp",
        "mean",
        "relu",
        "relu_",
    ]
    module_type_list = [
        torch.nn.AdaptiveAvgPool1d,
        torch.nn.AdaptiveAvgPool2d,
        torch.nn.AdaptiveAvgPool3d,
        torch.nn.AvgPool1d,
        torch.nn.AvgPool2d,
        torch.nn.AvgPool3d,
        torch.nn.Hardtanh,
        torch.nn.MaxPool1d,
        torch.nn.MaxPool2d,
        torch.nn.MaxPool3d,
        torch.nn.ReLU,
        torch.nn.ReLU6,
        torch.nn.ChannelShuffle,
    ]
    return _is_node_in_list(node, modules, func_list, method_list, module_type_list)


def is_general_tensor_shape_node(node, modules):
    func_list = [
        torch.narrow,
        torch.transpose,
        torch.repeat_interleave,
        torch.squeeze,
        torch.stack,
        torch.unsqueeze,
        torch.nn.functional.pixel_shuffle,
        torch.nn.functional.pixel_unshuffle,
    ]
    method_list = [
        "contiguous",
        "detach",
        "detach_",
        "permute",
        "repeat",
        "repeat_interleave",
        "reshape",
        "resize_",
        "shape",
        "size",
        "squeeze",
        "squeeze_",
        "transpose",
        "unsqueeze",
        "unsqueeze_",
        "view",
    ]
    module_type_list = [
        torch.nn.Identity,
        torch.nn.PixelShuffle,
        torch.nn.PixelUnshuffle,
    ]
    return _is_node_in_list(node, modules, func_list, method_list, module_type_list)


def is_other_node(node, modules):
    func_list = [
        torch.cat,
    ]
    method_list: List[Any] = []
    module_type_list: List[Any] = []
    return _is_node_in_list(node, modules, func_list, method_list, module_type_list)


def is_special_pattern_node(node, modules):
    res_function, res_method, res_module = False, False, False
    for checker in [
        is_fixed_qparams_node,
        is_default_node,
        is_copy_node,
        is_general_tensor_shape_node,
        is_other_node,
    ]:
        is_call_function, is_call_method, is_call_module = checker(node, modules)
        res_function = res_function or is_call_function
        res_method = res_method or is_call_method
        res_module = res_module or is_call_module
    return res_function, res_method, res_module


def is_dequantize_node(node):
    return (
        isinstance(node, Node)
        and node.op == "call_method"
        and node.target == "dequantize"
    )


def is_getattr_tensor_metadata_node(node):
    return (
        node.op == "call_function"
        and node.target == getattr
        and node.args[1] in ["shape"]
    )


def is_get_tensor_info_node(node):
    return node.op == "call_method" and node.target in ["shape", "size"]


def should_skip_lowering(op: torch.fx.node.Node, qconfig_map: Dict[str, QConfigAny]):
    """
    Return True if the op is configured with a None qconfig, False otherwise.
    Note: maybe need to generalize this to also check for the dtype, and we
    only lower when dtype matches, but right now fbgemm/qnnpack only support
    a single dtype, so it is OK for now.
    """
    return op.name in qconfig_map and qconfig_map[op.name] is None


# Mapping from reference module class to the replacement static quantized module class for lowering
STATIC_LOWER_MODULE_MAP: Dict[Type[nn.Module], Type[WeightedQuantizedModule]] = {
    nnqr.Linear: nnq.Linear,
    nnqr.Conv1d: nnq.Conv1d,
    nnqr.Conv2d: nnq.Conv2d,
    nnqr.Conv3d: nnq.Conv3d,
}

# Mapping from reference module class to the replacement dynamic quantized module class for lowering
DYNAMIC_LOWER_MODULE_MAP: Dict[Type[nn.Module], Type[nn.Module]] = {
    nnqr.Linear: nnqd.Linear,
    nnqr.GRUCell: nnqd.GRUCell,
    nnqr.LSTMCell: nnqd.LSTMCell,
    nnqr.RNNCell: nnqd.RNNCell,
    nnqr.LSTM: nnqd.LSTM,
    nnqr.GRU: nnqd.GRU,
}

# Mapping from reference module class to the replacement weight only quantized module class for lowering
# TODO: correct the namespace for these modules
WEIGHT_ONLY_LOWER_MODULE_MAP: Dict[Type[nn.Module], Type[nn.Module]] = {
    nnqr.Embedding: nnq.Embedding,
    nnqr.EmbeddingBag: nnq.EmbeddingBag,
}

# TODO: merge with STATIC_LOWER_MODULE_MAP after we merge
# _lower_static_weighted_ref_module and special_pattern_replacement
SPECIAL_PATTERN_LOWER_MODULE_MAP = {
    nn.BatchNorm2d: nnq.BatchNorm2d,
    nn.BatchNorm3d: nnq.BatchNorm3d,
    nnqr.ConvTranspose1d: nnq.ConvTranspose1d,
    nnqr.ConvTranspose2d: nnq.ConvTranspose2d,
    nnqr.ConvTranspose3d: nnq.ConvTranspose3d,
    nn.ELU: nnq.ELU,
    nn.LeakyReLU: nnq.LeakyReLU,
    nn.Hardswish: nnq.Hardswish,
    nn.InstanceNorm1d: nnq.InstanceNorm1d,
    nn.InstanceNorm2d: nnq.InstanceNorm2d,
    nn.InstanceNorm3d: nnq.InstanceNorm3d,
    nn.LayerNorm: nnq.LayerNorm,
    nn.Dropout: nnq.Dropout,
    nn.Softmax: nnq.Softmax,
    nn.PReLU: nnq.PReLU,
    nni.BNReLU2d: nniq.BNReLU2d,
    nni.BNReLU3d: nniq.BNReLU3d,
}

# Mapping from fused module class to a 2-tuple of:
#   1) The inner reference module class
#   2) The replacement static quantized module class for lowering
STATIC_LOWER_FUSED_MODULE_MAP: Dict[
    Type[nn.Module], Tuple[Type[nn.Module], Type[WeightedQuantizedModule]]
] = {
    nni.LinearReLU: (nnqr.Linear, nniq.LinearReLU),
    # TODO: LinearLeakyReLU is registered as global but it is only fused and
    # lowered when ondnn's backend config is used. Maybe need to separate
    # registration and lowering functions for different backends in the future.
    nni.LinearLeakyReLU: (nnqr.Linear, nniq.LinearLeakyReLU),
    nni.LinearTanh: (nnqr.Linear, nniq.LinearTanh),
    nni.ConvReLU1d: (nnqr.Conv1d, nniq.ConvReLU1d),
    nni.ConvReLU2d: (nnqr.Conv2d, nniq.ConvReLU2d),
    nni.ConvReLU3d: (nnqr.Conv3d, nniq.ConvReLU3d),
}

# The difference between STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP and STATIC_LOWER_FUSED_MODULE_MAP:
# The refer node inside STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP has 2 inputs.
# Mapping from fused module class to a 2-tuple of:
#   1) The inner reference module class
#   2) The replacement static quantized module class for lowering
STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP: Dict[
    Type[nn.Module], Tuple[Type[nn.Module], Type[WeightedQuantizedModule]]
] = {
    nni.ConvAdd2d: (nnqr.Conv2d, nniq.ConvAdd2d),
    nni.ConvAddReLU2d: (nnqr.Conv2d, nniq.ConvAddReLU2d),
}

# Mapping from fused module class to a 2-tuple of:
#   1) The inner reference module class
#   2) The replacement dynamic quantized module class for lowering
DYNAMIC_LOWER_FUSED_MODULE_MAP: Dict[
    Type[nn.Module], Tuple[Type[nn.Module], Type[nn.Module]]
] = {
    nni.LinearReLU: (nnqr.Linear, nniqd.LinearReLU),
}

# Mapping from a functional to lower to a 2-tuple of
#   1) The quantized version of the op
#   2) The quantized version of the op fused with relu, if it exists, else None
STATIC_LOWER_FUNCTIONAL_MAP: Dict[Callable, Tuple[Callable, Optional[Callable]]] = {
    F.linear: (torch.ops.quantized.linear, torch.ops.quantized.linear_relu),
    F.conv1d: (torch.ops.quantized.conv1d, torch.ops.quantized.conv1d_relu),
    F.conv2d: (torch.ops.quantized.conv2d, torch.ops.quantized.conv2d_relu),
    F.conv3d: (torch.ops.quantized.conv3d, torch.ops.quantized.conv3d_relu),
    F.conv_transpose1d: (torch.ops.quantized.conv_transpose1d, None),
    F.conv_transpose2d: (torch.ops.quantized.conv_transpose2d, None),
    F.conv_transpose3d: (torch.ops.quantized.conv_transpose3d, None),
}

WEIGHT_PREPACK_OPS: Set[Callable] = {
    torch._ops.ops.quantized.linear_prepack,
    torch._ops.ops.quantized.linear_prepack_fp16,
    torch._ops.ops.quantized.conv1d_prepack,
    torch._ops.ops.quantized.conv2d_prepack,
    torch._ops.ops.quantized.conv3d_prepack,
    torch.ops.quantized.conv_transpose1d_prepack,
    torch.ops.quantized.conv_transpose2d_prepack,
    torch.ops.quantized.conv_transpose3d_prepack,
}

# Mapping from a functional to a dictionary, where the key is a 2-tuple of
# (input_activation_dtype, weight_dtype) and the value is a 2-tuple of
#   1) The dynamically quantized version of the op
#   2) The dynamically quantized version of the op fused with relu, if it exists, else None
DYNAMIC_LOWER_FUNCTIONAL_MAP: Dict[
    Callable, Dict[Tuple[torch.dtype, torch.dtype], Tuple[Callable, Optional[Callable]]]
] = {
    F.linear: {
        (torch.quint8, torch.qint8): (
            torch.ops.quantized.linear_dynamic,
            torch.ops.quantized.linear_relu_dynamic,
        ),
        (torch.float16, torch.float16): (
            torch.ops.quantized.linear_dynamic_fp16,
            torch.ops.quantized.linear_relu_dynamic_fp16,
        ),
    },
    # dynamic conv + relu is not available yet
    F.conv1d: {
        (torch.quint8, torch.qint8): (torch.ops.quantized.conv1d_dynamic, None),
    },
    F.conv2d: {
        (torch.quint8, torch.qint8): (torch.ops.quantized.conv2d_dynamic, None),
    },
    F.conv3d: {
        (torch.quint8, torch.qint8): (torch.ops.quantized.conv3d_dynamic, None),
    },
}

CONV_FUNCTIONAL_OPS: Set[Callable] = {
    F.conv1d,
    F.conv2d,
    F.conv3d,
}

CONV_TRANSPOSE_FUNCTIONAL_OPS: Set[Callable] = {
    F.conv_transpose1d,
    F.conv_transpose2d,
    F.conv_transpose3d,
}

# TODO: add tests for lowering these ops
QBIN_OP_MAPPING: Dict[Union[Callable, str], Callable] = {
    operator.add: torch.ops.quantized.add,
    torch.add: torch.ops.quantized.add,
    operator.mul: torch.ops.quantized.mul,
    operator.matmul: torch.ops.quantized.matmul,
    torch.mul: torch.ops.quantized.mul,
    torch.matmul: torch.ops.quantized.matmul,
}
QBIN_RELU_OP_MAPPING: Dict[Union[Callable, str], Callable] = {
    operator.add: torch.ops.quantized.add_relu,
    torch.add: torch.ops.quantized.add_relu,
    operator.mul: torch.ops.quantized.mul_relu,
    torch.mul: torch.ops.quantized.mul_relu,
}


def _save_packed_weight(self, destination, prefix, keep_vars):
    for attr_name in dir(self):
        if "_packed_weight" in attr_name and isinstance(
            getattr(self, attr_name), torch._C.ScriptObject
        ):  # type: ignore[attr-defined]
            packed_weight = getattr(self, attr_name)
            destination[prefix + attr_name] = packed_weight


def _load_packed_weight(
    self,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    attrs_to_pop = []
    for attr_name in state_dict:
        if attr_name.startswith("_packed_weight") and isinstance(state_dict[attr_name], torch._C.ScriptObject):  # type: ignore[attr-defined] # noqa: B950
            setattr(self, attr_name, state_dict[attr_name])
            attrs_to_pop.append(attr_name)

    # pop the packed param attributesn
    for attr_name in attrs_to_pop:
        state_dict.pop(attr_name)


def fold_weight(
    quantized_model: GraphModule, node_name_to_scope: Dict[str, Tuple[str, type]]
) -> GraphModule:
    """
    Trace back from the weight node util we hit getattr, reconstruct the
    graph module with the traced nodes and run the graph module to pack the
    weight. then replace the original chain of ops with the packed weight.
    """
    packed_weights = {}
    # map from folded node name to the prepacked weight name
    folded_nodes = {}
    # get packed weights
    for node in quantized_model.graph.nodes:
        if node.op == "call_function" and node.target in WEIGHT_PREPACK_OPS:
            nodes_to_fold = collect_producer_nodes(node)
            if nodes_to_fold is not None:
                for node_to_fold in nodes_to_fold:
                    folded_nodes[node_to_fold.name] = node

                prepacking_module = graph_module_from_producer_nodes(
                    quantized_model, nodes_to_fold
                )
                packed_weight = prepacking_module()
                packed_weights[node.name] = packed_weight

    # remove folded nodes and replace the prepacking node with getattr
    folded_graph = Graph()
    env: Dict[Any, Any] = {}

    def load_arg(a):
        return map_arg(a, lambda node: env[node.name])

    for node in quantized_model.graph.nodes:
        prepack_node = folded_nodes.get(node.name, None)
        if prepack_node is node:
            packed_weight = packed_weights[node.name]
            # add a prepacked attribute to root
            op_node = next(iter(prepack_node.users))
            module_path, _ = node_name_to_scope[op_node.name]
            get_new_packed_weight_name = get_new_attr_name_with_prefix(
                module_path + "_packed_weight_"
            )
            packed_weight_name = get_new_packed_weight_name(quantized_model)
            setattr(quantized_model, packed_weight_name, packed_weight)
            # replace prepack node with a getattr node
            env[node.name] = folded_graph.create_node(
                "get_attr", packed_weight_name, (), {}
            )
        elif prepack_node is not None:
            # remove the foled node
            continue
        else:
            # copy other nodes
            env[node.name] = folded_graph.node_copy(node, load_arg)

    quantized_model = GraphModule(quantized_model, folded_graph)
    quantized_model._register_state_dict_hook(_save_packed_weight)
    quantized_model.register_load_state_dict_pre_hook(_load_packed_weight)
    return quantized_model


def _get_module(node: Node, modules: Dict[str, nn.Module]) -> Optional[nn.Module]:
    """
    Return the `torch.nn.Module` that corresponds to the specified node's target.
    If no such node exists, return None.
    """
    if node.op == "call_module" and str(node.target) in modules:
        return modules[str(node.target)]
    else:
        return None


def _match_static_pattern(
    node: Node,
    modules: Dict[str, nn.Module],
    qconfig_map: Dict[str, QConfigAny],
    matching_modules_or_ops: List[Callable],
    dequantize_node_arg_indices: List[int],
) -> Union[Tuple[Node, Node, Node], Tuple[None, None, None]]:
    """
    Match the pattern (dequantize - ref node - quantize) against the node provided.

    If there is a match, return a 3-tuple of:
      1) q_node: the quantize node,
      2) relu_node: a relu node wrapping the ref_node, and
      3) ref_node: a reference module or functional node to replace with its quantized counterpart
    Otherwise, if there is no match, return a 3-tuple of (None, None, None).

    Parameters:
      node: The `torch.fx.Node` to match against.
      modules: A mapping from node names to modules in the model graph, used for module lookup.
      qconfig_map: A mapping from node names to the qconfigs associated with the nodes.
          If the corresponding qconfig for the reference node is None, then return no match.
      matching_modules_or_ops: Either a list of functions or a list of `torch.nn.Module`s.
          If the reference node is not in this list, then return no match.
      dequantize_node_arg_indices: A list of indices in the reference node args where dequantize
          nodes may be present. An empty list means skipping the check for dequantize nodes.
    """
    SKIP_LOWERING_VALUE = (None, None, None)

    # Match quantize node
    if node.op != "call_function" or node.target != torch.quantize_per_tensor:
        return SKIP_LOWERING_VALUE
    q_node = node
    ref_node = q_node.args[0]
    assert isinstance(ref_node, Node)

    # Handle cases where the node is wrapped in a ReLU
    if (ref_node.op == "call_function" and ref_node.target in (F.relu, torch.relu)) or (
        ref_node.op == "call_module" and type(_get_module(ref_node, modules)) == nn.ReLU
    ):
        relu_node = ref_node
        ref_node = relu_node.args[0]
        assert isinstance(ref_node, Node)
    else:
        relu_node = None
    if should_skip_lowering(ref_node, qconfig_map):
        return SKIP_LOWERING_VALUE

    # Match reference module or functional
    if isinstance(matching_modules_or_ops[0], type) and issubclass(
        matching_modules_or_ops[0], nn.Module
    ):
        expected_op = "call_module"
        match_key = type(_get_module(ref_node, modules))
    else:
        expected_op = "call_function"
        match_key = ref_node.target  # type: ignore[assignment]
    if ref_node.op != expected_op or match_key not in matching_modules_or_ops:
        return SKIP_LOWERING_VALUE

    # Match dequantize node(s). Both of the following conditions must pass:
    # (1) All `torch.fx.Node`s at the matching indices must be a dequantize node
    # (2) There must be at least one dequantize node
    matched_dequantize = False
    for i in dequantize_node_arg_indices:
        assert i < len(
            ref_node.args
        ), f"Dequantize index {i} exceeded reference node's arg length {len(ref_node.args)}"
        arg = ref_node.args[i]
        if is_dequantize_node(arg):
            matched_dequantize = True
        elif isinstance(arg, Node):
            return SKIP_LOWERING_VALUE
    if not matched_dequantize:
        return SKIP_LOWERING_VALUE

    return (q_node, relu_node, ref_node)  # type: ignore[return-value]


def _match_static_pattern_with_two_inputs(
    node: Node,
    modules: Dict[str, nn.Module],
    qconfig_map: Dict[str, QConfigAny],
    matching_modules_or_ops: List[Callable],
) -> Union[Tuple[Node, Node], Tuple[None, None]]:
    """
                      (dequantize \
    Match the pattern (dequantize - ref node - quantize) against the node provided.

    If there is a match, return a 2-tuple of:
      1) q_node: the quantize node,
      2) ref_node: a reference module or functional node to replace with its quantized counterpart
    Otherwise, if there is no match, return a 2-tuple of (None, None).

    Parameters:
      node: The `torch.fx.Node` to match against.
      modules: A mapping from node names to modules in the model graph, used for module lookup.
      qconfig_map: A mapping from node names to the qconfigs associated with the nodes.
          If the corresponding qconfig for the reference node is None, then return no match.
      matching_modules_or_ops: Either a list of functions or a list of `torch.nn.Module`s.
          If the reference node is not in this list, then return no match.
    """
    SKIP_LOWERING_VALUE = (None, None)

    # Match quantize node
    if node.op != "call_function" or node.target != torch.quantize_per_tensor:
        return SKIP_LOWERING_VALUE
    q_node = node
    ref_node = q_node.args[0]
    assert isinstance(ref_node, Node)

    if should_skip_lowering(ref_node, qconfig_map):
        return SKIP_LOWERING_VALUE

    # Match reference module or functional
    if isinstance(matching_modules_or_ops[0], type) and issubclass(
        matching_modules_or_ops[0], nn.Module
    ):
        expected_op = "call_module"
        match_key = type(_get_module(ref_node, modules))
    else:
        # This pass only support op of "call_module"
        return SKIP_LOWERING_VALUE

    if ref_node.op != expected_op or match_key not in matching_modules_or_ops:
        return SKIP_LOWERING_VALUE

    # Check ref_node has 2 input nodes, both are dq node.
    if len(ref_node.args) != 2:
        return SKIP_LOWERING_VALUE
    for i in range(len(ref_node.args)):
        arg = ref_node.args[i]
        if not is_dequantize_node(arg):
            return SKIP_LOWERING_VALUE

    return (q_node, ref_node)


def _lower_static_weighted_ref_module(
    model: GraphModule, qconfig_map: Dict[str, QConfigAny]
):
    """
    Traverse the graph and find dequantize - ref module - quantize patterns
    and replace them with the quantized version of the ref module.
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    for n in model.graph.nodes:
        # Step 0: Find nodes that match this pattern (dequantize - ref module - quantize)
        matching_modules = list(STATIC_LOWER_MODULE_MAP.keys()) + list(
            STATIC_LOWER_FUSED_MODULE_MAP.keys()
        )
        (q_node, _, ref_node) = _match_static_pattern(
            n, modules, qconfig_map, matching_modules, dequantize_node_arg_indices=[0]  # type: ignore[arg-type]
        )
        if q_node is None:
            continue
        assert ref_node is not None
        (_, scale_node, zero_point_node, _) = q_node.args
        ref_module = _get_module(ref_node, modules)
        ref_class = type(ref_module)
        assert isinstance(scale_node, Node)
        assert isinstance(zero_point_node, Node)
        assert issubclass(ref_class, nn.Module)

        # Step 1: Change this pattern to use the corresponding quantized module
        # For fused modules, we also check whether the inner module is a reference module
        # If so, we replace the entire fused module with the corresponding quantized module
        if ref_class in STATIC_LOWER_FUSED_MODULE_MAP:
            inner_ref_class, q_class = STATIC_LOWER_FUSED_MODULE_MAP[ref_class]
            if type(ref_module[0]) != inner_ref_class:  # type: ignore[index]
                continue
        else:
            q_class = STATIC_LOWER_MODULE_MAP[ref_class]
        output_scale = getattr(model, scale_node.target)  # type: ignore[arg-type]
        output_zero_point = getattr(model, zero_point_node.target)  # type: ignore[arg-type]
        q_module = q_class.from_reference(ref_module, output_scale, output_zero_point)
        # replace reference module with quantized module
        parent_name, module_name = _parent_name(ref_node.target)
        setattr(modules[parent_name], module_name, q_module)

        # Step 2: Reroute around dq_node, and remove q_node and its args
        assert len(ref_node.args) == 1
        dq_node = ref_node.args[0]
        assert isinstance(dq_node, Node)
        ref_node.replace_input_with(dq_node, dq_node.args[0])  # type: ignore[arg-type]
        q_node.replace_all_uses_with(ref_node)
        model.graph.erase_node(q_node)
        model.graph.erase_node(scale_node)
        model.graph.erase_node(zero_point_node)


def _lower_static_weighted_ref_module_with_two_inputs(
    model: GraphModule, qconfig_map: Dict[str, QConfigAny]
):
    """
    Traverse the graph and find patterns
    dequantize   dequantize
       \\         //
        ref module
            \\
          quantize
    and replace them with the quantized version of the ref module.
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    for n in model.graph.nodes:
        #                                            (dequantize \
        # Step 0: Find nodes that match this pattern (dequantize - ref module - quantize)
        matching_modules = list(STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP.keys())
        (q_node, ref_node) = _match_static_pattern_with_two_inputs(
            n, modules, qconfig_map, matching_modules  # type: ignore[arg-type]
        )
        if q_node is None:
            continue
        assert ref_node is not None
        (_, scale_node, zero_point_node, _) = q_node.args
        ref_module = _get_module(ref_node, modules)
        ref_class = type(ref_module)
        assert isinstance(scale_node, Node)
        assert isinstance(zero_point_node, Node)
        assert issubclass(ref_class, nn.Module)

        # Step 1: Change this pattern to use the corresponding quantized module
        # For fused modules, we also check whether the inner module is a reference module
        # If so, we replace the entire fused module with the corresponding quantized module
        if ref_class in STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP:
            inner_ref_class, q_class = STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP[
                ref_class
            ]
            if type(ref_module[0]) != inner_ref_class:  # type: ignore[index]
                continue
        else:
            continue
        output_scale = getattr(model, scale_node.target)  # type: ignore[arg-type]
        output_zero_point = getattr(model, zero_point_node.target)  # type: ignore[arg-type]
        q_module = q_class.from_reference(ref_module, output_scale, output_zero_point)
        # replace reference module with quantized module
        parent_name, module_name = _parent_name(ref_node.target)
        setattr(modules[parent_name], module_name, q_module)

        # Step 2: Reroute around dq_node, and remove q_node and its args
        assert len(ref_node.args) == 2
        for arg in ref_node.args:
            if not is_dequantize_node(arg):
                continue
            dq_node = arg
            assert isinstance(dq_node, Node)
            ref_node.replace_input_with(dq_node, dq_node.args[0])  # type: ignore[arg-type]

        q_node.replace_all_uses_with(ref_node)
        model.graph.erase_node(q_node)
        model.graph.erase_node(scale_node)
        model.graph.erase_node(zero_point_node)


def _lower_dynamic_weighted_ref_module(model: GraphModule):
    """
    Traverse the graph and find quantize_per_tensor_dynamic - dequantize - ref_module patterns
    and replace them with the dynamically quantized version of the ref module.
    """
    named_modules = dict(model.named_modules(remove_duplicate=False))
    for n in model.graph.nodes:
        if n.op != "call_module" or type(named_modules[str(n.target)]) not in set(
            DYNAMIC_LOWER_MODULE_MAP.keys()
        ).union(set(DYNAMIC_LOWER_FUSED_MODULE_MAP.keys())):
            continue
        ref_node = n
        dq_node = ref_node.args[0]
        if dq_node.op != "call_method" or dq_node.target != "dequantize":
            continue

        input_dynamic_q_node = dq_node.args[0]

        if (
            input_dynamic_q_node.op != "call_function"
            or input_dynamic_q_node.target != torch.quantize_per_tensor_dynamic
        ):
            continue

        activation_dtype = input_dynamic_q_node.args[1]
        is_fp16 = activation_dtype == torch.float16
        is_int8 = activation_dtype in [torch.quint8, torch.qint8]
        if not is_int8 and not is_fp16:
            continue

        ref_module = named_modules[str(ref_node.target)]
        ref_class = type(ref_module)
        if ref_class in DYNAMIC_LOWER_FUSED_MODULE_MAP:
            inner_ref_class, q_class = DYNAMIC_LOWER_FUSED_MODULE_MAP[ref_class]
            if type(ref_module[0]) != inner_ref_class:
                continue
        else:
            q_class = DYNAMIC_LOWER_MODULE_MAP.get(ref_class)  # type: ignore[assignment]
        # TODO: maybe define a WeightedDynamicallyQuantizedModule
        q_module = q_class.from_reference(ref_module)  # type: ignore[attr-defined]

        # replace reference module with dynamically quantized module
        parent_name, module_name = _parent_name(ref_node.target)
        setattr(named_modules[parent_name], module_name, q_module)
        ref_node.replace_input_with(dq_node, input_dynamic_q_node.args[0])


def _lower_weight_only_weighted_ref_module(model: GraphModule):
    """
    Traverse the graph and find ref_module patterns
    and replace them with the weight only quantized version of the ref module.
    """
    named_modules = dict(model.named_modules(remove_duplicate=False))
    for n in model.graph.nodes:
        if n.op != "call_module" or type(named_modules[str(n.target)]) not in set(
            WEIGHT_ONLY_LOWER_MODULE_MAP.keys()
        ):
            continue
        ref_node = n
        ref_module = named_modules[str(ref_node.target)]
        ref_class = type(ref_module)
        q_class = WEIGHT_ONLY_LOWER_MODULE_MAP.get(ref_class)
        # TODO: WeightedQuantizedModule is currently assuming static quant apis
        # with output_scale, output_zero_point in from_reference, we may want to
        # relax that, or rename this
        # TODO: maybe define a WeightedWeightOnlyQuantizedModule
        q_module = q_class.from_reference(ref_module)  # type: ignore[union-attr]

        # replace reference module with dynamically quantized module
        parent_name, module_name = _parent_name(ref_node.target)
        setattr(named_modules[parent_name], module_name, q_module)


def _lower_static_weighted_ref_functional(
    model: GraphModule, qconfig_map: Dict[str, QConfigAny]
):
    """
    Traverse the graph and replace functional reference patterns with their quantized versions.
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    for n in model.graph.nodes:
        # Step 0: Find nodes that match this pattern (dequantize - functional op - quantize)
        matching_ops = list(STATIC_LOWER_FUNCTIONAL_MAP.keys())
        (q_node, relu_node, func_node) = _match_static_pattern(
            n, modules, qconfig_map, matching_ops, dequantize_node_arg_indices=[0, 1]
        )
        if q_node is None:
            continue
        assert func_node is not None
        (_, output_scale_node, output_zp_node, _) = q_node.args
        (input_dq_node, weight_dq_node, *remaining_func_args) = func_node.args
        assert isinstance(output_zp_node, Node)
        assert isinstance(input_dq_node, Node)
        assert isinstance(weight_dq_node, Node)
        quantized_weight = weight_dq_node.args[0]
        assert isinstance(quantized_weight, Node)
        if quantized_weight.op != "call_function" or quantized_weight.target not in (
            torch.quantize_per_tensor,
            torch.quantize_per_channel,
        ):
            continue

        # Step 1: Replace quantized weights with packed weights, which will be folded later
        # Use the right prepack op and prepare the corresponding args
        # Linear prepack args: (quantized weights[, bias])
        # Conv prepack args: (quantized weights[, bias, stride, padding, dilation, groups])
        prepack_args = [quantized_weight] + remaining_func_args
        if func_node.target == F.linear:
            weight_dtype = quantized_weight.args[-1]
            prepack_op = get_linear_prepack_op_for_dtype(weight_dtype)
        elif func_node.target in CONV_FUNCTIONAL_OPS:
            prepack_op = get_qconv_prepack_op(func_node.target)  # type: ignore[arg-type]
            # For conv1d, the stride, padding, and dilation args may be ints,
            # in which case we need to convert them to tuples
            if func_node.target == F.conv1d:
                for i in [2, 3, 4]:
                    if len(prepack_args) > i and isinstance(prepack_args[i], int):
                        prepack_args[i] = (prepack_args[i],)
        elif func_node.target in CONV_TRANSPOSE_FUNCTIONAL_OPS:
            prepack_op = get_qconv_prepack_op(func_node.target)  # type: ignore[arg-type]
            # For conv_transpose1d, the stride, padding, and dilation args may be ints,
            # in which case we need to convert them to tuples
            if func_node.target == F.conv_transpose1d:
                # Note prepack_args[5] is groups.
                for i in [2, 3, 4, 6]:
                    if len(prepack_args) > i and isinstance(prepack_args[i], int):
                        prepack_args[i] = (prepack_args[i],)
            # swap dilation and groups
            # prepack op has arguments: {w, b, stride, padding, output_padding, dilation, groups}
            # transposed conv op has arguments: {x, w, b, stride, padding, output_padding, groups, dilation}
            if len(prepack_args) > 6:
                prepack_args[5], prepack_args[6] = prepack_args[6], prepack_args[5]
        else:
            raise ValueError(f"Lowering is not supported for op '{func_node.target}'")
        with model.graph.inserting_before(output_scale_node):  # type: ignore[arg-type]
            # kwargs of the func node are needed for prepack op (i.e., quantized::linear_prepack)
            # They are not needed for compute op (i.e., quantized::linear)
            kwargs = func_node.kwargs
            # F.linear uses 'bias' key for bias while qlinear_prepack uses 'B' for bias
            if func_node.target == F.linear and "bias" in kwargs:
                kwargs = kwargs.copy()
                kwargs["B"] = kwargs["bias"]
                del kwargs["bias"]
            packed_weight = model.graph.create_node(
                "call_function", prepack_op, tuple(prepack_args), kwargs
            )

        # Step 2: Replace reference pattern with the corresponding quantized op
        (q_func, q_relu_func) = STATIC_LOWER_FUNCTIONAL_MAP[func_node.target]  # type: ignore[index]
        # conv_transpose does not support fusion with relu yet. q_relu_func is None in such cases
        if q_relu_func is not None:
            func_node.target = q_relu_func if relu_node is not None else q_func
        else:
            func_node.target = q_func
        func_node.args = (
            input_dq_node.args[0],
            packed_weight,
            output_scale_node,
            output_zp_node,
        )
        # kwargs for func_node has been moved to kwargs for prepack op
        func_node.kwargs = {}
        q_node.replace_all_uses_with(func_node)
        # Move func_node after output_zp_node in the graph
        output_zp_node.append(func_node)

        # Clean up: Remove quantize node, and the relu node if it exists
        model.graph.erase_node(q_node)
        if relu_node is not None and q_relu_func is not None:
            model.graph.erase_node(relu_node)


def _lower_dynamic_weighted_ref_functional(
    model: GraphModule, qconfig_map: Dict[str, QConfigAny]
):
    """
    Traverse the graph and replace functional reference patterns with their dynamically
    quantized versions.
    Examples:
    quantize_per_tensor_dynamic - dequantize - functional linear --> linear_dynamic
    to(torch.float16) - dequantize - functional linear --> linear_dynamic_fp16
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    # we want to search in reserved order so that we can match the larger patterns first
    # e.g. we want to match linear - relu before linear.
    for n in reversed(model.graph.nodes):
        # Step 0: Find nodes that match this pattern
        # (quantize_per_tensor_dynamic - dequantize - dynamically quantized op)
        # We search for the pattern backwards, starting with the quantize node
        # Quantize node args: (func, scale, zp, dtype)
        func_node = n
        # Handle cases where the functional op is wrapped in a ReLU
        if (
            func_node.op == "call_function"
            and func_node.target == F.relu
            or func_node.op == "call_module"
            and type(modules[str(func_node.target)]) == torch.nn.ReLU
        ):
            relu_node = func_node
            func_node = relu_node.args[0]
        else:
            relu_node = None
        if should_skip_lowering(func_node, qconfig_map):
            continue
        # Linear args: (dequantized inputs, dequantized weights[, bias])
        # Conv args: (dequantized inputs, dequantized weights[, bias, stride, padding, dilation, groups])
        if (
            func_node.op != "call_function"
            or func_node.target not in DYNAMIC_LOWER_FUNCTIONAL_MAP
        ):
            continue
        (input_dq_node, weight_dq_node, *remaining_func_args) = func_node.args
        if (
            input_dq_node.op != "call_method"
            or input_dq_node.target != "dequantize"
            or weight_dq_node.op != "call_method"
            or weight_dq_node.target != "dequantize"
        ):
            continue

        input_dynamic_q_node = input_dq_node.args[0]

        if (
            input_dynamic_q_node.op != "call_function"
            or input_dynamic_q_node.target != torch.quantize_per_tensor_dynamic
        ):
            continue

        reduce_range_node = None
        (pattern_input, activation_dtype, reduce_range_node) = input_dynamic_q_node.args
        is_fp16 = activation_dtype == torch.float16
        is_int8 = activation_dtype in [torch.quint8, torch.qint8]
        if not is_int8 and not is_fp16:
            continue

        quantized_weight = weight_dq_node.args[0]
        weight_dtype = quantized_weight.args[-1]

        # Step 1: Try to select reference pattern with the corresponding quantized op
        dynamic_quant_dtype_key = (activation_dtype, weight_dtype)
        if (
            dynamic_quant_dtype_key
            not in DYNAMIC_LOWER_FUNCTIONAL_MAP[func_node.target]
        ):
            print(
                f"Didn't find dtype combination {dynamic_quant_dtype_key} during "
                f"dynamic quantized op lowering for {func_node.target}"
            )
            continue
        (q_func, q_relu_func) = DYNAMIC_LOWER_FUNCTIONAL_MAP[func_node.target][
            dynamic_quant_dtype_key
        ]

        if q_func is None or q_relu_func is None:
            print(
                "Didn't find corresponding quantized function or quantized relu function "
                f"for {func_node.target}, {dynamic_quant_dtype_key}"
            )
            continue

        # Step 2: Replace quantized weights with packed weights, which will be folded later
        # Use the right prepack op and prepare the corresponding args
        # Linear prepack args: (quantized weights[, bias])
        # Conv prepack args: (quantized weights[, bias, stride, padding, dilation, groups])
        prepack_args = [quantized_weight] + remaining_func_args
        prepack_kwargs = {}
        if func_node.target == F.linear:
            prepack_op = get_linear_prepack_op_for_dtype(weight_dtype)
            kwargs = func_node.kwargs.copy()
            if "bias" in kwargs:
                prepack_kwargs["B"] = kwargs["bias"]
                del kwargs["bias"]
                func_node.kwargs = kwargs
        elif func_node.target in CONV_FUNCTIONAL_OPS:
            prepack_op = get_qconv_prepack_op(func_node.target)
            # For conv1d, the stride, padding, and dilation args may be ints,
            # in which case we need to convert them to tuples
            if func_node.target == F.conv1d:
                for i in [2, 3, 4]:
                    if len(prepack_args) > i and isinstance(prepack_args[i], int):
                        prepack_args[i] = (prepack_args[i],)
        else:
            raise ValueError(f"Lowering is not supported for op '{func_node.target}'")
        with model.graph.inserting_before(func_node):
            packed_weight = model.graph.create_node(
                "call_function", prepack_op, tuple(prepack_args), prepack_kwargs
            )

        # Step 3: Replace reference pattern with the corresponding quantized op
        func_node.target = q_relu_func if relu_node is not None else q_func
        if is_int8:
            func_node.args = (pattern_input, packed_weight, reduce_range_node)
        else:
            func_node.args = (pattern_input, packed_weight)

        if relu_node is not None:
            relu_node.replace_all_uses_with(func_node)

        # Step 4: Remove the relu node if it exists
        if relu_node is not None:
            model.graph.erase_node(relu_node)


def _lower_quantized_binary_op(model: GraphModule, qconfig_map: Dict[str, QConfigAny]):
    binary_ops_to_lower: List[Callable] = [
        operator.add,
        torch.add,
        operator.mul,
        torch.mul,
        torch.matmul,
    ]
    modules = dict(model.named_modules(remove_duplicate=False))
    for n in model.graph.nodes:
        # Step 0: Find nodes that match this pattern (dequantize - ref module - quantize)
        (q_node, relu_node, bop_node) = _match_static_pattern(
            n,
            modules,
            qconfig_map,
            binary_ops_to_lower,
            dequantize_node_arg_indices=[0, 1],
        )
        if q_node is None:
            continue
        assert bop_node is not None
        (_, scale_node, zero_point_node, _) = q_node.args

        # Step 1: Remove dequant nodes
        num_dq_nodes = 0
        for arg in bop_node.args:
            if not is_dequantize_node(arg):
                continue
            dq_node = arg
            assert isinstance(dq_node, Node)
            dn_input = dq_node.args[0]
            bop_node.replace_input_with(dq_node, dn_input)  # type: ignore[arg-type]
            num_dq_nodes += 1
        assert num_dq_nodes > 0

        # Step 2: Swap binary op to quantized binary op
        assert bop_node.target in QBIN_OP_MAPPING
        binop_to_qbinop = QBIN_OP_MAPPING if relu_node is None else QBIN_RELU_OP_MAPPING
        qbin_op = binop_to_qbinop[bop_node.target]
        # prepare the args for quantized binary op
        # (x, y)
        qop_node_args = list(bop_node.args)
        # (x, y, scale, zero_point)
        # add scale and zero_point arguments for Tensor - Tensor operation
        if num_dq_nodes == 2:
            qop_node_args.extend([scale_node, zero_point_node])
        # insert a call to quantized binary op and remove the original binary op
        with model.graph.inserting_after(q_node):
            qop_node = create_node_from_old_node_preserve_meta(
                model.graph,
                ("call_function", qbin_op, tuple(qop_node_args), {}),
                bop_node,
            )
            q_node.replace_all_uses_with(qop_node)

        # Step 3: Remove quantize node, binary op node, and relu node if any
        model.graph.erase_node(q_node)
        if relu_node is not None:
            model.graph.erase_node(relu_node)
        model.graph.erase_node(bop_node)


def special_pattern_replacement(model: GraphModule):
    modules = dict(model.named_modules(remove_duplicate=False))
    for n in model.graph.nodes:
        q_node = n
        is_quantize = q_node.target == torch.quantize_per_tensor
        is_to_fp16 = (
            q_node.op == "call_method"
            and q_node.target == "to"
            and len(q_node.args) == 2
            and q_node.args[1] == torch.float16
        )
        if not (is_quantize or is_to_fp16):
            continue
        ref_node = q_node.args[0]
        # get output scale/zero_point/dtype from the quantize node
        # ref_node, scale_node, zero_point_node, dtype = q_node.args
        # TODO: add safety checks that users for the ref_node and dq_node needs to be one
        is_call_function, is_call_method, is_call_module = is_fixed_qparams_node(
            ref_node, modules
        )
        if is_to_fp16 and (is_call_function or is_call_method or is_call_module):
            # TODO: add a warning or error out here? (bc-breaking if error out)
            # warnings.warn(
            #     "Only reference patterns are currently supported for {dtype} dtype with {op} op"
            #     "".format(dtype=dtypes, op=ref_node))
            continue

        is_call_function, is_call_method, is_call_module = is_default_node(
            ref_node, modules
        )
        if is_to_fp16 and (is_call_function or is_call_method or is_call_module):
            # TODO: add a warning or error out here? (bc-breaking if error out)
            continue

        # This check includes all supported ops
        is_call_function, is_call_method, is_call_module = is_special_pattern_node(
            ref_node, modules
        )
        if not (is_call_module or is_call_function or is_call_method):
            continue
        assert len(ref_node.args) > 0 or len(ref_node.kwargs) > 0
        dq_node_or_nodes = (
            ref_node.args[0]
            if len(ref_node.args) > 0
            else next(iter(ref_node.kwargs.values()))
        )
        assert isinstance(dq_node_or_nodes, (Node, tuple, list))
        is_dequantize = False
        if isinstance(dq_node_or_nodes, Node):
            is_dequantize = (
                dq_node_or_nodes.op == "call_method"
                and dq_node_or_nodes.target == "dequantize"
            )
        elif isinstance(dq_node_or_nodes, (tuple, list)):
            is_dequantize = all(
                x.op == "call_method" and x.target == "dequantize"
                for x in dq_node_or_nodes
            )

        if not is_dequantize:
            continue

        # TODO: enable we have patterns that needs to swap the modules
        if is_call_module:
            ref_module = modules[ref_node.target]
            if type(ref_module) in SPECIAL_PATTERN_LOWER_MODULE_MAP and is_quantize:
                qmodule_cls = SPECIAL_PATTERN_LOWER_MODULE_MAP.get(type(ref_module))
                scale_node = q_node.args[1]
                zero_point_node = q_node.args[2]
                output_scale = getattr(model, scale_node.target)
                output_zero_point = getattr(model, zero_point_node.target)

                qmodule = qmodule_cls.from_reference(  # type:ignore[union-attr]
                    ref_module, output_scale, output_zero_point
                )
                # replace reference module with quantized module
                parent_name, module_name = _parent_name(ref_node.target)
                setattr(modules[parent_name], module_name, qmodule)

        # reroute around dq node:
        dq_nodes: List[Node] = []
        if isinstance(dq_node_or_nodes, Node):
            dq_nodes = [dq_node_or_nodes]
        elif isinstance(dq_node_or_nodes, (tuple, list)):
            dq_nodes = list(dq_node_or_nodes)

        for dq_node in dq_nodes:
            dn_input = dq_node.args[0]
            ref_node.replace_input_with(dq_node, dn_input)

        # store q node args
        qnode_qparams = list(q_node.args)[1:]
        # replace uses of q node with input and remove q node
        q_node_input = q_node.args[0]
        q_node.replace_all_uses_with(q_node_input)
        model.graph.erase_node(q_node)

        is_call_function, is_call_method, is_call_module = is_default_node(
            ref_node, modules
        )
        if is_call_function:
            # pass scale/zer_point arguments from quantize_per_tensor to the default node operator
            # insert an op after the zero_point node so that the scale/zero_point
            # nodes are is available
            qop = get_quantized_operator(ref_node.target)
            args = list(ref_node.args)
            kwargs = dict(ref_node.kwargs)
            if qop in QOP_TO_ARG_NAMES_TO_SKIP:
                args_to_skip = QOP_TO_ARG_NAMES_TO_SKIP[qop]
                for arg in args_to_skip:
                    if arg in kwargs:
                        kwargs.pop(arg)
            kwargs["output_scale"] = qnode_qparams[0]
            kwargs["output_zero_point"] = qnode_qparams[1]
            with model.graph.inserting_after(qnode_qparams[1]):
                qop_node = create_node_from_old_node_preserve_meta(
                    model.graph, ("call_function", qop, tuple(args), kwargs), ref_node
                )
                ref_node.replace_all_uses_with(qop_node)
                model.graph.erase_node(ref_node)
        else:
            # remove scale/zero_point node for quantize node
            for n in qnode_qparams:
                if isinstance(n, Node):
                    model.graph.erase_node(n)

    return model


def _lower_getattr_tensor_metadta_op(model: GraphModule):
    """Modified the graph of the model inplace, to skip extra dequantize op before
    the general tensor shape ops when possible
    """
    for n in model.graph.nodes:
        if is_getattr_tensor_metadata_node(n):
            maybe_dq = n.args[0]
            if maybe_dq.op != "call_method" or maybe_dq.target != "dequantize":
                continue
            # skip the dequantize node
            args = list(n.args)
            args[0] = n.args[0].args[0]
            n.args = tuple(args)


def _lower_get_tensor_info_op(model: GraphModule):
    """Modified the graph of the model inplace, to skip extra dequantize op before
    the general tensor shape ops when possible
    """
    for n in model.graph.nodes:
        if not is_get_tensor_info_node(n):
            continue
        maybe_dq = n.args[0]
        if maybe_dq.op != "call_method" or maybe_dq.target != "dequantize":
            continue
        # skip the dequantize node
        args = list(n.args)
        args[0] = n.args[0].args[0]
        n.args = tuple(args)


def _lower_to_native_backend(
    model: GraphModule,
    qconfig_map: Dict[str, QConfigAny],
    node_name_to_scope: Dict[str, Tuple[str, type]],
) -> GraphModule:
    """Lower a quantized reference model (with reference quantized operator patterns)
    to the native backend in PyTorch (fbgemm/qnnpack), both backends shares the same
    operator signature so they can be lowered with the same function
    """
    _lower_static_weighted_ref_module(model, qconfig_map)
    _lower_static_weighted_ref_module_with_two_inputs(model, qconfig_map)
    _lower_dynamic_weighted_ref_module(model)
    _lower_weight_only_weighted_ref_module(model)
    _lower_static_weighted_ref_functional(model, qconfig_map)
    _lower_dynamic_weighted_ref_functional(model, qconfig_map)
    _lower_quantized_binary_op(model, qconfig_map)
    _lower_getattr_tensor_metadta_op(model)
    _lower_get_tensor_info_op(model)
    special_pattern_replacement(model)
    model.graph.eliminate_dead_code()
    model = fold_weight(model, node_name_to_scope)
    model.graph.eliminate_dead_code()
    model.recompile()
    model.graph.lint()
    return model
