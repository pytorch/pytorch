import itertools
import torch
from torch.fx import map_arg
from torch.fx.graph import Graph
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
import torch.nn.quantized as nnq
import torch.nn.quantized._reference as nnqr
from torch.nn.quantized.modules.utils import ReferenceableQuantizedModule
from . import subgraph_rewriter_FORKED_DO_NOT_USE
from .graph_module import QuantizedGraphModule
from .quantized_fusion_patterns_and_replacements import get_fbgemm_patterns_and_replacements
from .match_utils import is_match, MatchAllNode
from .utils import (
    collect_producer_nodes,
    get_linear_prepack_op_for_dtype,
    get_new_attr_name_with_prefix,
    graph_module_from_producer_nodes,
)
from ..utils import _parent_name, check_node
from typing import Dict, Tuple, Type, List, Any
from torch.fx import Node


def is_fixed_qparams_node(node, modules):
    func_list = [
        torch.nn.functional.hardsigmoid,
        torch.nn.functional.sigmoid,
        torch.sigmoid,
        torch.tanh,
    ]
    method_list = [
        'hardsigmoid',
        'hardsigmoid_',
        'sigmoid',
        'sigmoid_',
        'tanh',
        'tanh_',
    ]
    module_type_list = [
        torch.nn.Hardsigmoid,
        torch.nn.Sigmoid,
        torch.nn.Tanh,
    ]
    is_call_function = node.op == "call_function" and node.target in func_list
    is_call_method = node.op == "call_method" and node.target in method_list
    is_call_module = node.op == "call_module" and type(modules[str(node.target)]) in module_type_list
    return is_call_function, is_call_method, is_call_module

# Mapping from reference module class to the replacement quantized module class for lowering
# TODO: fix typing, the key is reference module
LOWER_MODULE_MAP: Dict[Type[torch.nn.Module], Type[ReferenceableQuantizedModule]] = {
    nnqr.Linear: nnq.Linear,
    nnqr.Conv1d: nnq.Conv1d,
    nnqr.Conv2d: nnq.Conv2d,
    nnqr.Conv3d: nnq.Conv3d,
}

# TODO: merge with LOWER_MODULE_MAP after we merge
# _lower_weighted_ref_module and special_pattern_replacement
SPECIAL_PATTERN_LOWER_MODULE_MAP = {
    nn.BatchNorm2d: nnq.BatchNorm2d,
    nn.BatchNorm3d: nnq.BatchNorm3d,
}

# Mapping from fused module class to a 2-tuple of:
#   1) The inner reference module class
#   2) The replacement quantized module class for lowering
LOWER_FUSED_MODULE_MAP: Dict[Type[nn.Module], Tuple[Type[nn.Module], Type[ReferenceableQuantizedModule]]] = {
    nni.LinearReLU: (nnqr.Linear, nniq.LinearReLU)
}

# Mapping from a functional to lower to a 2-tuple of
#   1) The quantized version of the op
#   2) The quantized version of the op fused with relu, if it exists, else None
LOWER_FUNCTIONAL_MAP = {
    F.linear: (torch.ops.quantized.linear, torch.ops.quantized.linear_relu),
}

WEIGHT_PREPACK_OPS = {
    torch._ops.ops.quantized.linear_prepack,
    torch._ops.ops.quantized.linear_prepack_fp16,
    torch._ops.ops.quantized.conv1d_prepack,
    torch._ops.ops.quantized.conv2d_prepack,
    torch._ops.ops.quantized.conv3d_prepack,
}

def fold_weight(
        quantized: QuantizedGraphModule,
        node_name_to_scope: Dict[str, Tuple[str, type]]) -> QuantizedGraphModule:
    """
    Trace back from the weight node util we hit getattr, reconstruct the
    graph module with the traced nodes and run the graph module to pack the
    weight. then replace the original chain of ops with the packed weight.
    """
    packed_weights = dict()
    # map from folded node name to the prepacked weight name
    folded_nodes = dict()
    # get packed weights
    for node in quantized.graph.nodes:
        if node.op == 'call_function' and node.target in WEIGHT_PREPACK_OPS:
            nodes_to_fold = collect_producer_nodes(node)
            if nodes_to_fold is not None:
                for node_to_fold in nodes_to_fold:
                    folded_nodes[node_to_fold.name] = node

                prepacking_module = graph_module_from_producer_nodes(
                    quantized, nodes_to_fold)
                packed_weight = prepacking_module()
                packed_weights[node.name] = packed_weight

    # remove folded nodes and replace the prepacking node with getattr
    folded_graph = Graph()
    env: Dict[Any, Any] = {}

    def load_arg(a):
        return map_arg(a, lambda node: env[node.name])
    quantized_root = quantized
    quantized_graph = quantized.graph

    for node in quantized_graph.nodes:
        prepack_node = folded_nodes.get(node.name, None)
        if prepack_node is node:
            packed_weight = packed_weights[node.name]
            # add a prepacked attribute to root
            op_node = list(prepack_node.users)[0]
            module_path, _ = node_name_to_scope[op_node.name]
            get_new_packed_weight_name = \
                get_new_attr_name_with_prefix(module_path + '_packed_weight_')
            packed_weight_name = get_new_packed_weight_name(quantized_root)
            setattr(quantized_root, packed_weight_name, packed_weight)
            # replace prepack node with a getattr node
            env[node.name] = folded_graph.create_node(
                'get_attr', packed_weight_name, (), {})
        elif prepack_node is not None:
            # remove the foled node
            continue
        else:
            # copy other nodes
            env[node.name] = folded_graph.node_copy(node, load_arg)
    quantized = QuantizedGraphModule(quantized_root, folded_graph, quantized_root.preserved_attr_names)
    return quantized

def _lower_weighted_ref_module(model: QuantizedGraphModule) -> QuantizedGraphModule:
    """
    Traverse the graph and find dequantize - ref module - quantize patterns
    and replace them with the quantized version of the ref module.
    """
    for ref_class in list(LOWER_MODULE_MAP.keys()) + list(LOWER_FUSED_MODULE_MAP.keys()):
        pattern = (torch.quantize_per_tensor,
                   (ref_class, "dequantize"),
                   MatchAllNode, MatchAllNode, MatchAllNode)
        modules = dict(model.named_modules(remove_duplicate=False))
        nodes = list(model.graph.nodes)
        # TODO: maybe orgnize this better (e.g. break down to more functions)
        # to make this function more readable
        for n in model.graph.nodes:
            if not is_match(modules, n, pattern):
                continue
            q_node = n
            ref_node = q_node.args[0]
            dq_node = ref_node.args[0]
            # get output scale/zero_point/dtype from the quantize node
            scale_node = q_node.args[1]
            zero_point_node = q_node.args[2]
            dtype = q_node.args[3]

            # this can be removed if we add support for "get_attr" in is_match
            if scale_node.op != "get_attr" or zero_point_node.op != "get_attr":
                print("Find the pattern but scale_node and zero_point node are not `get_attr`,"
                      f"got: {scale_node.format_node} {zero_point_node.format_node()}")
                continue

            # this can be removed if we add support for constants in is_match
            if dtype != torch.quint8:
                print(f"Only qint8 output for quantized op is supported, got: {dtype}")
                continue

            # change this pattern to use the corresponding quantized module
            ref_module = modules[ref_node.target]
            output_scale = getattr(model, scale_node.target)
            output_zero_point = getattr(model, zero_point_node.target)
            # For fused modules, we also check whether the inner module is a reference module
            # If so, we replace the entire fused module with the corresponding quantized module
            if ref_class in LOWER_FUSED_MODULE_MAP:
                inner_ref_class, q_class = LOWER_FUSED_MODULE_MAP[ref_class]
                if type(ref_module[0]) != inner_ref_class:
                    continue
            else:
                q_class = LOWER_MODULE_MAP[type(ref_module)]
            assert issubclass(q_class, ReferenceableQuantizedModule)  # suppress mypy warnings
            q_module = q_class.from_reference(ref_module, output_scale, output_zero_point)

            # replace reference module with quantized module
            parent_name, module_name = _parent_name(ref_node.target)
            setattr(modules[parent_name], module_name, q_module)
            # remove dq node:
            dq_node_input = dq_node.args[0]

            dq_node.replace_all_uses_with(dq_node_input)
            model.graph.erase_node(dq_node)

            # remove q node and args:
            q_node.replace_all_uses_with(ref_node)
            model.graph.erase_node(q_node)
            model.graph.erase_node(scale_node)
            model.graph.erase_node(zero_point_node)
    return model

def _lower_weighted_ref_functional(model: QuantizedGraphModule) -> QuantizedGraphModule:
    """
    Traverse the graph and replace functional reference patterns with their quantized versions.
    """
    for ref_func, (q_func, q_relu_func) in LOWER_FUNCTIONAL_MAP.items():
        configurations = itertools.product(
            (False, True),  # is_relu: whether ref_func is wrapped in a relu op
            (False, True),  # has_bias: whether bias is passed as an extra argument to ref_func
        )
        for is_relu, has_bias in configurations:
            if is_relu and q_relu_func is None:
                continue

            # Set up match pattern: (dequantize - [relu_op - ] func_op - quantize)
            # Func args: (dequantized inputs, dequantized weights[, bias])
            # Quantize args: (func, scale, zp, dtype)
            func_pattern: Tuple[Any, ...] = ()
            if has_bias:
                func_pattern = (ref_func, "dequantize", "dequantize", MatchAllNode)
            else:
                func_pattern = (ref_func, "dequantize", "dequantize")
            if is_relu:
                func_pattern = (F.relu, func_pattern)
            pattern = (torch.quantize_per_tensor, func_pattern, MatchAllNode, MatchAllNode, MatchAllNode)

            # Iterate through nodes in the graph to find a match
            # If there is a match, replace the above pattern with the corresponding quantized op
            modules = dict(model.named_modules(remove_duplicate=False))
            nodes = list(model.graph.nodes)
            for n in model.graph.nodes:
                if not is_match(modules, n, pattern):
                    continue
                q_node = n
                (func_node, output_scale_node, output_zp_node, dtype) = q_node.args
                if is_relu:
                    relu_node = func_node
                    func_node = relu_node.args[0]
                else:
                    relu_node = None
                input_dq_node = func_node.args[0]
                weight_dq_node = func_node.args[1]

                # Step 1: Replace quantized weights with packed weights, which will be folded later
                quantized_weight = weight_dq_node.args[0]
                weight_dtype = quantized_weight.args[-1]
                if has_bias:
                    bias = func_node.args[2]
                else:
                    bias = func_node.kwargs.get("bias", None)
                prepack_args = (quantized_weight, bias)
                if ref_func == F.linear:
                    prepack_op = get_linear_prepack_op_for_dtype(weight_dtype)
                else:
                    raise ValueError("Lowering for functional currently only supports linear op")
                insert_prepack_after = bias if has_bias else quantized_weight
                with model.graph.inserting_after(insert_prepack_after):
                    packed_weight = model.graph.create_node("call_function", prepack_op, prepack_args, {})

                # Step 2: Replace reference pattern with the corresponding quantized op
                func_node.args = (input_dq_node.args[0], packed_weight, output_scale_node, output_zp_node)
                func_node.target = q_relu_func if is_relu else q_func
                q_node.replace_all_uses_with(func_node)
                output_zp_node.append(func_node)

                # Clean up: Remove dequantize and quantize nodes and the old func node
                for dqn in [input_dq_node, weight_dq_node]:
                    dqn_input = dqn.args[0]
                    dqn.replace_all_uses_with(dqn_input)
                    model.graph.erase_node(dqn)
                model.graph.erase_node(q_node)
                if is_relu:
                    model.graph.erase_node(relu_node)
    return model

def special_pattern_replacement(model: QuantizedGraphModule) -> QuantizedGraphModule:
    modules = dict(model.named_modules(remove_duplicate=False))
    nodes = list(model.graph.nodes)
    for n in model.graph.nodes:
        q_node = n
        is_quantize = q_node.target == torch.quantize_per_tensor
        is_to_fp16 = q_node.op == "call_method" and q_node.target == "to" and q_node.args[1] == torch.float16
        if not (is_quantize or is_to_fp16):
            continue
        ref_node = q_node.args[0]
        # get output scale/zero_point/dtype from the quantize node
        # ref_node, scale_node, zero_point_node, dtype = q_node.args
        # TODO: add safety checks that users for the ref_node and dq_node needs to be one

        is_call_function, is_call_method, is_call_module = is_fixed_qparams_node(ref_node, modules)
        if is_to_fp16 and (is_call_function or is_call_method or is_call_module):
            # TODO: add a warning or error out here? (bc-breaking if error out)
            continue

        is_call_function, is_call_method, is_call_module = check_node(ref_node, modules)
        if not (is_call_module or is_call_function or is_call_method):
            continue
        dq_node_or_nodes = ref_node.args[0]
        assert isinstance(dq_node_or_nodes, Node) or isinstance(dq_node_or_nodes, (tuple, list))
        is_dequantize = False
        if isinstance(dq_node_or_nodes, Node):
            is_dequantize = dq_node_or_nodes.op == 'call_method' and \
                dq_node_or_nodes.target == 'dequantize'
        elif isinstance(dq_node_or_nodes, (tuple, list)):
            is_dequantize = all(
                x.op == 'call_method' and x.target == 'dequantize'
                for x in dq_node_or_nodes)

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

                qmodule = qmodule_cls.from_reference(ref_module, output_scale, output_zero_point)  # type:ignore[union-attr]
                # replace reference module with quantized module
                parent_name, module_name = _parent_name(ref_node.target)
                setattr(modules[parent_name], module_name, qmodule)

        # remove dq node:
        dq_nodes: List[Node] = []
        if isinstance(dq_node_or_nodes, Node):
            dq_nodes = [dq_node_or_nodes]
        elif isinstance(dq_node_or_nodes, (tuple, list)):
            dq_nodes = list(dq_node_or_nodes)

        for dq_node in dq_nodes:
            dn_input = dq_node.args[0]
            dq_node.replace_all_uses_with(dn_input)
            model.graph.erase_node(dq_node)

        # store q node args
        q_node_args = list(q_node.args)[1:]

        # replace uses of q node with input and remove q node
        q_node_input = q_node.args[0]
        q_node.replace_all_uses_with(q_node_input)
        model.graph.erase_node(q_node)

        # remove q node args
        for n in q_node_args:
            if isinstance(n, Node):
                model.graph.erase_node(n)
    return model

def _lower_to_native_backend(
        model: QuantizedGraphModule,
        node_name_to_scope: Dict[str, Tuple[str, type]]) -> QuantizedGraphModule:
    """ Lower a quantized reference model (with reference quantized operator patterns)
    to the native backend in PyTorch (fbgemm/qnnpack), both backends shares the same
    operator signature so they can be lowered with the same function
    """
    model = _lower_weighted_ref_module(model)
    model = _lower_weighted_ref_functional(model)
    for pattern, replacement in get_fbgemm_patterns_and_replacements():
        subgraph_rewriter_FORKED_DO_NOT_USE.replace_pattern(model, pattern, replacement)
    special_pattern_replacement(model)
    model = fold_weight(model, node_name_to_scope)
    model.recompile()
    model.graph.lint()
    return model
