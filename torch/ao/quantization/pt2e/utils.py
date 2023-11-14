import operator
import types

import torch
from torch._export import capture_pre_autograd_graph
from torch.fx import (
    GraphModule,
    Node,
)
from torch.nn.utils.fusion import fuse_conv_bn_weights
from typing import Any, Callable, Dict, Optional, Tuple, List, Union
from torch.utils._pytree import LeafSpec

# Makes sure that quantized_decomposed ops are registered
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401

from torch.ao.quantization.quantizer import QuantizationAnnotation


__all__ = [
    "fold_bn_weights_into_conv_node",
    "get_aten_graph_module",
    "remove_tensor_overload_for_qdq_ops",
]

_QUANTIZE_OPS = [
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.quantize_per_channel.default,
]


_DEQUANTIZE_OPS = [
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.dequantize_per_channel.default,
]


def _is_connected(source: torch.fx.Node, dest: torch.fx.Node) -> bool:
    """
    Assuming dest is one of the ops inserted by quant workflow, this function
    finds if source and dest are connected. Assumption is that only quant workflow
    inserted ops exist between source and dest
    """
    quant_workflow_ops = _QUANTIZE_OPS + _DEQUANTIZE_OPS
    quant_workflow_ops.append(torch.ops.quantized_decomposed.choose_qparams.tensor)
    while dest.target in quant_workflow_ops:
        if not isinstance(dest.args[0], torch.fx.Node):
            raise ValueError(f"expected arg[0] of quant workflow ops to be a node but found {dest.args[0]}")
        dest = dest.args[0]
    return (dest == source)


def _find_q_dq_node_for_user(
    produer: torch.fx.Node, user: torch.fx.Node
) -> Tuple[Any, Any]:
    """
    Find q, dq pair corresponding to [producer -> q -> dq -> user]
    Utils works by finding dq arg of user and ensuring it is connected to
    producer
    """
    dq_node = None
    for n in user.args:
        if isinstance(n, torch.fx.Node) and n.op == "call_function" and n.target in _DEQUANTIZE_OPS:
            if _is_connected(produer, n):
                dq_node = n
                break
    if dq_node is None:
        for n in user.kwargs:
            if isinstance(n, torch.fx.Node) and n.op == "call_function" and n.target in _DEQUANTIZE_OPS:
                if _is_connected(produer, n):
                    dq_node = n
                    break
    if dq_node is None:
        return (None, None)

    q_node = None
    if dq_node.args[0].op == "call_function" and dq_node.args[0].target in _QUANTIZE_OPS:
        q_node = dq_node.args[0]
    return (q_node, dq_node)



def _is_sym_size_node(node: Node):
    return (
        node.op == "call_function"
        and node.target == torch.ops.aten.sym_size.default
        or node.target == torch.ops.aten.sym_numel.default
        or node.target == torch.ops.aten.sym_numel
        or node.target == torch.ops.aten.sym_size
    )


def _filter_sym_size_users(node: torch.fx.Node) -> List[torch.fx.Node]:
    node_users = list(filter((lambda x: (_is_sym_size_node(x) is False)), node.users))
    return node_users


def _is_valid_annotation(annotation: QuantizationAnnotation) -> bool:
    if annotation is None:
        return False
    input_qspec_map = annotation.input_qspec_map
    output_qspec = annotation.output_qspec
    if len(input_qspec_map) == 0 and output_qspec is None:
        return False
    return True


def _get_tensor_constant_from_node(node, m):
    if node is None:
        return None
    assert node.op == "get_attr"
    return getattr(m, node.target)

def _get_all_arguments(orig_args, orig_kwargs, args_schema):
    all_args = []
    for i, schema in enumerate(args_schema):
        if schema.name in orig_kwargs:
            all_args.append(orig_kwargs[schema.name])
        elif not schema.kwarg_only and i < len(orig_args):
            all_args.append(orig_args[i])
        else:
            all_args.append(schema.default_value)
    return all_args

def _is_supported_batch_norm_for_training(node: Node):
    """
    Return True if the given node refers to an aten batch norm op QAT supports.
    """
    supported_ops = [
        torch.ops.aten._native_batch_norm_legit.default,
        # Note: we won't need this op anymore after batch norm consolidation
        # For now, we need to continue to support it because it gives better
        # training numerics than `_native_batch_norm_legit`
        torch.ops.aten.cudnn_batch_norm.default,
        torch.ops.aten.miopen_batch_norm.default,
    ]
    return node.target in supported_ops

def fold_bn_weights_into_conv_node(
    conv_node: Node,
    conv_weight_node: Node,
    conv_bias_node: Optional[Node],
    bn_node: Node,
    m: GraphModule
) -> None:
    # conv2d args: input, weight, bias, stride, padding, dilation, ...
    # Note: this should also work for conv1d, conv3d and transposed conv1-3d as well with
    # easy tweaks
    conv_w = _get_tensor_constant_from_node(conv_weight_node, m)
    conv_b = _get_tensor_constant_from_node(conv_bias_node, m)
    transpose = not (conv_node.target == torch.ops.aten.conv2d.default)

    # eval bn args: input, weight, bias, running mean, running var, momentum, eps
    # train bn args: input, weight, bias, running mean, running var, training, momentum, eps
    bn_args_schema = bn_node.target._schema.arguments  # type: ignore[union-attr]
    bn_args = _get_all_arguments(bn_node.args, bn_node.kwargs, bn_args_schema)
    bn_w = _get_tensor_constant_from_node(bn_args[1], m)
    bn_b = _get_tensor_constant_from_node(bn_args[2], m)
    bn_rm = _get_tensor_constant_from_node(bn_args[3], m)
    bn_rv = _get_tensor_constant_from_node(bn_args[4], m)
    if bn_node.target == torch.ops.aten._native_batch_norm_legit_no_training.default:
        eps_arg_index = 6
    elif _is_supported_batch_norm_for_training(bn_node):
        eps_arg_index = 7
    else:
        raise ValueError("BN node target is unexpected ", bn_node.target)
    bn_eps = bn_args[eps_arg_index]

    fused_weight, fused_bias = fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b, transpose=transpose)

    # update the weight and bias for conv
    conv_args = list(conv_node.args)
    # filling in the default bias argument
    if len(conv_args) == 2:
        conv_args.append(None)

    # calling data since the fused_weight and fused_bias are nn.Parameter
    weight_attr_name = conv_weight_node.target
    assert isinstance(weight_attr_name, str)
    setattr(m, weight_attr_name, fused_weight)
    if conv_bias_node is not None:
        bias_attr_name = conv_bias_node.target
        setattr(m, bias_attr_name, fused_bias)  # type: ignore[arg-type]
    else:
        bias_attr_name = weight_attr_name + "_bias"
        setattr(m, bias_attr_name, fused_bias)  # type: ignore[arg-type]
        with m.graph.inserting_before(conv_node):
            get_bias_node = m.graph.get_attr(bias_attr_name)
        # NOTE: here we assume the bias of conv is not quantized!
        conv_args[2] = get_bias_node
    conv_node.args = tuple(conv_args)

    # native_batch_norm has 3 outputs, we expect getitem calls on the output
    # and we want to replace the uses of getitem 0 with the output of conv
    #
    # Before:
    # conv -> bn - (first output) -> users1
    #          \ - (second output) -> users2
    #          \ - (third output) -> users3
    # After:
    # conv -> (first output) -> users1
    #       bn -
    #          \ - (second output) -> users2
    #          \ - (third output) -> users3
    # if users2 and users3 are empty then bn will be removed through dead code elimination

    for user in bn_node.users:
        if user.op != "call_function" or user.target != operator.getitem or user.args[1] != 0:
            continue
        user.replace_all_uses_with(conv_node)

# fuse conv bn weights, inplace modification of the graph_module and graph
def _fuse_conv_bn_(m: GraphModule) -> None:
    for n in m.graph.nodes:
        if n.op != "call_function" or n.target != torch.ops.aten._native_batch_norm_legit_no_training.default:
            continue
        bn_node = n
        n = bn_node.args[0]
        if n.op != "call_function" or n.target != torch.ops.aten.conv2d.default:
            continue
        conv_node = n
        conv_weight_node = conv_node.args[1]
        conv_bias_node = conv_node.args[2] if len(conv_node.args) > 2 else None
        fold_bn_weights_into_conv_node(conv_node, conv_weight_node, conv_bias_node, bn_node, m)

    m.graph.eliminate_dead_code()
    m.recompile()

def _get_node_name_to_scope(model: GraphModule) -> Dict[str, Tuple[str, type]]:
    # TODO: move this information to fx node itself
    node_name_to_scope: Dict[str, Tuple[str, type]] = {}
    for n in model.graph.nodes:
        nn_module_stack = n.meta.get("nn_module_stack", None)
        current_scope = ("", type(None))
        if nn_module_stack:
            bt = list(nn_module_stack.values())[-1]
            current_scope = (bt[0].split(".")[-1], bt[1])
        node_name_to_scope[n.name] = current_scope
    return node_name_to_scope

def get_aten_graph_module(
    pattern: Callable,
    example_inputs: Tuple[Any, ...],
    is_cuda: bool = False,
    **kwargs,
) -> GraphModule:
    """
    Convert the pattern to an FX graph with decomposed aten ops.
    """
    if is_cuda:
        example_inputs = tuple([x.cuda() if isinstance(x, torch.Tensor) else x for x in example_inputs])
    aten_pattern = capture_pre_autograd_graph(
        pattern,
        example_inputs,
        kwargs,
    )
    aten_pattern.graph.eliminate_dead_code()
    aten_pattern.recompile()
    return aten_pattern

def remove_tensor_overload_for_qdq_ops(match_pattern: GraphModule) -> None:
    """ Remove .tensor overload for quantize/dequantize ops so that we can
    use the match_pattern that we get from torchdynamo export to match the output of convert_pt2e
    """
    _MAP = {
        torch.ops.quantized_decomposed.quantize_per_tensor.default: torch.ops.quantized_decomposed.quantize_per_tensor,
        torch.ops.quantized_decomposed.dequantize_per_tensor.default: torch.ops.quantized_decomposed.dequantize_per_tensor,
        torch.ops.quantized_decomposed.quantize_per_tensor.tensor: torch.ops.quantized_decomposed.quantize_per_tensor,
        torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: torch.ops.quantized_decomposed.dequantize_per_tensor,
        torch.ops.quantized_decomposed.quantize_per_tensor.tensor2: torch.ops.quantized_decomposed.quantize_per_tensor,
        torch.ops.quantized_decomposed.dequantize_per_tensor.tensor2: torch.ops.quantized_decomposed.dequantize_per_tensor,
        torch.ops.quantized_decomposed.quantize_per_channel.default: torch.ops.quantized_decomposed.quantize_per_channel,
        torch.ops.quantized_decomposed.dequantize_per_channel.default: torch.ops.quantized_decomposed.dequantize_per_channel,
        torch.ops.aten.clamp.Tensor: torch.ops.aten.clamp,
    }
    for n in match_pattern.graph.nodes:
        if n.op != "call_function":
            continue
        if n.target in _MAP:
            n.target = _MAP[n.target]

def _is_literal(arg):
    if isinstance(arg, (int, float)):
        return True
    if isinstance(arg, (tuple, list)):
        return all(map(_is_literal, arg))
    return False

def _replace_literals_with_new_placeholders(
    gm: torch.fx.GraphModule,
    merge_dup: bool = False,
    exclude_literals: Optional[List[Any]] = None
):
    """Replace the literals in the graph with placeholder nodes that's created on the fly while we
    traverse the graph, so that the literal arguments in the graph can be matched and replaced

    To use this, the pattern and replacement graph should have the exact same number of literal args
    and they should be used in the exact same order in the pattern and replacement graph.

    If the literal arguments are not used in the same order in pattern and replacement graph, please
    use `_replace_literals_with_existing_placeholders` instead

    Args:
        `gm`: input GraphModule that we'll transform
        `merge_dup`: boolean flag to indicate that if the same literal appears multiple times in
         the graph, whether they should correspond to the same placeholder or not
        `exclude_literals`: a list of literals that will not be replaced with placeholders

    Example:

    # 1. Original Graph
    def pattern(self, x):
        return x + 3

    def replacement(self, x):
        return x - 3

    example_inputs = (torch.randn(1, 3, 3, 3),)
    pattern_gm = get_aten_graph_module(pattern, example_inputs)
    replacement_gm = get_aten_graph_module(pattern, example_inptus)

    # 2. Before calling replace literals we'll see the following graph:
    def pattern(self, x):
        return x + 3

    def replacement(self, x):
        return x - 3

    pattern_gm = _replace_literals_with_new_placeholders(pattern_gm)
    replacement_gm = _replace_literals_with_new_placeholders(replacement_gm)

    # 3. After replacing literals with new placeholder nodes

    def pattern(self, x, new_ph):
        return x + new_ph

    def pattern(self, x, new_ph):
        return x - new_ph

    """
    last_ph = None
    cnt = 0
    literal_to_ph: Dict[Union[float, bool, int, torch.dtype], Node] = {}
    if exclude_literals is None:
        exclude_literals = []

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            last_ph = node
            cnt += 1
            continue
        with gm.graph.inserting_after(last_ph):
            new_args = []
            for arg in node.args:
                if _is_literal(arg) and arg not in exclude_literals:
                    if merge_dup and arg in literal_to_ph:
                        new_args.append(literal_to_ph[arg])
                    else:
                        ph_node = gm.graph.placeholder("arg" + str(cnt))
                        new_args.append(ph_node)
                        gm._in_spec.children_specs[0].children_specs.append(LeafSpec())
                        cnt += 1
                        if merge_dup:
                            literal_to_ph[arg] = ph_node
                else:
                    new_args.append(arg)
            new_args = tuple(new_args)

        node.args = new_args
    return gm


def _replace_literals_with_existing_placeholders(
    gm: torch.fx.GraphModule,
    exclude_literals: Optional[List[Any]] = None,
    literal_to_ph_idx: Optional[Dict[Union[float, int, bool, torch.dtype], int]] = None
):
    """Replace the literals in the graph with **existing** placeholder nodes, so that the literal arguments
    in the graph can be matched and replaced

    To use this, all literal args in the graph should be unique and each of them should correspond
    to exactly one placeholder node

    # 1. Original Graph
    def pattern(self, x_i8, scale, zero_point, quant_min, quant_max):
        return torch.dequantize_per_tensor(x_i8, scale, zero_point, quant_min, quant_max)

    def replacement(x_i8, scale, zero_point, quant_min, quant_max):
        x_i8 = torch.clamp(x_i8, quant_min, quant_max)
        return ((x_i8.to(torch.float32) - zero_point) * scale).to(dtype=torch.float32)

    example_inputs = (
        torch.randn(1, 3, 3, 3),
        1.0,
        0,
        -128,
        127,
    )
    pattern_gm = get_aten_graph_module(pattern, example_inputs)
    replacement_gm = get_aten_graph_module(pattern, example_inptus)

    # 2. Before calling replace literals we'll see the following graph:
    def pattern(self, x_i8, scale, zero_point, quant_min, quant_max):
        # scale/zero_point/quant_min/quant_max are burnt in since they are scalar values
        return torch.dequantize_per_tensor(x_i8, 1.0, 0, -128, 127)

    def replacement(x_i8, scale, zero_point, quant_min, quant_max):
        # scale/zero_point/quant_min/quant_max are burnt in since they are scalar values
        x_i8 = torch.clamp(x_i8, -128, 127)
        return ((x_i8.to(torch.float32) - 0) * 1.0).to(dtype=torch.float32)

    # Note that literal args appear in different order in pattern and replacement graph, so
    # we can't use _replace_literals_with_new_placeholders

    literal_to_ph_idx = {1.0: 1, 0: 2, -128: 3, 127: 4}
    pattern_gm = _replace_literals_with_existing_placeholders(pattern_gm, literal_to_ph_idx)
    replacement_gm = _replace_literals_with_existing_placeholders(replacement_gm, literal_to_ph_idx)

    # 3. After replacing literals with existing placeholder nodes

    def pattern(self, x_i8, scale, zero_point, quant_min, quant_max):
        # scale/zero_point/quant_min/quant_max are burnt in since they are scalar values
        return torch.dequantize_per_tensor(x_i8, scale, zero_point, quant_min, quant_max)

    def replacement(x_i8, scale, zero_point, quant_min, quant_max):
        # scale/zero_point/quant_min/quant_max are burnt in since they are scalar values
        x_i8 = torch.clamp(x_i8, quant_min, quant_max)
        return ((x_i8.to(torch.float32) - zero_point) * scale).to(dtype=torch.float32)
    """
    if exclude_literals is None:
        exclude_literals = []

    if literal_to_ph_idx is None:
        literal_to_ph_idx = {}

    phs = [node for node in gm.graph.nodes if node.op == "placeholder"]

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        new_args = []
        for arg in node.args:
            if _is_literal(arg) and arg not in exclude_literals and arg in literal_to_ph_idx:
                ph_idx = literal_to_ph_idx[arg]
                ph_node = phs[ph_idx]
                new_args.append(ph_node)
            else:
                new_args.append(arg)
        new_args = tuple(new_args)
        node.args = new_args
    return gm

# TODO: Handle this in export itself and don't wrap the model in another GraphModule
# in prepare and convert
def _disallow_eval_train(model: GraphModule):
    """
    Disallow calling `model.train()` or `model.eval()` on the given GraphModule.
    This is useful for exported models, where these methods don't actually behave as expected.
    """
    def _train(self, mode: bool = True):
        raise NotImplementedError("Calling train() is not supported yet.")

    def _eval(self, mode: bool = True):
        raise NotImplementedError("Calling eval() is not supported yet.")

    model.train = types.MethodType(_train, model)  # type: ignore[method-assign]
    model.eval = types.MethodType(_eval, model)  # type: ignore[method-assign]
    return model
