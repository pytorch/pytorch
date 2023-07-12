import dataclasses
import itertools
import operator
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.subgraph_rewriter import replace_pattern_with_filters
import torch.nn.functional as F
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from .quantizer import (
    DerivedQuantizationSpec,
    EdgeOrNode,
    SharedQuantizationSpec,
    QuantizationSpecBase,
)
from .utils import (
    _fold_bn_weights_into_conv_node,
    _get_aten_graph_module,
)

# Example inputs for `_conv2d_bn_pattern`, `_qat_conv2d_bn_pattern`, and `_qat_conv2d_bn_pattern_no_bias`
_conv2d_bn_pattern_example_inputs = (
    torch.randn(1, 1, 3, 3),  # x
    torch.randn(1, 1, 1, 1),  # conv_weight
    torch.randn(1),           # conv_bias
    torch.randn(1),           # bn_weight
    torch.randn(1),           # bn_bias
    torch.randn(1),           # bn_running_mean
    torch.randn(1),           # bn_running_var
)

# Example inputs for both `_quantized_qat_conv2d_bn_pattern` and `_folded_quantized_qat_conv2d_bn_pattern`
_quantized_conv2d_bn_pattern_example_inputs = (
    torch.randn(1, 1, 3, 3),  # x
    torch.randn(1, 1, 1, 1),  # conv_weight
    torch.randn(1),           # bn_weight
    torch.randn(1),           # bn_bias
    torch.randn(1),           # bn_running_mean
    torch.randn(1),           # bn_running_var
)

def _get_quantized_conv2d_bn_pattern_example_inputs_kwargs(
    is_per_channel: bool,
    has_bias: bool,
) -> Dict[str, Any]:
    """
    Optional example inputs for both `_quantized_qat_conv2d_bn_pattern`
    and `_folded_quantized_qat_conv2d_bn_pattern`, expressed as kwargs.

    Note that weight_scale and weight_zero_point are only used when
    `is_per_channel` is True. This is because for per tensor quantization,
    scale and zero point are hard coded into quantize/dequantize ops
    in the pattern.
    """
    kwargs = {}
    if is_per_channel:
        kwargs["weight_scale"] = torch.tensor([1], dtype=torch.float)
        kwargs["weight_zero_point"] = torch.tensor([0], dtype=torch.int)
    if has_bias:
        kwargs["conv_bias"] = torch.randn(1)
    return kwargs

def _conv2d_bn_pattern(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
) -> torch.Tensor:
    x = F.conv2d(x, conv_weight, conv_bias)
    x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True)
    return x

# TODO: merge this with the `no_conv_bias` case
def _qat_conv2d_bn_pattern(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
) -> torch.Tensor:
    """
    Approximated method to fuse conv and bn. It requires only one forward pass.
    conv_orig = conv / scale_factor where scale_factor = bn.weight / running_std.
    This is based on `nniqat.ConvBn2d._forward_approximate`.
    """
    # TODO: allow setting eps
    bn_eps = 1e-5
    running_std = torch.sqrt(bn_running_var + bn_eps)
    scale_factor = bn_weight / running_std
    weight_shape = [1] * len(conv_weight.shape)
    weight_shape[0] = -1
    bias_shape = [1] * len(conv_weight.shape)
    bias_shape[1] = -1
    scaled_weight = conv_weight * scale_factor.reshape(weight_shape)
    zero_bias = torch.zeros_like(conv_bias, dtype=x.dtype)
    x = F.conv2d(x, scaled_weight, zero_bias)
    x = x / scale_factor.reshape(bias_shape)
    x = x + conv_bias.reshape(bias_shape)
    x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True, eps=bn_eps)
    return x

def _qat_conv2d_bn_pattern_no_conv_bias(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    # Not used, only for matching convenience
    conv_bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
) -> torch.Tensor:
    """
    Same as `_qat_conv2d_bn_pattern`, but handles the case with no conv bias.
    """
    # TODO: allow setting eps
    bn_eps = 1e-5
    running_std = torch.sqrt(bn_running_var + bn_eps)
    scale_factor = bn_weight / running_std
    weight_shape = [1] * len(conv_weight.shape)
    weight_shape[0] = -1
    bias_shape = [1] * len(conv_weight.shape)
    bias_shape[1] = -1
    scaled_weight = conv_weight * scale_factor.reshape(weight_shape)
    x = F.conv2d(x, scaled_weight, None)
    x = x / scale_factor.reshape(bias_shape)
    x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True, eps=bn_eps)
    return x

def _get_input_output_quantized_filter():
    def _input_output_quantized_filter(
        match: "InternalMatch",  # type: ignore[name-defined]
        original_graph: Graph,
        pattern_graph: Graph,
    ) -> bool:
        """
        Make sure that the matched pattern's input is coming from dq node
        and the output is from q node. This is used to filter out the nodes for
        conv-bn pattern.
        We need to replace qat's conv-bn pattern with just conv-bn nodes.
        QAT's conv-bn pattern has q-dq node inserted after convert step.
        In order to replace QAT pattern, see _get_quantized_qat_conv2d_bn_pattern,
        with a simpler pattern, see _get_folded_quantized_qat_conv2d_bn_pattern,
        we need to port the quantization parameters from q/dq nodes of weight.
        This porting becomes easier if there is only one q/dq node because we dont have to
        reason about about finding the right q/dq node from original graph.
        In order to facilitate that matched pattern and replacement pattern cannot have q for
        input activation and dq for output of the fusion. Thus those nodes are removed from
        pattern to be matched, however we still want to make sure that input activation of
        the pattern is actually quantized and output is dequantized. Hence this filter.
        """
        input_dq_node = None
        output_q_node = None
        for pattern_node, original_node in match.nodes_map.items():
            if pattern_node.op == "placeholder":
                if (
                    original_node.target
                    == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                ):
                    input_dq_node = original_node
            # output node is not a separate node in the list of nodes seen in the matçh
            # it is a node in the node.users list of the last node.
            if (
                len(pattern_node.users) == 1
                and list(pattern_node.users.keys())[0].op == "output"
            ):
                output_node = list(original_node.users.keys())[0]
                if (
                    output_node.target
                    == torch.ops.quantized_decomposed.quantize_per_tensor.default
                ):
                    output_q_node = original_node
        return (input_dq_node is not None) and (output_q_node is not None)

    return _input_output_quantized_filter


def _get_quantized_qat_conv2d_bn_pattern(
    is_per_channel: bool,
    has_relu: bool,
    has_bias: bool,
    relu_is_inplace: bool,
) -> Callable:
    """
    Return the quantized version of QAT conv + BN pattern.
    This is based on `nniqat.ConvBn2d._forward_approximate`,
    used in QAT convert. We first match this pattern and replace
    it with the normal [conv - bn] pattern, then fold the BN
    weights into conv.
    """
    # TODO: allow setting eps
    bn_eps = 1e-5
    weight_quant_min = -127
    weight_quant_max = 127
    per_channel_axis = 0

    def _quantized_qat_conv2d_bn_pattern(
        x: torch.Tensor,
        conv_weight: torch.Tensor,
        bn_weight: torch.Tensor,
        bn_bias: torch.Tensor,
        bn_running_mean: torch.Tensor,
        bn_running_var: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        running_std = torch.sqrt(bn_running_var + bn_eps)
        scale_factor = bn_weight / running_std
        weight_shape = [1] * len(conv_weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(conv_weight.shape)
        bias_shape[1] = -1
        scaled_weight = conv_weight * scale_factor.reshape(weight_shape)
        if is_per_channel:
            scaled_weight = torch.ops.quantized_decomposed.quantize_per_channel(
                scaled_weight, kwargs['weight_scale'], kwargs['weight_zero_point'], per_channel_axis,
                weight_quant_min, weight_quant_max, torch.int8,
            )
            scaled_weight = torch.ops.quantized_decomposed.dequantize_per_channel(
                scaled_weight, kwargs['weight_scale'], kwargs['weight_zero_point'], per_channel_axis,
                weight_quant_min, weight_quant_max, torch.int8,
            )
        else:
            scaled_weight = torch.ops.quantized_decomposed.quantize_per_tensor(
                scaled_weight, 1.0, int(0), weight_quant_min, weight_quant_max, torch.int8,
            )
            scaled_weight = torch.ops.quantized_decomposed.dequantize_per_tensor(
                scaled_weight, 1.0, int(0), weight_quant_min, weight_quant_max, torch.int8,
            )
        if has_bias:
            zero_bias = torch.zeros_like(kwargs["conv_bias"], dtype=x.dtype)
            x = F.conv2d(x, scaled_weight, zero_bias)
        else:
            x = F.conv2d(x, scaled_weight, None)
        x = x / scale_factor.reshape(bias_shape)
        if has_bias:
            x = x + kwargs["conv_bias"].reshape(bias_shape)
        x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True, eps=bn_eps)
        if has_relu:
            if relu_is_inplace:
                x = F.relu_(x)
            else:
                x = F.relu(x)
        return x
    return _quantized_qat_conv2d_bn_pattern

def _get_folded_quantized_qat_conv2d_bn_pattern(
    is_per_channel: bool,
    has_relu: bool,
    has_bias: bool,
    relu_is_inplace: bool,
) -> Callable:
    """
    Quantized QAT conv - bn pattern with bn weights being folded into conv.
    """
    # TODO: allow setting eps
    bn_eps = 1e-5
    weight_quant_min = -127
    weight_quant_max = 127
    per_channel_axis = 0

    def _folded_quantized_qat_conv2d_bn_pattern(
        x: torch.Tensor,
        conv_weight: torch.Tensor,
        bn_weight: torch.Tensor,
        bn_bias: torch.Tensor,
        bn_running_mean: torch.Tensor,
        bn_running_var: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if is_per_channel:
            conv_weight = torch.ops.quantized_decomposed.quantize_per_channel(
                conv_weight, kwargs['weight_scale'], kwargs['weight_zero_point'], per_channel_axis,
                weight_quant_min, weight_quant_max, torch.int8,
            )
            conv_weight = torch.ops.quantized_decomposed.dequantize_per_channel(
                conv_weight, kwargs['weight_scale'], kwargs['weight_zero_point'], per_channel_axis,
                weight_quant_min, weight_quant_max, torch.int8,
            )
        else:
            conv_weight = torch.ops.quantized_decomposed.quantize_per_tensor(
                conv_weight, 1.0, int(0), weight_quant_min, weight_quant_max, torch.int8,
            )
            conv_weight = torch.ops.quantized_decomposed.dequantize_per_tensor(
                conv_weight, 1.0, int(0), weight_quant_min, weight_quant_max, torch.int8,
            )
        if has_bias:
            x = F.conv2d(x, conv_weight, kwargs["conv_bias"])
        else:
            x = F.conv2d(x, conv_weight, None)
        x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True, eps=bn_eps)
        if has_relu:
            if relu_is_inplace:
                x = F.relu_(x)
            else:
                x = F.relu(x)
        return x
    return _folded_quantized_qat_conv2d_bn_pattern

def _has_conv_bias_filter(
    match: "InternalMatch",  # type: ignore[name-defined]
    original_graph: Graph,
    pattern_graph: Graph,
) -> bool:
    """
    Match filter for the subgraph rewriter that returns True if the conv node in
    the original graph has bias.
    """
    for n in match.nodes_map.values():
        if n.target == torch.ops.aten.convolution.default:
            return n.args[2] is not None
    raise ValueError("Could not find conv node in matched conv + bn pattern")

def _no_conv_bias_filter(
    match: "InternalMatch",  # type: ignore[name-defined]
    original_graph: Graph,
    pattern_graph: Graph,
) -> bool:
    """
    Match filter for the subgraph rewriter that returns True if the conv node in
    the original graph does NOT have bias.
    """
    return not _has_conv_bias_filter(match, original_graph, pattern_graph)

def _get_fused_convbn_q_dq_nodes(nodes: List[Node]) -> Tuple[Node, Node]:
    """
    This util just identifies the q/dq nodes in the list of nodes.
    If there are more than one d nodes or more than one dq nodes, it will assert.
    """
    q_node, dq_node = None, None
    for n in nodes:
        if n.op != "call_function":
            continue
        if n.target in [
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
            torch.ops.quantized_decomposed.quantize_per_channel.default,
        ]:
            assert q_node is None
            q_node = n
        elif n.target in [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
        ]:
            assert dq_node is None
            dq_node = n
    assert q_node is not None
    assert dq_node is not None
    return (q_node, dq_node)

def _get_conv_bn_getitem_nodes(nodes: List[Node]) -> Tuple[Node, Node, Node]:
    """
    Helper function to extract the conv, bn, and getitem nodes from the list.
    This asserts that the list contains exactly one of each of the above nodes.

    Return a 3-tuple of (conv node, bn node, getitem node).
    """
    conv_node, bn_node, getitem_node = None, None, None
    for n in nodes:
        if n.op != "call_function":
            continue
        if n.target == torch.ops.aten.convolution.default:
            assert conv_node is None
            conv_node = n
        elif n.target == torch.ops.aten._native_batch_norm_legit.default:
            assert bn_node is None
            bn_node = n
        elif n.target == operator.getitem:
            assert getitem_node is None
            getitem_node = n
    assert conv_node is not None
    assert bn_node is not None
    assert getitem_node is not None
    return (conv_node, bn_node, getitem_node)

def _filter_nodes_map(nodes_map: Dict[Node, Node]) -> Dict[Node, Node]:
    """
    Return a filtered `nodes_map` returned from the subgraph rewriter.
    The filtered `nodes_map` will contain only nodes that are actually
    matched in the pattern, excluding None or placeholder nodes.
    """
    new_nodes_map: Dict[Node, Node] = {}
    for pattern_node, graph_node in nodes_map.items():
        # bias can be None
        if graph_node is None:
            continue
        # skip pattern placeholder nodes
        if pattern_node.op == "placeholder":
            continue
        new_nodes_map[pattern_node] = graph_node
    return new_nodes_map

def _copy_over_literal_conv_args(original_node: Node, new_node: Node):
    """
    Copy over literal args in conv, such as stride and padding, from the matched node
    in the original graph to its replacement in the new graph.

    This is needed due to the following limitation in the subgraph rewriter when used
    with dynamo export: literal (non-tensor) args are not supported in the match and
    replacement patterns. This is because dynamo export automatically inlines these
    literal args, making them dead placeholder nodes. In the future, we should check
    if dynamo export can optionally disable this inlining, or if subgraph rewriter
    can do the copying for us. See https://github.com/pytorch/pytorch/issues/100419.

    Note: Unlike other tensor args like conv weights and biases, literal args are
    preserved in the original nodes after replacement, so we can access them here.
    """
    assert original_node.target == torch.ops.aten.convolution.default
    assert new_node.target == torch.ops.aten.convolution.default
    # x, weight, bias, [stride, padding, dilation, transposed, output_padding, groups]
    new_node.args = new_node.args[:3] + original_node.args[3:]

def _update_conv_input_qspec_map_after_replacement(original_node: Node, replacement_node: Node):
    """
    Update the `input_qspec_map` in the annotation after subgraph rewriting.

    The original annotation referred to the nodes in the original graph,
    so the keys in the `input_qspec_map` will need to be updated to reflect
    the corresponding nodes in the replacement graph.
    """
    assert original_node.target == torch.ops.aten.convolution.default
    assert replacement_node.target == torch.ops.aten.convolution.default
    if "quantization_annotation" not in original_node.meta:
        return
    original_input_qspec_map = original_node.meta["quantization_annotation"].input_qspec_map
    input_qspec_map = {}
    # get the list of configs, it should be ordered as input, weight, bias
    # note: this is really hacky, we need a better solution, hopefully
    # in subgraph_rewriter, issue tracking the problem: https://github.com/pytorch/pytorch/issues/101820
    all_configs = list(original_input_qspec_map.items())
    # input activation
    input_qspec_map[replacement_node.args[0]] = all_configs[0][1]
    # weight
    input_qspec_map[replacement_node.args[1]] = all_configs[1][1]
    # bias
    if len(replacement_node.args) > 2 and len(all_configs) > 2:
        input_qspec_map[replacement_node.args[2]] = all_configs[2][1]
    replacement_node.meta["quantization_annotation"].input_qspec_map = input_qspec_map

def _update_special_qspecs_after_replacement(
    node: Node,
    original_to_replacement_node: Dict[Node, Node],
):
    """
    Update the `SharedQuantizationSpec`s and `DerivedQuantizationSpec`s
    used in `node`'s quantization annotation after subgraph rewriting.

    The original annotation referred to the nodes in the original graph,
    so the nodes used in these special quantization specs will need to
    be updated to the corresponding nodes in the replacement graph.
    """
    def _get_new_edge_or_node(edge_or_node: EdgeOrNode):
        if isinstance(edge_or_node, Node):
            _node = edge_or_node
            return original_to_replacement_node.get(_node, _node)
        elif isinstance(edge_or_node, Tuple[Node, Node]):
            src, dest = edge_or_node
            return (
                original_to_replacement_node.get(src, src),
                original_to_replacement_node.get(dest, dest),
            )
        else:
            raise ValueError("unexpected type for edge_or_node: ", type(edge_or_node))

    def _get_new_qspec(qspec: QuantizationSpecBase):
        if isinstance(qspec, SharedQuantizationSpec):
            new_edge_or_node = _get_new_edge_or_node(qspec.edge_or_node)
            return SharedQuantizationSpec(new_edge_or_node)
        elif isinstance(qspec, DerivedQuantizationSpec):
            new_derived_from = [_get_new_edge_or_node(x) for x in qspec.derived_from]
            return dataclasses.replace(qspec, derived_from=new_derived_from)
        else:
            return qspec

    if "quantization_annotation" not in node.meta:
        return
    annotation = node.meta["quantization_annotation"]
    for input_node, qspec in annotation.input_qspec_map.items():
        annotation.input_qspec_map[input_node] = _get_new_qspec(qspec)
    annotation.output_qspec = _get_new_qspec(annotation.output_qspec)

def _fuse_conv_bn_qat(m: GraphModule) -> GraphModule:
    """
    Given a graph of decomposed aten ops, replace the (conv + bn) pattern with
    the fused QAT subgraph equivalent. The input graph should already be annotated.
    The annotations in the original nodes will be preserved in the corresponding
    nodes in the new subgraph.

    Note: This also handles the (conv + bn + relu) pattern.
    """
    m.graph.eliminate_dead_code()
    m.recompile()
    example_inputs = _conv2d_bn_pattern_example_inputs
    match_pattern = _get_aten_graph_module(_conv2d_bn_pattern, example_inputs)

    # Step (1): Replace patterns with conv bias
    #
    # Here we do replacement separately for cases with and without conv bias, since
    # the replacement patterns for these two cases are substantially different.
    # TODO: use the public replace_pattern API once it also returns replacement nodes

    replacement_pattern_with_conv_bias = _get_aten_graph_module(
        _qat_conv2d_bn_pattern,
        example_inputs,
    )
    replacements_with_conv_bias = replace_pattern_with_filters(
        m,
        match_pattern,
        replacement_pattern_with_conv_bias,
        match_filters=[_has_conv_bias_filter],
        ignore_literals=True,
    )
    m.recompile()

    # Step (2): Replace patterns without conv bias

    replacement_pattern_no_conv_bias = _get_aten_graph_module(
        _qat_conv2d_bn_pattern_no_conv_bias,
        example_inputs,
    )
    replacements_no_conv_bias = replace_pattern_with_filters(
        m,
        match_pattern,
        replacement_pattern_no_conv_bias,
        match_filters=[_no_conv_bias_filter],
        ignore_literals=True,
    )
    m.recompile()

    # Step (3): Post processing
    #
    # Due to limited functionality in the subgraph rewriter, here we manually
    # update the replacement graph as follows:
    #
    #   (a) Copy over metadata from original subgraph. This ensures the stack traces
    #       and annotations are preserved in the new subgraph
    #
    #   (b) Copy over literal args for conv from the original subgraph
    #       TODO: do this for literal args for batchnorm as well
    #
    #   (c) Update all references of the old nodes in the original subgraph to refer
    #       to the corresponding nodes in the new subgraph in the annotations
    #
    # In the future, we should try to push as much of this functionality into the
    # subgraph rewriter as possible, so we don't have to manually copy anything over.
    # For more detail, see https://github.com/pytorch/pytorch/issues/100419.

    original_to_replacement_node = {}
    for r in replacements_with_conv_bias + replacements_no_conv_bias:
        (replacement_conv_node, replacement_bn_node, replacement_getitem_node) =\
            _get_conv_bn_getitem_nodes(r.replacements)

        # Step (3a): Copy over metadata for all three nodes in [conv - bn - getitem]
        for match_pattern_node, original_node in _filter_nodes_map(r.nodes_map).items():
            if original_node.target == torch.ops.aten.convolution.default:
                replacement_conv_node.meta = original_node.meta
                original_to_replacement_node[original_node] = replacement_conv_node
                # Step (3b): Copy over conv literal args
                _copy_over_literal_conv_args(original_node, replacement_conv_node)
                # Step (3c): Update old references in the conv node's input_qspec_map
                _update_conv_input_qspec_map_after_replacement(original_node, replacement_conv_node)
            if original_node.target == torch.ops.aten._native_batch_norm_legit.default:
                replacement_bn_node.meta = original_node.meta
                original_to_replacement_node[original_node] = replacement_bn_node
            if original_node.target == operator.getitem:
                replacement_getitem_node.meta = original_node.meta
                original_to_replacement_node[original_node] = replacement_getitem_node

    # Step (3c): Update old references in the special qspecs for all nodes in the graph
    for n in m.graph.nodes:
        _update_special_qspecs_after_replacement(n, original_to_replacement_node)

    return m

def _duplicate_dequantize_node(m: GraphModule):
    """
    Helper function to duplicate all dequantize nodes in the graph if the
    node has more than one user. For example:

    Before:
      quantize -> dequantize -> a
                          \\--> b
                          \\--> c

    After:
      quantize -> dequantize_1 -> a
            \\--> dequantize_2 -> b
            \\--> dequantize_3 -> c

    This is useful for subgraph rewriting. E.g. if we wish to match the
    pattern [dequantize - a] above, subgraph matching would fail because
    the dequantize node has users outside the matched portion of the graph.
    Instead, we match [dequantize_1 - a], which is safe.
    """
    dq_op = torch.ops.quantized_decomposed.dequantize_per_tensor
    for n in m.graph.nodes:
        if n.op != "call_function" or n.target != dq_op or len(n.users) == 1:
            continue
        for user in list(n.users):
            with m.graph.inserting_before(n):
                new_node = m.graph.create_node("call_function", dq_op, n.args, n.kwargs)
            user.replace_input_with(n, new_node)
        m.graph.erase_node(n)
    m.recompile()

def _remove_extra_dequantize(m: GraphModule):
    """
    Removes duplicate dequant nodes in the graph, for an operator that has
    multiple dequant nodes as a user, replace them with a single dequant node
    that can be shared across all the uses. This should be seen as the "reverse"
    of `_duplicate_dequantize_node`.
    """
    dq_op = torch.ops.quantized_decomposed.dequantize_per_tensor
    for n in m.graph.nodes:
        dq_users = [user for user in n.users if user.op == "call_function" and user.target == dq_op]
        if len(dq_users) > 1:
            with m.graph.inserting_after(dq_users[0]):
                new_node = m.graph.create_node("call_function", dq_op, dq_users[0].args, {})
            for dq_user in dq_users:
                dq_user.replace_all_uses_with(new_node)
                m.graph.erase_node(dq_user)
    m.recompile()

def _fold_conv_bn_qat(m: GraphModule) -> GraphModule:
    """
    Replace the quantized (conv + bn) pattern with conv with bn weights folded into the weights of conv.
    """
    m.graph.eliminate_dead_code()
    m.recompile()
    _duplicate_dequantize_node(m)

    # Step (1): Replace QAT pattern with simple [conv - bn] pattern
    replacements = []
    replacement_options = itertools.product(
        [True, False],  # is_per_channel
        [True, False],  # has_relu
        [True, False],  # has_bias
        [True, False],  # relu_is_inplace
    )
    for is_per_channel, has_relu, has_bias, relu_is_inplace in replacement_options:
        # For the cases without relu, `relu_is_inplace` is irrelevant, so here we arbitrarily
        # filter out one of the values for this flag to avoid having duplicate patterns
        if not has_relu and relu_is_inplace:
            continue
        example_inputs = _quantized_conv2d_bn_pattern_example_inputs
        kwargs = _get_quantized_conv2d_bn_pattern_example_inputs_kwargs(is_per_channel, has_bias)
        match_pattern = _get_quantized_qat_conv2d_bn_pattern(
            is_per_channel, has_relu, has_bias, relu_is_inplace,
        )
        match_pattern = _get_aten_graph_module(match_pattern, example_inputs, **kwargs)
        replacement_pattern = _get_folded_quantized_qat_conv2d_bn_pattern(
            is_per_channel, has_relu, has_bias, relu_is_inplace,
        )
        replacement_pattern = _get_aten_graph_module(replacement_pattern, example_inputs, **kwargs)
        replacements.extend(
            replace_pattern_with_filters(
                m,
                match_pattern,
                replacement_pattern,
                match_filters=[_get_input_output_quantized_filter()],
                ignore_literals=True,
            )
        )
    m.recompile()
    _remove_extra_dequantize(m)

    # Step (2): Fold BN weights into conv
    for r in replacements:
        (conv_node, bn_node, _) = _get_conv_bn_getitem_nodes(r.replacements)

        # get conv weight and bias
        conv_weight_dq = conv_node.args[1]
        assert isinstance(conv_weight_dq, Node)
        assert conv_weight_dq.target in (
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
        )
        conv_weight_q = conv_weight_dq.args[0]
        assert isinstance(conv_weight_q, Node)
        assert conv_weight_q.target in (
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
            torch.ops.quantized_decomposed.quantize_per_channel.default,
        )
        conv_weight = conv_weight_q.args[0]
        assert isinstance(conv_weight, Node)
        assert conv_weight.op == "get_attr"
        conv_bias = conv_node.args[2]
        assert conv_bias is None or isinstance(conv_bias, Node)

        (weight_q_node, weight_dq_node) = _get_fused_convbn_q_dq_nodes(r.replacements)
        original_weight_q_node = None
        original_weight_dq_node = None
        for pattern_node, original_node in r.nodes_map.items():
            if pattern_node.op == 'placeholder':
                continue
            if (
                original_node.target
                == torch.ops.quantized_decomposed.quantize_per_tensor.default
            ):
                assert original_weight_q_node is None
                original_weight_q_node = original_node
                weight_q_node.args = (
                    weight_q_node.args[:1] + original_weight_q_node.args[1:]
                )
            if (
                original_node.target
                == torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ):
                assert original_weight_dq_node is None
                original_weight_dq_node = original_node
                weight_dq_node.args = (
                    weight_dq_node.args[:1] + original_weight_dq_node.args[1:]
                )

        # fold bn weights into conv
        _fold_bn_weights_into_conv_node(conv_node, conv_weight, conv_bias, bn_node, m)

        # Copy over literal args for conv
        for original_node in _filter_nodes_map(r.nodes_map).values():
            if original_node.target == torch.ops.aten.convolution.default:
                _copy_over_literal_conv_args(original_node, conv_node)

    m.graph.eliminate_dead_code()
    m.recompile()
    return m
