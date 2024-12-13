# mypy: allow-untyped-defs
import copy
import dataclasses
import itertools
import operator
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.ao.quantization.pt2e.export_utils import _WrapperModule
from torch.ao.quantization.quantizer import (
    DerivedQuantizationSpec,
    EdgeOrNode,
    QuantizationSpecBase,
    SharedQuantizationSpec,
)
from torch.fx import Graph, GraphModule, Node
from torch.fx.subgraph_rewriter import replace_pattern_with_filters, ReplacedPatterns

from .utils import (
    _conv1d_bn_example_inputs,
    _get_aten_graph_module_for_pattern,
    _is_bn_node,
    _is_conv_or_conv_transpose_node,
    _is_conv_transpose_fn,
    fold_bn_weights_into_conv_node,
)


if TYPE_CHECKING:
    from torch.fx.passes.utils.matcher_with_name_node_map_utils import InternalMatch

__all__ = []  # type: ignore[var-annotated]

def _get_quantized_conv_bn_example_inputs_kwargs(
    is_per_channel: bool,
    has_bias: bool,
    bias_is_quantized: bool,
    is_cuda: bool,
) -> Dict[str, Any]:
    """
    Optional example inputs for quantized and folded conv-bn patterns
    used in convert, expressed as kwargs.
    """
    kwargs = {}
    # Per tensor quantization uses literals to represent scale and zero
    # point, so there is no need to include them here as kwargs
    if is_per_channel:
        kwargs["weight_scale"] = torch.tensor([1], dtype=torch.float)
        kwargs["weight_zero_point"] = torch.tensor([0], dtype=torch.int)
        if has_bias and bias_is_quantized:
            kwargs["bias_scale"] = torch.tensor([1], dtype=torch.float)
            kwargs["bias_zero_point"] = torch.tensor([0], dtype=torch.int)
    if has_bias:
        kwargs["conv_bias"] = torch.randn(1)
    if is_cuda:
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.cuda()
    return kwargs


def _get_conv_bn_pattern(conv_fn: Callable) -> Callable:
    def _conv_bn_pattern(
        x: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        bn_weight: torch.Tensor,
        bn_bias: torch.Tensor,
        bn_running_mean: torch.Tensor,
        bn_running_var: torch.Tensor,
    ) -> torch.Tensor:
        x = conv_fn(x, conv_weight, conv_bias)
        x = F.batch_norm(
            x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True
        )
        return x

    return _WrapperModule(_conv_bn_pattern)


# TODO: merge this with the `no_conv_bias` case
def _get_qat_conv_bn_pattern(conv_fn: Callable) -> Callable:
    def _qat_conv_bn_pattern(
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
        weight_in_channel_axis = 1 if _is_conv_transpose_fn(conv_fn) else 0
        weight_shape[weight_in_channel_axis] = -1
        bias_shape = [1] * len(conv_weight.shape)
        bias_shape[1] = -1
        scaled_weight = conv_weight * scale_factor.reshape(weight_shape)
        zero_bias = torch.zeros_like(conv_bias, dtype=x.dtype)
        x = conv_fn(x, scaled_weight, zero_bias)
        x = x / scale_factor.reshape(bias_shape)
        x = x + conv_bias.reshape(bias_shape)
        x = F.batch_norm(
            x,
            bn_running_mean,
            bn_running_var,
            bn_weight,
            bn_bias,
            training=True,
            eps=bn_eps,
        )
        return x

    return _WrapperModule(_qat_conv_bn_pattern)


def _get_qat_conv_bn_pattern_no_conv_bias(conv_fn: Callable) -> Callable:
    def _qat_conv_bn_pattern_no_conv_bias(
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
        Same as `_get_qat_conv_bn_pattern`, but handles the case with no conv bias.
        """
        # TODO: allow setting eps
        bn_eps = 1e-5
        running_std = torch.sqrt(bn_running_var + bn_eps)
        scale_factor = bn_weight / running_std
        weight_shape = [1] * len(conv_weight.shape)
        weight_in_channel_axis = 1 if _is_conv_transpose_fn(conv_fn) else 0
        weight_shape[weight_in_channel_axis] = -1
        bias_shape = [1] * len(conv_weight.shape)
        bias_shape[1] = -1
        scaled_weight = conv_weight * scale_factor.reshape(weight_shape)
        x = conv_fn(x, scaled_weight, None)
        x = x / scale_factor.reshape(bias_shape)
        x = F.batch_norm(
            x,
            bn_running_mean,
            bn_running_var,
            bn_weight,
            bn_bias,
            training=True,
            eps=bn_eps,
        )
        return x

    return _WrapperModule(_qat_conv_bn_pattern_no_conv_bias)


def _append_qdq(x, is_per_channel, is_bias, kwargs):
    """
    Helper function to append q-dq ops after `x`, using dummy values for the qparams
    and qmin/qmax. We use dummy values here because we match with `ignore_literals=True`
    and will manually replace these values after subgraph rewriting.

    Return the dq node.
    """
    # Dummy args to be passed into q-dq ops
    per_channel_axis = 0
    scale_key = "bias_scale" if is_bias else "weight_scale"
    zp_key = "bias_zero_point" if is_bias else "weight_zero_point"
    scale = kwargs[scale_key] if is_per_channel else 1.0
    zp = kwargs[zp_key] if is_per_channel else 0
    qmin = -127
    qmax = 127
    dtype = torch.int8

    qd = torch.ops.quantized_decomposed
    if is_per_channel:
        x = qd.quantize_per_channel(x, scale, zp, per_channel_axis, qmin, qmax, dtype)
        x = qd.dequantize_per_channel(x, scale, zp, per_channel_axis, qmin, qmax, dtype)
    else:
        x = qd.quantize_per_tensor(x, scale, zp, qmin, qmax, dtype)
        x = qd.dequantize_per_tensor(x, scale, zp, qmin, qmax, dtype)
    return x


def _get_quantized_qat_conv_bn_pattern(
    is_per_channel: bool,
    has_bias: bool,
    bias_is_quantized: bool,
    conv_fn: Callable,
    bn_is_training: bool,
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

    def _quantized_qat_conv_bn_pattern(
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
        scaled_weight = _append_qdq(
            scaled_weight,
            is_per_channel,
            is_bias=False,
            kwargs=kwargs,
        )
        if has_bias:
            zero_bias = torch.zeros_like(kwargs["conv_bias"], dtype=x.dtype)
            if bias_is_quantized:
                zero_bias = _append_qdq(
                    zero_bias,
                    is_per_channel,
                    is_bias=True,
                    kwargs=kwargs,
                )
            x = conv_fn(x, scaled_weight, zero_bias)
        else:
            x = conv_fn(x, scaled_weight, None)
        x = x / scale_factor.reshape(bias_shape)
        if has_bias:
            x = x + kwargs["conv_bias"].reshape(bias_shape)
        x = F.batch_norm(
            x,
            bn_running_mean,
            bn_running_var,
            bn_weight,
            bn_bias,
            training=bn_is_training,
            eps=bn_eps,
        )
        return x

    return _WrapperModule(_quantized_qat_conv_bn_pattern)


def _get_folded_quantized_qat_conv_bn_pattern(
    is_per_channel: bool,
    has_bias: bool,
    bias_is_quantized: bool,
    conv_fn: Callable,
    bn_is_training: bool,
) -> Callable:
    """
    Quantized QAT conv - bn pattern with bn weights being folded into conv.
    """
    # TODO: allow setting eps
    bn_eps = 1e-5

    def _folded_quantized_qat_conv_bn_pattern(
        x: torch.Tensor,
        conv_weight: torch.Tensor,
        bn_weight: torch.Tensor,
        bn_bias: torch.Tensor,
        bn_running_mean: torch.Tensor,
        bn_running_var: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        conv_weight = _append_qdq(
            conv_weight,
            is_per_channel,
            is_bias=False,
            kwargs=kwargs,
        )
        if has_bias:
            bias = kwargs["conv_bias"]
            if bias_is_quantized:
                bias = _append_qdq(
                    bias,
                    is_per_channel,
                    is_bias=True,
                    kwargs=kwargs,
                )
        else:
            bias = None
        x = conv_fn(x, conv_weight, bias)
        x = F.batch_norm(
            x,
            bn_running_mean,
            bn_running_var,
            bn_weight,
            bn_bias,
            training=bn_is_training,
            eps=bn_eps,
        )
        return x

    return _WrapperModule(_folded_quantized_qat_conv_bn_pattern)


def _has_conv_bias_filter(
    match: "InternalMatch",
    original_graph: Graph,
    pattern_graph: Graph,
) -> bool:
    """
    Match filter for the subgraph rewriter that returns True if the conv node in
    the original graph has bias.
    """
    for n in match.nodes_map.values():
        if _is_conv_or_conv_transpose_node(n):
            return len(n.args) > 2 and n.args[2] is not None
    raise ValueError("Could not find conv node in matched conv + bn pattern")


def _no_conv_bias_filter(
    match: "InternalMatch",
    original_graph: Graph,
    pattern_graph: Graph,
) -> bool:
    """
    Match filter for the subgraph rewriter that returns True if the conv node in
    the original graph does NOT have bias.
    """
    return not _has_conv_bias_filter(match, original_graph, pattern_graph)


def _is_quantize(n: Node) -> bool:
    return n.target in [
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
        torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
        torch.ops.quantized_decomposed.quantize_per_channel.default,
    ]


def _is_dequantize(n: Node) -> bool:
    return n.target in [
        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
        torch.ops.quantized_decomposed.dequantize_per_channel.default,
    ]


def _get_conv_bn_pattern_nodes(r: ReplacedPatterns) -> Dict[str, Tuple[Node, Node]]:
    """
    Helper function to extract the nodes in the conv-bn fusion pattern after
    subgraph rewriting, in the form of a map:

        {name: (original_node, replacement_node)}

    The following names must exist in the map:

        "conv", "conv_weight", "conv_input", "bn", "getitem"

    The following names may exist in the map:

        "conv_weight_q", "conv_weight_dq", "conv_bias",
        "conv_bias_q", "conv_bias_dq"
    """

    def _get_nodes(nodes: List[Node]) -> Tuple[Node, Node, Optional[Node]]:
        """
        Return a 3-tuple of (conv_node, bn_node, getitem_node).
        This asserts that the match contains exactly one of each node.
        """
        conv_node, bn_node, getitem_node = None, None, None
        for n in nodes:
            if n.op != "call_function":
                continue
            if _is_conv_or_conv_transpose_node(n):
                assert conv_node is None
                conv_node = n
            if _is_bn_node(n):
                assert bn_node is None
                bn_node = n
            if n.target == operator.getitem:
                assert getitem_node is None
                getitem_node = n
        assert conv_node is not None
        assert bn_node is not None
        # getitem_node might be None in new training IR
        return (conv_node, bn_node, getitem_node)

    def _get_q_dq_nodes(n: Node) -> Tuple[Node, Node, Node]:
        """
        Return a 3-tuple of (orig_node, q_node, dq_node).
        """
        assert _is_dequantize(n)
        q_node = n.args[0]
        assert isinstance(q_node, Node)
        assert _is_quantize(q_node)
        orig_node = q_node.args[0]
        assert isinstance(orig_node, Node)
        return (orig_node, q_node, n)

    original_nodes = list(_filter_nodes_map(r.nodes_map).values())
    o_conv, o_bn, o_getitem = _get_nodes(original_nodes)
    r_conv, r_bn, r_getitem = _get_nodes(r.replacements)

    # Create the mapping from original node to replacement node
    if o_getitem is None:
        # getitem is None is new training IR
        assert r_getitem is None
        mapping = {
            "conv": (o_conv, r_conv),
            "bn": (o_bn, r_bn),
        }
    else:
        # TODO: This branch is going through a deprecated branch and should be deleted soon,
        # after capture_pre_autograd_graph fully migrate to training IR
        # T199018392
        assert r_getitem is not None
        assert o_getitem is not None
        mapping = {
            "conv": (o_conv, r_conv),
            "bn": (o_bn, r_bn),
            "getitem": (o_getitem, r_getitem),
        }

    # Extract conv input and weight
    # Note: here we extract the original nodes indirectly through the pattern nodes
    # because the args of the original nodes are no longer available after replacement
    (p_conv, _, _) = _get_nodes(list(r.nodes_map.keys()))
    (p_conv_input, p_conv_weight, *_) = p_conv.args
    (r_conv_input, r_conv_weight, *_) = r_conv.args
    assert isinstance(p_conv_input, Node)
    assert isinstance(p_conv_weight, Node)
    assert isinstance(r_conv_input, Node)
    assert isinstance(r_conv_weight, Node)
    o_conv_input = r.nodes_map[p_conv_input]
    o_conv_weight = r.nodes_map[p_conv_weight]

    # If conv weight is quantized, extract the q - dq nodes
    if _is_dequantize(p_conv_weight):
        p_conv_weight, p_conv_weight_q, p_conv_weight_dq = _get_q_dq_nodes(
            p_conv_weight
        )
        r_conv_weight, r_conv_weight_q, r_conv_weight_dq = _get_q_dq_nodes(
            r_conv_weight
        )
        o_conv_weight = r.nodes_map[p_conv_weight]
        o_conv_weight_q = r.nodes_map[p_conv_weight_q]
        o_conv_weight_dq = r.nodes_map[p_conv_weight_dq]
        mapping["conv_weight_q"] = (o_conv_weight_q, r_conv_weight_q)
        mapping["conv_weight_dq"] = (o_conv_weight_dq, r_conv_weight_dq)
    mapping["conv_input"] = (o_conv_input, r_conv_input)
    mapping["conv_weight"] = (o_conv_weight, r_conv_weight)

    # Extract conv bias
    if len(p_conv.args) > 2 and len(r_conv.args) > 2:
        p_conv_bias = p_conv.args[2]
        r_conv_bias = r_conv.args[2]
        assert isinstance(p_conv_bias, Node)
        assert isinstance(r_conv_bias, Node)
        o_conv_bias = r.nodes_map[p_conv_bias]

        # If conv bias is quantized, extract the q - dq nodes
        if _is_dequantize(p_conv_bias):
            p_conv_bias, p_conv_bias_q, p_conv_bias_dq = _get_q_dq_nodes(p_conv_bias)
            r_conv_bias, r_conv_bias_q, r_conv_bias_dq = _get_q_dq_nodes(r_conv_bias)
            o_conv_bias = r.nodes_map[p_conv_bias]
            o_conv_bias_q = r.nodes_map[p_conv_bias_q]
            o_conv_bias_dq = r.nodes_map[p_conv_bias_dq]
            mapping["conv_bias_q"] = (o_conv_bias_q, r_conv_bias_q)
            mapping["conv_bias_dq"] = (o_conv_bias_dq, r_conv_bias_dq)
        mapping["conv_bias"] = (o_conv_bias, r_conv_bias)
    return mapping


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


# TODO: this is error prone, use the replace_literals_with_placeholders hack instead
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
    assert _is_conv_or_conv_transpose_node(original_node)
    assert _is_conv_or_conv_transpose_node(new_node)
    # x, weight, bias, [stride, padding, dilation, transposed, output_padding, groups]
    new_args = list(new_node.args)
    if len(new_args) < 3:
        # bias is optional, when it is not present, it means it is None
        new_args.append(None)
    new_node.args = tuple(new_args[:3]) + original_node.args[3:]


def _update_conv_input_qspec_map_after_replacement(
    original_node: Node, replacement_node: Node
):
    """
    Update the `input_qspec_map` in the annotation after subgraph rewriting.

    The original annotation referred to the nodes in the original graph,
    so the keys in the `input_qspec_map` will need to be updated to reflect
    the corresponding nodes in the replacement graph.
    """
    assert _is_conv_or_conv_transpose_node(original_node)
    assert _is_conv_or_conv_transpose_node(replacement_node)
    if "quantization_annotation" not in original_node.meta:
        return
    original_input_qspec_map = original_node.meta[
        "quantization_annotation"
    ].input_qspec_map
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
        elif (
            isinstance(edge_or_node, tuple)
            and len(edge_or_node) == 2
            and all(isinstance(x, Node) for x in edge_or_node)
        ):
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
    # Example inputs for conv-bn1d patterns
    _conv1d_bn_example_inputs = (
        torch.randn(1, 1, 3),  # x
        torch.randn(1, 1, 1),  # conv_weight
        torch.randn(1),  # conv_bias
        torch.randn(1),  # bn_weight
        torch.randn(1),  # bn_bias
        torch.randn(1),  # bn_running_mean
        torch.randn(1),  # bn_running_var
    )

    # Example inputs for conv-bn2d patterns
    _conv2d_bn_example_inputs = (
        torch.randn(1, 1, 3, 3),  # x
        torch.randn(1, 1, 1, 1),  # conv_weight
        torch.randn(1),  # conv_bias
        torch.randn(1),  # bn_weight
        torch.randn(1),  # bn_bias
        torch.randn(1),  # bn_running_mean
        torch.randn(1),  # bn_running_var
    )
    
    has_bn = any(_is_bn_node(n) for n in m.graph.nodes)
    if not has_bn:
        return m
    is_cuda_options = [True, False] if torch.cuda.is_available() else [False]
    for is_cuda in is_cuda_options:
        m = _fuse_conv_bn_qat_helper(
            m, F.conv1d, _conv1d_bn_example_inputs, is_cuda=is_cuda
        )
        m = _fuse_conv_bn_qat_helper(
            m, F.conv2d, _conv2d_bn_example_inputs, is_cuda=is_cuda
        )
        m = _fuse_conv_bn_qat_helper(
            m, F.conv_transpose1d, _conv1d_bn_example_inputs, is_cuda=is_cuda
        )
        m = _fuse_conv_bn_qat_helper(
            m, F.conv_transpose2d, _conv2d_bn_example_inputs, is_cuda=is_cuda
        )
    return m


def _fuse_conv_bn_qat_helper(
    m: GraphModule,
    conv_fn: Callable,
    example_inputs: Tuple[Any, ...],
    is_cuda: bool,
) -> GraphModule:
    """
    Given a graph of decomposed aten ops, replace the (conv + bn) pattern with
    the fused QAT subgraph equivalent. The input graph should already be annotated.
    The annotations in the original nodes will be preserved in the corresponding
    nodes in the new subgraph.

    Note: This also handles the (conv + bn + relu) pattern.
    """
    m.graph.eliminate_dead_code()
    m.recompile()

    from torch._export import gm_using_training_ir

    using_training_ir = gm_using_training_ir(m)

    conv_bn_pattern = _get_conv_bn_pattern(conv_fn)
    match_pattern = _get_aten_graph_module_for_pattern(
        conv_bn_pattern,
        example_inputs,
        is_cuda,
        using_training_ir=using_training_ir,
    )

    # Step (1): Replace patterns with conv bias
    #
    # Here we do replacement separately for cases with and without conv bias, since
    # the replacement patterns for these two cases are substantially different.
    # TODO: use the public replace_pattern API once it also returns replacement nodes

    qat_conv_bn_pattern = _get_qat_conv_bn_pattern(conv_fn)
    replacement_pattern_with_conv_bias = _get_aten_graph_module_for_pattern(
        qat_conv_bn_pattern,
        example_inputs,
        is_cuda,
        using_training_ir=using_training_ir,
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

    qat_conv_bn_pattern_no_conv_bias = _get_qat_conv_bn_pattern_no_conv_bias(conv_fn)
    replacement_pattern_no_conv_bias = _get_aten_graph_module_for_pattern(
        qat_conv_bn_pattern_no_conv_bias,
        example_inputs,
        is_cuda,
        using_training_ir=using_training_ir,
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

    all_original_to_replacement_nodes = {}
    for r in replacements_with_conv_bias + replacements_no_conv_bias:
        replacement_dict = _get_conv_bn_pattern_nodes(r)
        # The original conv node's "nn_module_stack"
        conv_nn_module = replacement_dict["conv"][0].meta.get("nn_module_stack", None)
        for k, node_tuple in replacement_dict.items():
            original_node, replacement_node = node_tuple
            # Step (3a): Copy over metadata for all nodes in [conv - bn - getitem]
            replacement_node.meta = original_node.meta
            # If original_node is a get_attr node, it doesn't have nn_module_stack.
            # In this case, we copy nn_module_stack from the original conv node.
            if (
                k in ["conv_input", "conv_weight"]
                and conv_nn_module
                and "nn_module_stack" not in replacement_node.meta
            ):
                replacement_node.meta["nn_module_stack"] = copy.deepcopy(conv_nn_module)
            if _is_conv_or_conv_transpose_node(original_node):
                # Step (3b): Copy over conv literal args
                _copy_over_literal_conv_args(original_node, replacement_node)
                # Step (3c): Update old references in the conv node's input_qspec_map
                _update_conv_input_qspec_map_after_replacement(
                    original_node, replacement_node
                )
            all_original_to_replacement_nodes[original_node] = replacement_node

    # Step (3c): Update old references in the special qspecs for all nodes in the graph
    for n in m.graph.nodes:
        _update_special_qspecs_after_replacement(n, all_original_to_replacement_nodes)

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
        dq_users = [
            user
            for user in n.users
            if user.op == "call_function" and user.target == dq_op
        ]
        if len(dq_users) > 1:
            with m.graph.inserting_after(dq_users[0]):
                new_node = m.graph.create_node(
                    "call_function", dq_op, dq_users[0].args, {}
                )
            for dq_user in dq_users:
                dq_user.replace_all_uses_with(new_node)
                m.graph.erase_node(dq_user)
    m.recompile()


def _copy_over_q_dq_args(original_node: Node, replacement_node: Node):
    """
    Given a pair of quantize or dequantize nodes, copy over all literal args
    from the original node to the replacement node.
    """
    # For quantize_per_tensor, scale and zp are literals and need to be copied
    # For quantize_per_channel, scale and zp are get_attr nodes and should be skipped
    assert original_node.target == replacement_node.target
    if original_node.target in (
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    ):
        # Args: input, [scale, zp, qmin, qmax, dtype]
        start_copy_arg_index = 1
    elif original_node.target in (
        torch.ops.quantized_decomposed.quantize_per_channel.default,
        torch.ops.quantized_decomposed.dequantize_per_channel.default,
    ):
        # Args: input, scale, zp, [axis, qmin, qmax, dtype]
        start_copy_arg_index = 3
    else:
        raise ValueError(
            f"Expected quantize/dequantize nodes, got '{original_node.target}'"
        )
    replacement_node.args = (
        replacement_node.args[:start_copy_arg_index]
        + original_node.args[start_copy_arg_index:]
    )


def _fold_conv_bn_qat(m: GraphModule) -> GraphModule:
    # Example inputs for quantized and folded conv-bn1d patterns used in convert
    _quantized_conv1d_bn_example_inputs = (
        torch.randn(1, 1, 3),  # x
        torch.randn(1, 1, 1),  # conv_weight
        torch.randn(1),  # bn_weight
        torch.randn(1),  # bn_bias
        torch.randn(1),  # bn_running_mean
        torch.randn(1),  # bn_running_var
    )

    # Example inputs for quantized and folded conv-bn2d patterns used in convert
    _quantized_conv2d_bn_example_inputs = (
        torch.randn(1, 1, 3, 3),  # x
        torch.randn(1, 1, 1, 1),  # conv_weight
        torch.randn(1),  # bn_weight
        torch.randn(1),  # bn_bias
        torch.randn(1),  # bn_running_mean
        torch.randn(1),  # bn_running_var
    )

    has_bn = any(_is_bn_node(n) for n in m.graph.nodes)
    if not has_bn:
        return m
    is_cuda_options = [True, False] if torch.cuda.is_available() else [False]
    for is_cuda in is_cuda_options:
        m = _fold_conv_bn_qat_helper(
            m, F.conv1d, _quantized_conv1d_bn_example_inputs, is_cuda=is_cuda
        )
        m = _fold_conv_bn_qat_helper(
            m, F.conv2d, _quantized_conv2d_bn_example_inputs, is_cuda=is_cuda
        )
        m = _fold_conv_bn_qat_helper(
            m, F.conv_transpose1d, _quantized_conv1d_bn_example_inputs, is_cuda=is_cuda
        )
        m = _fold_conv_bn_qat_helper(
            m, F.conv_transpose2d, _quantized_conv2d_bn_example_inputs, is_cuda=is_cuda
        )

    # remove in place add from batchnorm tracking traning stats
    for node in m.graph.nodes:
        if (
            node.target == torch.ops.aten.add_.Tensor
            and node.args[0].op == "get_attr"
            and node.args[1] == 1
            and torch.nn.modules.batchnorm.BatchNorm2d
            in [val[1] for val in node.meta["source_fn_stack"]]
        ):
            m.graph.erase_node(node)

    m.graph.eliminate_dead_code()
    m.recompile()

    return m


def _fold_conv_bn_qat_helper(
    m: GraphModule,
    conv_fn: Callable,
    example_inputs: Tuple[Any, ...],
    is_cuda: bool,
) -> GraphModule:
    """
    Replace the quantized (conv + bn) pattern with conv with bn weights folded into the weights of conv.
    """
    from torch._export import gm_using_training_ir

    using_training_ir = gm_using_training_ir(m)

    m.graph.eliminate_dead_code()
    m.recompile()
    _duplicate_dequantize_node(m)

    # Step (1): Replace QAT pattern with simple [conv - bn] pattern
    replacements = []
    replacement_options = itertools.product(
        [True, False],  # is_per_channel
        [True, False],  # has_bias
        [True, False],  # bias_is_quantized
        [True, False],  # bn_is_training
    )
    for (
        is_per_channel,
        has_bias,
        bias_is_quantized,
        bn_is_training,
    ) in replacement_options:
        # For the cases without bias, `bias_is_quantized` is irrelevant, so here we arbitrarily
        # filter out one of the values for this flag to avoid having duplicate patterns
        if not has_bias and bias_is_quantized:
            continue
        kwargs = _get_quantized_conv_bn_example_inputs_kwargs(
            is_per_channel, has_bias, bias_is_quantized, is_cuda
        )
        match_pattern = _get_quantized_qat_conv_bn_pattern(
            is_per_channel, has_bias, bias_is_quantized, conv_fn, bn_is_training
        )
        match_pattern = _get_aten_graph_module_for_pattern(
            match_pattern,
            example_inputs,
            is_cuda,
            using_training_ir=using_training_ir,
            **kwargs,
        )
        replacement_pattern = _get_folded_quantized_qat_conv_bn_pattern(
            is_per_channel, has_bias, bias_is_quantized, conv_fn, bn_is_training
        )
        replacement_pattern = _get_aten_graph_module_for_pattern(
            replacement_pattern,
            example_inputs,
            is_cuda,
            using_training_ir=using_training_ir,
            **kwargs,
        )
        replacements.extend(
            replace_pattern_with_filters(
                m,
                match_pattern,
                replacement_pattern,
                ignore_literals=True,
            )
        )
    m.recompile()
    _remove_extra_dequantize(m)

    for r in replacements:
        node_map = _get_conv_bn_pattern_nodes(r)

        # Step (2): Copy over metadata from original subgraph
        for original_node, replacement_node in node_map.values():
            replacement_node.meta = original_node.meta

        # Step (3): Copy over args for weight (and optionally bias) q - dq nodes
        _copy_over_q_dq_args(*node_map["conv_weight_q"])
        _copy_over_q_dq_args(*node_map["conv_weight_dq"])
        if "conv_bias_q" in node_map:
            assert "conv_bias_dq" in node_map
            _copy_over_q_dq_args(*node_map["conv_bias_q"])
            _copy_over_q_dq_args(*node_map["conv_bias_dq"])

        # Step (4): Fold BN weights into conv
        conv_bias = None
        (_, conv_node) = node_map["conv"]
        (_, bn_node) = node_map["bn"]
        (_, conv_weight) = node_map["conv_weight"]
        if "conv_bias" in node_map:
            (_, conv_bias) = node_map["conv_bias"]
        fold_bn_weights_into_conv_node(conv_node, conv_weight, conv_bias, bn_node, m)

        # Copy over literal args for conv
        for original_node in _filter_nodes_map(r.nodes_map).values():
            if _is_conv_or_conv_transpose_node(original_node):
                _copy_over_literal_conv_args(original_node, conv_node)

    m.graph.eliminate_dead_code()
    m.recompile()
    return m
