import functools
import itertools

import torch
from .. import config
from ..pattern_matcher import (
    _return_true,
    Arg,
    CallFunction,
    inference_graph,
    init_once_fakemode,
    KeywordArg,
    PatternMatcherPass,
    register_graph_pattern,
    register_replacement,
    stable_topological_sort,
)

aten = torch.ops.aten

# First pass_patterns[0] are applied, then [1], then [2]
pass_patterns = [
    PatternMatcherPass(),
    PatternMatcherPass(),
    PatternMatcherPass(),
]


def freezing_passes(gm: torch.fx.GraphModule):
    """
    Passes that are applied to the graph to freeze pass.
    """

    lazy_init()
    for patterns in pass_patterns:
        patterns.apply(gm.graph)

    if torch._C._has_mkldnn and config.cpp.weight_prepack:
        from .mkldnn_fusion import _eliminate_duplicate_packed_nodes

        _eliminate_duplicate_packed_nodes(gm)

    stable_topological_sort(gm.graph)
    gm.recompile()
    gm.graph.lint()


@init_once_fakemode
def lazy_init():
    if torch._C._has_mkldnn and config.cpp.weight_prepack:
        from .mkldnn_fusion import _mkldnn_weight_pack_init

        _mkldnn_weight_pack_init()

    addmm_patterns_init()
    binary_folding_init()


def register_freezing_graph_pattern(pattern, extra_check=_return_true, pass_number=0):
    return register_graph_pattern(
        pattern,
        extra_check=extra_check,
        pass_dict=pass_patterns[pass_number],
    )


@functools.lru_cache(None)
def addmm_patterns_init():
    if torch.cuda.is_available():
        # workaround https://github.com/pytorch/pytorch/issues/97894
        device = "cuda"
    else:
        device = "cpu"
    val = functools.partial(torch.empty, (10, 10), device=device, requires_grad=False)

    def check_concat_weights(match):
        weights = [
            match.kwargs["w1"],
            match.kwargs["w2"],
            match.kwargs["w3"],
        ]
        return all(
            w.op == "get_attr" and w.meta["val"].shape == weights[0].meta["val"].shape
            for w in weights
        )

    def matmul_fuse_pattern(inp, w1, w2, w3):
        return (inp @ w1, inp @ w2, inp @ w3)

    def matmul_replacement(inp, w1, w2, w3):
        cat_t = torch.cat((w1, w2, w3), dim=1)
        mm = inp @ cat_t
        return mm.chunk(3, dim=1)

    register_replacement(
        matmul_fuse_pattern,
        matmul_replacement,
        [val(), val(), val(), val()],
        inference_graph,
        pass_patterns[0],
        extra_check=check_concat_weights,
        exclusive_arg_names=("w1", "w2", "w3"),
    )

    def addmm_fuse_pattern_second(inp, w1, w2, w3, b1, b2, b3):
        return (
            aten.addmm(b1, inp, w1),
            aten.addmm(b2, inp, w2),
            aten.addmm(b3, inp, w3),
        )

    def addmm_fuse_replacement_second(inp, w1, w2, w3, b1, b2, b3):
        cat_w = torch.cat((w1, w2, w3), dim=1)
        cat_b = torch.cat((b1, b2, b3))
        return aten.addmm(cat_b, inp, cat_w).chunk(3, dim=1)

    register_replacement(
        addmm_fuse_pattern_second,
        addmm_fuse_replacement_second,
        [val() for _ in range(7)],
        inference_graph,
        pass_patterns[0],
        extra_check=check_concat_weights,
        exclusive_arg_names=("w1", "w2", "w3", "b1", "b2", "b3"),
    )


@functools.lru_cache(None)
def binary_folding_init():
    _conv_args = [Arg() for _ in range(9)]
    _computation_ops = [aten.convolution.default]
    _binary_ops = [aten.add.Tensor, aten.sub.Tensor, aten.mul.Tensor, aten.div.Tensor]
    _computation_calls = [CallFunction(aten.convolution.default, *_conv_args)]

    def _is_constant_node(node):
        return isinstance(node, torch.fx.Node) and node.op == "get_attr"

    def _check_conv_and_broadcast_op(conv_node, other):
        if not all(
            _is_constant_node(other)
            for n in [conv_node.args[1], conv_node.args[2], other]
        ):
            return False

        weight_meta_value = conv_node.args[1].meta.get("val")
        other_meta_value = other.meta.get("val")
        if weight_meta_value is None or other_meta_value is None:
            return False

        # TODO: weight_meta_value.dtype < other_meta_value.dtype?
        if weight_meta_value.dtype != other_meta_value.dtype:
            return False
        weight_shape = weight_meta_value.shape
        other_shape = other_meta_value.shape
        # make sure len(weight_shape) ==  len(other_shape)(or +1)
        if (
            len(weight_shape) != len(other_shape)
            and len(weight_shape) != len(other_shape) + 1
        ):
            return False

        if len(weight_shape) == len(other_shape):
            for i in reversed(range(len(other_shape))):
                if i == 1 and weight_shape[0] == other_shape[i]:
                    continue
                if other_shape[i] != 1:
                    return False
        else:
            for i in reversed(range(len(other_shape))):
                if i == 0 and weight_shape[0] == other_shape[i]:
                    continue
                if other_shape[i] != 1:
                    return False

        return True

    def _is_foldable_pattern(match):
        binary_node = match.output_node()
        computation_node = binary_node.args[0]
        other = binary_node.args[1]
        if binary_node.args[0].target not in _computation_ops:
            computation_node = binary_node.args[1]
            other = binary_node.args[0]
        if binary_node.args[0].target == aten.convolution.default:
            return _check_conv_and_broadcast_op(computation_node, other)

        return False

    def _create_new_conv_node(graph, conv_node, binary_node, other):
        assert conv_node.target == aten.convolution.default
        conv_args = list(conv_node.args)
        weight_meta_value = conv_node.args[1].meta.get("val")
        bias = conv_args[2]
        if binary_node.target in [aten.add.Tensor, aten.sub.Tensor]:
            other_reshape = graph.create_node(
                "call_function",
                aten.reshape.default,
                (other, (weight_meta_value.size(0),)),
            )
            if bias is not None:
                new_bias = graph.create_node(
                    "call_function", binary_node.target, (bias, other_reshape)
                )
            else:
                new_bias = graph.create_node(
                    "call_function", binary_node.target, (0, other_reshape)
                )
            conv_args[2] = new_bias
        else:
            assert binary_node.target in [aten.mul.Tensor, aten.div.Tensor]
            weight_broadcast_shape = [1 for _ in range(len(weight_meta_value.shape))]
            weight_broadcast_shape[0] = weight_meta_value.size(0)
            other_reshape1 = graph.create_node(
                "call_function",
                aten.reshape.default,
                (other, tuple(weight_broadcast_shape)),
            )
            conv_args[1] = graph.create_node(
                "call_function", binary_node.target, (conv_args[1], other_reshape1)
            )
            if conv_args[2] is not None:
                other_reshape = graph.create_node(
                    "call_function",
                    aten.reshape.default,
                    (other, (weight_meta_value.size(0),)),
                )
                conv_args[2] = graph.create_node(
                    "call_function", binary_node.target, (bias, other_reshape)
                )
        return graph.create_node("call_function", conv_node.target, tuple(conv_args))

    for _computation_call, binary_op in itertools.product(
        _computation_calls, _binary_ops
    ):

        @register_freezing_graph_pattern(
            CallFunction(binary_op, _computation_call, KeywordArg("other")),
            extra_check=_is_foldable_pattern,
        )
        def folded_op(match, *args, **kwargs):
            other = kwargs.get("other")
            binary_node = match.output_node()
            computation_node = (
                binary_node.args[0]
                if binary_node.args[0].target in _computation_ops
                else binary_node.args[1]
            )
            graph = match.graph
            with graph.inserting_before(binary_node):
                # TODO: support linear?
                assert computation_node.target == aten.convolution.default
                new_computation_node = _create_new_conv_node(
                    graph, computation_node, binary_node, other
                )
                binary_node.replace_all_uses_with(new_computation_node)
                graph.erase_node(binary_node)
                graph.erase_node(computation_node)
