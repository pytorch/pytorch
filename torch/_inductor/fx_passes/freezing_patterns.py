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

    """
    In order to fuse add/sub/mul/div with conv, the dimensions of its
    constant tensor must satisfy the following:
    - with resizing, broadcast to w/ weight/bias tensor shape
    - broadcast to the conv output shape
    It needs to have a shape that can resize to weight/bias
    tensor shape because we need to run the op with the conv
    weights/bias without changing their sizes.
    It needs to broadcast to the conv output shape so that we do
    accidentally change the shape of op output by pre-fusing it
    compared to eager.
    The only dimension value shared by weight/bias/conv output
    is they all contain a dim with value = channels-out. In the
    conv output tensor, this is in the second dimension,
    so the pointwise op tensor may have a second dimension of
    value == channels-out, but all the other dimensions have to be 1
    """

    def _op_not_broadcasting_with_conv(weight_tensor, other_tensor):
        # According to opDoesNotBroadCastWithConv of frozen_conv_folding.cpp
        weight_shape = weight_tensor.shape
        other_shape = other_tensor.shape
        if len(weight_shape) < len(other_shape):
            return False
        for i in reversed(range(len(other_shape))):
            if i == 1 and weight_shape[0] == other_shape[i]:
                continue
            if other_shape[i] != 1:
                return False
        return True

    def _check_conv_and_broadcast_op(conv_node, other):
        # According to checkConvAndBroadcastingOpPreConditions of frozen_conv_folding.cpp.
        # conv.weight
        if conv_node.args[1].op != "get_attr":
            return False
        # conv.bias
        if conv_node.args[1] is not None and conv_node.args[1].op != "get_attr":
            return False
        if (
            not isinstance(other, int)
            and not isinstance(other, float)
            and other.op != "get_attr"
        ):
            return False

        weight_meta_value = conv_node.args[1].meta.get("val")
        if weight_meta_value is None:
            return False
        # Avoid fusing op that causes type promotion
        # restricting to float avoids int/float difficulties with scalar overload
        if not weight_meta_value.is_floating_point():
            return False
        if isinstance(other, torch.fx.Node) and other.op == "get_attr":
            other_meta_value = other.meta.get("val")
            if not other_meta_value.is_floating_point():
                return False
            if (
                torch.promote_types(other_meta_value.dtype, weight_meta_value.dtype)
                != weight_meta_value.dtype
            ):
                return False
            if not _op_not_broadcasting_with_conv(weight_meta_value, other_meta_value):
                return False
        else:
            # TODO: support scalar case
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

    def resize_scalar_or_tensor_to_shape(graph, other, shape):
        # TODO: support scalar case
        if other.meta.get("val").numel() == 1:
            # expand errors if the shape input has less # dims than the tensor input
            res = graph.create_node(
                "call_function",
                aten.reshape.default,
                (other, (1,)),
            )
            res = graph.create_node(
                "call_function",
                aten.expand.default,
                (res, shape),
            )
        else:
            res = graph.create_node(
                "call_function",
                aten.reshape.default,
                (other, shape),
            )
        return res

    def _create_new_conv_node(graph, conv_node, binary_node, other):
        assert conv_node.target == aten.convolution.default
        conv_args = list(conv_node.args)
        weight_meta_value = conv_node.args[1].meta.get("val")
        bias = conv_args[2]
        if binary_node.target in [aten.add.Tensor, aten.sub.Tensor]:
            other_reshape = resize_scalar_or_tensor_to_shape(
                graph, other, (weight_meta_value.size(0),)
            )
            new_bias = graph.create_node(
                "call_function",
                binary_node.target,
                (0 if bias is None else bias, other_reshape),
            )
            conv_args[2] = new_bias
        else:
            assert binary_node.target in [aten.mul.Tensor, aten.div.Tensor]
            weight_broadcast_shape = [1 for _ in range(len(weight_meta_value.shape))]
            weight_broadcast_shape[0] = weight_meta_value.size(0)
            other_reshape1 = resize_scalar_or_tensor_to_shape(
                graph, other, tuple(weight_broadcast_shape)
            )
            new_weight = graph.create_node(
                "call_function", binary_node.target, (conv_args[1], other_reshape1)
            )
            new_weight.meta.update(conv_args[1].meta)
            conv_args[1] = new_weight
            if bias is not None:
                other_reshape = resize_scalar_or_tensor_to_shape(
                    graph, other, (weight_meta_value.size(0),)
                )
                new_bias = graph.create_node(
                    "call_function", binary_node.target, (bias, other_reshape)
                )
                new_bias.meta.update(bias.meta)
                conv_args[2] = new_bias
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
                new_computation_node.meta.update(binary_node.meta)
                graph.erase_node(binary_node)
                graph.erase_node(computation_node)
