# mypy: allow-untyped-defs
import functools
import itertools

import torch
from ..._dynamo.utils import counters

from ..pattern_matcher import Arg, CallFunction, KeywordArg
from .freezing_patterns import register_binary_folding_pattern

aten = torch.ops.aten
prims = torch.ops.prims


def mark_mixed_dtype_conv(conv):
    conv_dtype = conv.meta["val"].dtype
    if conv_dtype not in (torch.float16, torch.bfloat16):
        return

    if not len(conv.users) == 1:
        return

    conv_user = next(iter(conv.users.keys()))
    if not isinstance(conv_user.meta["val"], torch.Tensor):
        return

    if not conv_user.meta["val"].dtype == torch.float32:
        return

    while conv_user.target in _binary_ops:
        if not len(conv_user.users) == 1:
            return

        conv_user = next(iter(conv_user.users.keys()))

    if not (
        conv_user.target == prims.convert_element_type.default
        and conv_user.args[1] == conv_dtype
    ):
        return

    conv.meta["_allow_conv_mixed_dtype_folding"] = conv_dtype


def mark_mixed_dtype_allowed_convs(gm):
    """
    Mark convolutions which we will binary fold even with mixed precision constants. We constant fold in the higher precision
    for better accuracy and then recover the original precision after.
    """
    for node in gm.graph.find_nodes(
        op="call_function", target=aten.convolution.default
    ):
        mark_mixed_dtype_conv(node)


def recover_original_precision_folded_convs(gm):
    """
    After binary folding conv weights and biases to a higher dtype, recover the original precision they were in.
    """
    graph = gm.graph
    for node in graph.find_nodes(op="call_function", target=aten.convolution.default):
        orig_dtype = node.meta.get("_allow_conv_mixed_dtype_folding", None)
        if orig_dtype is None:
            continue

        with graph.inserting_before(node):
            for idx in [1, 2]:
                old_input = node.args[idx]
                if old_input is None:
                    continue

                new_input = graph.create_node(
                    "call_function",
                    prims.convert_element_type.default,
                    (old_input, orig_dtype),
                )
                node.replace_input_with(old_input, new_input)


_binary_ops = [aten.add.Tensor, aten.sub.Tensor, aten.mul.Tensor, aten.div.Tensor]


@functools.lru_cache(None)
def binary_folding_init():
    _conv_args = [Arg() for _ in range(9)]
    _computation_ops = [aten.convolution.default]
    _computation_calls = [CallFunction(aten.convolution.default, *_conv_args, _users=1)]

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
        if len(weight_shape) == len(other_shape) + 1:
            # weight shape is [o, i, *], other_shape is [o, 1...].
            for i in reversed(range(len(other_shape))):
                if i == 0 and weight_shape[0] == other_shape[i]:
                    continue
                if other_shape[i] != 1:
                    return False
        else:
            # weight shape is [o, i, *], other_shape is [1, i, *]
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

        if not len(conv_node.args[1].users) == 1:
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
                if not conv_node.meta.get("_allow_conv_mixed_dtype_folding", False):
                    return False

                if (
                    other_meta_value.dtype != torch.float
                    and weight_meta_value.dtype not in (torch.float16, torch.bfloat16)
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

        @register_binary_folding_pattern(
            CallFunction(binary_op, _computation_call, KeywordArg("other")),
            extra_check=_is_foldable_pattern,
        )
        def folded_op(match, *args, **kwargs):
            counters["inductor"]["binary_folding"] += 1
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
                new_computation_node.meta.update(computation_node.meta)
                graph.erase_node(binary_node)
                graph.erase_node(computation_node)
