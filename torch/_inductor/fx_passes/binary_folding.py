# mypy: allow-untyped-defs
import functools
import itertools

import torch

from ..._dynamo.utils import counters
from .. import config
from ..pattern_matcher import Arg, CallFunction, KeywordArg
from .freezing_patterns import register_binary_folding_pattern


aten = torch.ops.aten
prims = torch.ops.prims


def mark_mixed_dtype(computation_node):
    computation_node_dtype = computation_node.meta["val"].dtype
    if computation_node_dtype not in (torch.float16, torch.bfloat16):
        return

    if not len(computation_node.users) == 1:
        return

    computation_node_user = next(iter(computation_node.users.keys()))
    if not isinstance(computation_node_user.meta["val"], torch.Tensor):
        return

    if not computation_node_user.meta["val"].dtype == torch.float32:
        return

    while computation_node_user.target in _binary_ops:
        if not len(computation_node_user.users) == 1:
            return

        computation_node_user = next(iter(computation_node_user.users.keys()))

    if computation_node_user.target != prims.convert_element_type.default:
        return

    computation_node.meta["_allow_mixed_dtype_folding"] = computation_node_dtype


def mark_mixed_dtype_allowed_computation_ops(gm):
    """
    Mark convolutions/linear which we will binary fold even with mixed precision constants. We constant fold in the higher precision
    for better accuracy and then recover the original precision after.
    """
    for target in [aten.convolution.default, aten.addmm.default, aten.mm.default]:
        for node in gm.graph.find_nodes(op="call_function", target=target):
            mark_mixed_dtype(node)


def recover_original_precision_folded_computation_ops(gm):
    """
    After binary folding conv/linear weights and biases to a higher dtype, recover the original precision they were in.
    """
    graph = gm.graph
    for target, idx in (
        (aten.convolution.default, (1, 2)),
        (aten.addmm.default, (0, 2)),
        (aten.mm.default, (1,)),
    ):
        for node in graph.find_nodes(op="call_function", target=target):
            orig_dtype = node.meta.get("_allow_mixed_dtype_folding", None)
            if orig_dtype is None:
                continue

            with graph.inserting_before(node):
                for i in idx:
                    old_input = node.args[i]
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
    _addmm_args = [Arg() for _ in range(3)]
    _mm_args = [Arg() for _ in range(2)]
    _computation_ops = [aten.convolution.default, aten.addmm.default, aten.mm.default]
    _computation_calls = [
        CallFunction(aten.convolution.default, *_conv_args, _users=1),
        CallFunction(aten.addmm.default, *_addmm_args, _users=1),
        CallFunction(
            aten.reshape.default,
            CallFunction(aten.addmm.default, *_addmm_args, _users=1),
            Arg(),
            _users=1,
        ),
        CallFunction(aten.mm.default, *_mm_args, _users=1),
        CallFunction(
            aten.reshape.default,
            CallFunction(aten.mm.default, *_mm_args, _users=1),
            Arg(),
            _users=1,
        ),
    ]

    """
    In order to fuse add/sub/mul/div with conv/linear, the dimensions of its
    constant tensor must satisfy the following:
    - with resizing, broadcast to w/ weight/bias tensor shape
    - broadcast to the conv/linear output shape
    It needs to have a shape that can resize to weight/bias
    tensor shape because we need to run the op with the conv/linear
    weights/bias without changing their sizes.
    It needs to broadcast to the conv/linear output shape so that we do
    accidentally change the shape of op output by pre-fusing it
    compared to eager.
    The only dimension value shared by weight, bias, and conv/linear output
    is they all contain a dim with value = channels-out. In the
    conv/linear output tensor, this is in the second dimension,
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

    def _op_not_broadcasting_with_linear(weight_tensor, other_tensor, has_reshape):
        weight_shape = weight_tensor.shape
        other_shape = other_tensor.shape
        other_shapes = [
            torch.Size(
                [
                    weight_shape[1],
                ]
            ),
            torch.Size([1, weight_shape[1]]),
            torch.Size(
                [
                    1,
                ]
            ),
            torch.Size([1, 1]),
        ]
        if has_reshape:
            other_shapes.extend(
                [
                    torch.Size([1, 1, weight_shape[1]]),
                    torch.Size([1, 1, 1]),
                ]
            )
        return other_shape in other_shapes

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
            if not other_meta_value.is_floating_point():  # type: ignore[union-attr]
                return False
            if (
                torch.promote_types(other_meta_value.dtype, weight_meta_value.dtype)  # type: ignore[union-attr]
                != weight_meta_value.dtype
            ):
                if not conv_node.meta.get("_allow_mixed_dtype_folding", False):
                    return False

                if (
                    other_meta_value.dtype != torch.float  # type: ignore[union-attr]
                    and weight_meta_value.dtype not in (torch.float16, torch.bfloat16)
                ):
                    return False

            if not _op_not_broadcasting_with_conv(weight_meta_value, other_meta_value):
                return False
        elif not isinstance(other, float):
            return False

        return True

    def _check_linear_and_broadcast_op(linear_node, other, has_reshape):
        weight_node = (
            linear_node.args[2]
            if linear_node.target is aten.addmm.default
            else linear_node.args[1]
        )
        bias_node = (
            linear_node.args[0] if linear_node.target is aten.addmm.default else None
        )
        if weight_node.op != "get_attr":
            return False
        if bias_node is not None and bias_node.op != "get_attr":
            return False
        if (
            not isinstance(other, int)
            and not isinstance(other, float)
            and other.op != "get_attr"
        ):
            return False

        if not len(weight_node.users) == 1:
            return False

        weight_meta_value = weight_node.meta.get("val")
        if weight_meta_value is None:
            return False
        # Avoid fusing op that causes type promotion
        # restricting to float avoids int/float difficulties with scalar overload
        if not weight_meta_value.is_floating_point():
            return False
        if isinstance(other, torch.fx.Node) and other.op == "get_attr":
            other_meta_value = other.meta.get("val")
            if not other_meta_value.is_floating_point():  # type: ignore[union-attr]
                return False
            if (
                torch.promote_types(other_meta_value.dtype, weight_meta_value.dtype)  # type: ignore[union-attr]
                != weight_meta_value.dtype
            ):
                if not linear_node.meta.get("_allow_mixed_dtype_folding", False):
                    return False

                if (
                    other_meta_value.dtype != torch.float  # type: ignore[union-attr]
                    and weight_meta_value.dtype not in (torch.float16, torch.bfloat16)
                ):
                    return False

            if not _op_not_broadcasting_with_linear(
                weight_meta_value, other_meta_value, has_reshape
            ):
                return False
        elif not isinstance(other, float):
            return False

        return True

    def _is_foldable_pattern(match):
        binary_node = match.output_node()
        has_reshape = False
        if binary_node.args[0].target in _computation_ops:
            computation_node = binary_node.args[0]
            other = binary_node.args[1]
        elif binary_node.args[0].target == aten.reshape.default:
            computation_node = binary_node.args[0].args[0]
            other = binary_node.args[1]
            has_reshape = True
        elif binary_node.args[1].target in _computation_ops:
            computation_node = binary_node.args[1]
            other = binary_node.args[0]
        else:
            computation_node = binary_node.args[1].args[0]
            other = binary_node.args[0]
            has_reshape = False
        if computation_node.target == aten.convolution.default:
            return _check_conv_and_broadcast_op(computation_node, other)
        elif computation_node.target in [aten.addmm.default, aten.mm.default]:
            return (
                config.enable_linear_binary_folding
                and _check_linear_and_broadcast_op(computation_node, other, has_reshape)
            )

        return False

    def resize_scalar_or_tensor_to_shape(graph, other, shape, weight):
        if isinstance(other, float):
            with torch.utils._python_dispatch._disable_current_modes():
                other_tensor = torch.tensor(
                    other, dtype=weight.dtype, device=weight.device
                )
            graph.owning_module.register_buffer("other_tensor", other_tensor)
            res = graph.create_node("get_attr", "other_tensor")
            res = graph.create_node(
                "call_function",
                aten.reshape.default,
                (res, (1,)),
            )
            res = graph.create_node(
                "call_function",
                aten.expand.default,
                (res, shape),
            )
        elif other.meta.get("val").numel() == 1:
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
                graph,
                other,
                (weight_meta_value.size(0),),
                weight_meta_value,
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
                graph,
                other,
                tuple(weight_broadcast_shape),
                weight_meta_value,
            )
            new_weight = graph.create_node(
                "call_function", binary_node.target, (conv_args[1], other_reshape1)
            )
            new_weight.meta.update(conv_args[1].meta)
            conv_args[1] = new_weight
            if bias is not None:
                other_reshape = resize_scalar_or_tensor_to_shape(
                    graph,
                    other,
                    (weight_meta_value.size(0),),
                    weight_meta_value,
                )
                new_bias = graph.create_node(
                    "call_function", binary_node.target, (bias, other_reshape)
                )
                new_bias.meta.update(bias.meta)
                conv_args[2] = new_bias
        return graph.create_node("call_function", conv_node.target, tuple(conv_args))

    def _create_new_linear_node(graph, linear_node, binary_node, other):
        assert linear_node.target in [aten.addmm.default, aten.mm.default]
        input_node = (
            linear_node.args[1]
            if linear_node.target is aten.addmm.default
            else linear_node.args[0]
        )
        weight_node = (
            linear_node.args[2]
            if linear_node.target is aten.addmm.default
            else linear_node.args[1]
        )
        bias_node = (
            linear_node.args[0] if linear_node.target is aten.addmm.default else None
        )
        weight_meta_value = weight_node.meta.get("val")
        if binary_node.target in [aten.add.Tensor, aten.sub.Tensor]:
            other_reshape = resize_scalar_or_tensor_to_shape(
                graph,
                other,
                (weight_meta_value.size(1),),
                weight_meta_value,
            )
            new_bias_node = graph.create_node(
                "call_function",
                binary_node.target,
                (0 if bias_node is None else bias_node, other_reshape),
            )
            return graph.create_node(
                "call_function",
                aten.addmm.default,
                (new_bias_node, input_node, weight_node),
            )
        else:
            assert binary_node.target in [aten.mul.Tensor, aten.div.Tensor]
            weight_broadcast_shape = [1, weight_meta_value.size(1)]
            other_reshape1 = resize_scalar_or_tensor_to_shape(
                graph,
                other,
                tuple(weight_broadcast_shape),
                weight_meta_value,
            )
            new_weight_node = graph.create_node(
                "call_function", binary_node.target, (weight_node, other_reshape1)
            )
            new_weight_node.meta.update(weight_node.meta)
            if bias_node is not None:
                other_reshape = resize_scalar_or_tensor_to_shape(
                    graph,
                    other,
                    (weight_meta_value.size(1),),
                    weight_meta_value,
                )
                new_bias_node = graph.create_node(
                    "call_function", binary_node.target, (bias_node, other_reshape)
                )
                new_bias_node.meta.update(bias_node.meta)
                return graph.create_node(
                    "call_function",
                    linear_node.target,
                    (new_bias_node, input_node, new_weight_node),
                )
            else:
                return graph.create_node(
                    "call_function", linear_node.target, (input_node, new_weight_node)
                )

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
            reshape_node = None
            if binary_node.args[0].target in _computation_ops:
                computation_node = binary_node.args[0]
            elif binary_node.args[0].target == aten.reshape.default:
                computation_node = binary_node.args[0].args[0]
                reshape_node = binary_node.args[0]
            elif binary_node.args[1].target in _computation_ops:
                computation_node = binary_node.args[1]
            else:
                computation_node = binary_node.args[1].args[0]
                reshape_node = binary_node.args[1]
            graph = match.graph
            with graph.inserting_before(reshape_node if reshape_node else binary_node):
                assert computation_node.target in _computation_ops
                if computation_node.target == aten.convolution.default:
                    counters["inductor"]["binary_folding_conv"] += 1
                    new_computation_node = _create_new_conv_node(
                        graph, computation_node, binary_node, other
                    )
                else:
                    new_computation_node = _create_new_linear_node(
                        graph, computation_node, binary_node, other
                    )
                new_computation_node.meta.update(computation_node.meta)
                if reshape_node:
                    assert reshape_node.target == aten.reshape.default
                    computation_node.replace_all_uses_with(new_computation_node)
                    binary_node.replace_all_uses_with(reshape_node)
                else:
                    binary_node.replace_all_uses_with(new_computation_node)
                graph.erase_node(binary_node)
                graph.erase_node(computation_node)
