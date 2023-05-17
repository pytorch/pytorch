import functools

import torch

from .. import ir

from ..lowering import lowerings as L
from ..pattern_matcher import Arg, CallFunction, filter_nodes, get_arg_value, KeywordArg
from ..virtualized import ops
from .post_grad import register_lowering_pattern


if torch._C.has_mkldnn:
    aten = torch.ops.aten
    mkldnn = torch.ops.mkldnn
    _conv_args = (Arg(), Arg(), Arg(), Arg(), Arg(), Arg(), Arg(), Arg(), Arg(), Arg())
    _linear_args = (Arg(), Arg(), Arg(), Arg(), Arg(), Arg())
    _conv_transpose_args = (
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        Arg(),
    )
    _computation_user_1 = [
        CallFunction(mkldnn._convolution_pointwise.default, *_conv_args, _users=1),
        CallFunction(mkldnn._linear_pointwise.default, *_linear_args, _users=1),
        CallFunction(
            mkldnn._convolution_transpose_pointwise.default,
            *_conv_transpose_args,
            _users=1,
        ),
    ]
    _computation_user_2 = [
        CallFunction(mkldnn._convolution_pointwise.default, *_conv_args, _users=2),
        CallFunction(mkldnn._linear_pointwise.default, *_linear_args, _users=2),
        CallFunction(
            mkldnn._convolution_transpose_pointwise.default,
            *_conv_transpose_args,
            _users=2,
        ),
    ]
    _computation_user_3 = [
        CallFunction(mkldnn._convolution_pointwise.default, *_conv_args, _users=3),
        CallFunction(mkldnn._linear_pointwise.default, *_linear_args, _users=3),
        CallFunction(
            mkldnn._convolution_transpose_pointwise.default,
            *_conv_transpose_args,
            _users=3,
        ),
    ]
    _computation_user_4 = [
        CallFunction(mkldnn._convolution_pointwise.default, *_conv_args, _users=4),
        CallFunction(mkldnn._linear_pointwise.default, *_linear_args, _users=4),
        CallFunction(
            mkldnn._convolution_transpose_pointwise.default,
            *_conv_transpose_args,
            _users=4,
        ),
    ]

    def _gelu_fusion_1(computation_call):
        return CallFunction(
            aten.mul,
            CallFunction(aten.mul, computation_call, 0.5),
            CallFunction(
                aten.add,
                CallFunction(
                    aten.erf,
                    CallFunction(aten.mul, computation_call, 0.7071067811865476),
                ),
                1,
            ),
        )

    def _gelu_fusion_2(computation_call):
        return CallFunction(
            aten.mul,
            CallFunction(aten.mul, computation_call, 0.5),
            CallFunction(
                aten.add,
                CallFunction(
                    aten.tanh,
                    CallFunction(
                        aten.mul,
                        CallFunction(
                            aten.add,
                            computation_call,
                            CallFunction(
                                aten.mul,
                                CallFunction(
                                    aten.mul,
                                    CallFunction(
                                        aten.mul, computation_call, computation_call
                                    ),
                                    computation_call,
                                ),
                                0.044715,
                            ),
                        ),
                        0.7978845608028654,
                    ),
                ),
                1,
            ),
        )

    def _hardswish_fusion(computation_call):
        return CallFunction(
            aten.div,
            CallFunction(
                aten.mul,
                computation_call,
                CallFunction(
                    aten.clamp_max,
                    CallFunction(
                        aten.clamp_min, CallFunction(aten.add, computation_call, 3), 0
                    ),
                    6,
                ),
            ),
            6,
        )

    def _silu_fusion(computation_call):
        return CallFunction(
            aten.mul, computation_call, CallFunction(aten.sigmoid, computation_call)
        )

    def _hardsigmoid_fusion(computation_call):
        return CallFunction(
            aten.div,
            CallFunction(
                aten.clamp_max,
                CallFunction(
                    aten.clamp_min, CallFunction(aten.add, computation_call, 3), 0
                ),
                6,
            ),
            6,
        )

    def _leaky_relu_fusion(computation_call):
        return CallFunction(
            aten.where,
            CallFunction(aten.gt, computation_call, 0),
            computation_call,
            CallFunction(aten.mul, computation_call, KeywordArg("negative_slope")),
        )

    def _hardtanh_fusion(computation_call):
        return CallFunction(
            aten.clamp_max,
            CallFunction(aten.clamp_min, computation_call, KeywordArg("min_value")),
            KeywordArg("max_value"),
        )

    def _combined_fusion(computation_call, elementwise_op):
        return CallFunction(elementwise_op, computation_call)

    # binary_op(other, computation_op)
    def _binary_fusion_v1(computation_call, binary_fn):
        return CallFunction(binary_fn, KeywordArg("other"), computation_call)

    # binary_op(computation_op, other)
    def _binary_fusion_v2(computation_call, binary_fn):
        return CallFunction(binary_fn, computation_call, KeywordArg("other"))

    def _is_single_computation_op(computation_op):
        def fn(match):
            computation_nodes = filter_nodes(match.nodes, computation_op)
            if len(computation_nodes) < 1:
                return False
            if any(n.args[-3] != "none" for n in computation_nodes):
                return False
            return True

        return fn

    def _register_unary_fusion_lowering(pattern, unary_attr, computation_op):
        @register_lowering_pattern(
            pattern, extra_check=_is_single_computation_op(computation_op)
        )
        def fn(match, *args):
            computation_args = list(args)[:-3] + [
                unary_attr.op_name,
                unary_attr.scalars_attr,
                unary_attr.algorithm_attr,
            ]
            return L[computation_op](*computation_args)

        return fn

    def _register_leaky_relu_fusion_lowering(pattern, computation_op):
        @register_lowering_pattern(
            pattern, extra_check=_is_single_computation_op(computation_op)
        )
        def fn(match, *args, **kwargs):
            negative_slope = kwargs.get("negative_slope")
            if isinstance(negative_slope, ir.TensorBox):
                matched = False
            else:  # inp is a Number
                matched = True
            computation_args = list(args)
            if matched:
                computation_args = computation_args[:-3] + [
                    "leaky_relu",
                    [negative_slope],
                    "",
                ]
                return L[computation_op](*computation_args)
            else:
                # computation_args += ["none", [], ""]
                computation_out = L[computation_op](*computation_args)
                return L[aten.where](
                    L[aten.gt](computation_out, 0),
                    computation_out,
                    L[aten.mul](computation_out, negative_slope),
                )

        return fn

    def _register_hardtanh_fusion_lowering(pattern, computation_op):
        @register_lowering_pattern(
            pattern, extra_check=_is_single_computation_op(computation_op)
        )
        def fn(match, *args, **kwargs):
            min_value = kwargs.get("min_value")
            max_value = kwargs.get("max_value")
            if isinstance(min_value, ir.TensorBox) or isinstance(
                max_value, ir.TensorBox
            ):
                matched = False
            else:  # inp is a Number
                matched = min_value <= max_value
            computation_args = list(args)
            if matched:
                computation_args = computation_args[:-3] + [
                    "hardtanh",
                    [min_value, max_value],
                    "",
                ]
                return L[computation_op](*computation_args)
            else:
                conv_out = L[computation_op](*computation_args)
                return L[aten.clamp_max](
                    L[aten.clamp_min](conv_out, min_value), max_value
                )

        return fn

    _binary_attr = {
        aten.add: "add",
        ops.add: "add",
        aten.sub: "sub",
        ops.sub: "sub",
    }

    def _is_valid_binary(match, fn):
        binary_nodes = filter_nodes(match.nodes, fn)
        if len(binary_nodes) < 1:
            return False
        if any(
            not (
                hasattr(n.args[0], "meta")
                and isinstance(n.args[0].meta.get("val", None), torch.Tensor)
            )
            or not (
                hasattr(n.args[1], "meta")
                and isinstance(n.args[1].meta.get("val", None), torch.Tensor)
            )
            for n in binary_nodes
        ):
            return False
        # check alpha is one.
        if any(
            get_arg_value(n, 2, kwarg_name="alpha") != 1.0
            and get_arg_value(n, 2, kwarg_name="alpha") is not None
            for n in binary_nodes
        ):
            return False
        if any(
            n.args[0].meta["val"].size() != n.args[1].meta["val"].size()
            or n.args[0].meta["val"].device != n.args[1].meta["val"].device
            or n.args[0].meta["val"].dtype != n.args[1].meta["val"].dtype
            for n in binary_nodes
        ):
            return False
        # check args[0] and args[1] is not same
        if any(n.args[0] == n.args[1] for n in binary_nodes):
            return False
        return True

    def _is_valid_computation_binary(computation_op, binary_op, other_index=None):
        def fn(match):
            if not _is_single_computation_op(computation_op)(match):
                return False
            if not _is_valid_binary(match, binary_op):
                return False
            return True

        return fn

    def _is_valid_computation_binary_inplace(computation_op, binary_op, other_index):
        def fn(match):
            if not _is_valid_computation_binary(computation_op, binary_op)(match):
                return False
            binary_nodes = filter_nodes(match.nodes, binary_op)
            if any(len(n.args[other_index].users) > 1 for n in binary_nodes):
                return False
            if any(
                n.args[other_index].op in ["placeholder", "output"]
                for n in binary_nodes
            ):
                return False
            return True

        return fn

    def _register_binary_unary_fusion_lowering(
        pattern,
        computation_op,
        binary_op,
        fusion_op,
        unary_attr=None,
    ):
        @register_lowering_pattern(
            pattern, extra_check=_is_valid_computation_binary(computation_op, binary_op)
        )
        def fn(match, *args, **kwargs):
            other = kwargs.get("other")
            assert isinstance(other, ir.TensorBox)
            binary_attr = _binary_attr[binary_op]
            args_list = list(args)
            computation_args = [args_list[0], other] + args_list[1:-3] + [binary_attr]
            if len(args_list) > 6:
                if unary_attr is not None:
                    computation_args += [
                        1.0,
                        unary_attr.op_name,
                        unary_attr.scalars_attr,
                        unary_attr.algorithm_attr,
                    ]
                else:
                    computation_args += [1.0, None, [], None]
            return L[fusion_op](*computation_args)

        return fn

    def _register_binary_unary_maybe_inplace_fusion_lowering(
        pattern,
        computation_op,
        binary_op,
        inplace_fusion_op,
        outplace_fusion_op,
        unary_attr=None,
        other_index=None,
    ):
        @register_lowering_pattern(
            pattern,
            extra_check=_is_valid_computation_binary_inplace(
                computation_op, binary_op, other_index
            ),
        )
        def fn(match, *args, **kwargs):
            other = kwargs.get("other")
            assert isinstance(other, ir.TensorBox)
            binary_attr = _binary_attr[binary_op]
            args_list = list(args)
            computation_args = [args_list[0], other] + args_list[1:-3] + [binary_attr]
            if len(args_list) > 6:
                if unary_attr is not None:
                    computation_args += [
                        1.0,
                        unary_attr.op_name,
                        unary_attr.scalars_attr,
                        unary_attr.algorithm_attr,
                    ]
                else:
                    computation_args += [1.0, None, [], None]
            # Make sure the other is not an alias or mutation(fx side doesn't has such info).
            other.realize()
            can_be_inplace = not (
                isinstance(other.data, ir.ReinterpretView)
                or isinstance(other.get_layout(), (ir.MutationLayout, ir.AliasedLayout))
            )
            if not can_be_inplace:
                return L[outplace_fusion_op](*computation_args)
            return L[inplace_fusion_op](*computation_args)

        return fn

    computation_ops = [
        mkldnn._convolution_pointwise.default,
        mkldnn._linear_pointwise.default,
        mkldnn._convolution_transpose_pointwise.default,
    ]

    class UnaryAttr:
        def __init__(self, op_name: str, scalars_attr=None, algorithm_attr=None):
            self.op_name = op_name
            self.scalars_attr = scalars_attr if scalars_attr else []
            self.algorithm_attr = algorithm_attr if algorithm_attr else ""

    def _register_unary_fusion():
        replacement_unary_fusion_patterns = {
            UnaryAttr("gelu", algorithm_attr="tanh"): [
                _gelu_fusion_2(u) for u in _computation_user_4
            ],
            UnaryAttr("gelu", algorithm_attr="none"): [
                _gelu_fusion_1(u) for u in _computation_user_2
            ],
            UnaryAttr("hardswish"): [_hardswish_fusion(u) for u in _computation_user_2],
            UnaryAttr("hardsigmoid"): [
                _hardsigmoid_fusion(u) for u in _computation_user_1
            ],
            UnaryAttr("swish"): [_silu_fusion(u) for u in _computation_user_2],
            UnaryAttr("relu"): [
                _combined_fusion(u, aten.relu) for u in _computation_user_1
            ],
            UnaryAttr("sigmoid"): [
                _combined_fusion(u, aten.sigmoid) for u in _computation_user_1
            ],
            UnaryAttr("tanh"): [
                _combined_fusion(u, aten.tanh) for u in _computation_user_1
            ],
        }
        for unary_attr, patterns in replacement_unary_fusion_patterns.items():
            _register_unary_fusion_lowering(patterns[0], unary_attr, computation_ops[0])
            _register_unary_fusion_lowering(patterns[1], unary_attr, computation_ops[1])
            _register_unary_fusion_lowering(patterns[2], unary_attr, computation_ops[2])

        _leaky_relu_patterns = [
            _leaky_relu_fusion(user) for user in _computation_user_3
        ]
        _hardtanh_patterns = [_hardtanh_fusion(user) for user in _computation_user_1]
        for pattern, computation_op in zip(_leaky_relu_patterns, computation_ops):
            _register_leaky_relu_fusion_lowering(pattern, computation_op)
        for pattern, computation_op in zip(_hardtanh_patterns, computation_ops):
            _register_hardtanh_fusion_lowering(pattern, computation_op)

    def _register_inplace_fusion():
        binary_ops = [aten.add, ops.add]
        inplace_fusion_op = mkldnn._convolution_pointwise_.binary
        outplace_fusion_op = mkldnn._convolution_pointwise.binary
        computation_call = _computation_user_1[0]
        computation_op = computation_ops[0]
        for binary_op in binary_ops:
            binary_v1 = _binary_fusion_v1(computation_call, binary_op)
            binary_unary_v1 = _combined_fusion(binary_v1, aten.relu)
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_unary_v1,
                computation_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=0,
                unary_attr=UnaryAttr("relu"),
            )
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_v1,
                computation_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=0,
            )
            binary_v2 = _binary_fusion_v2(computation_call, binary_op)
            binary_unary_v2 = _combined_fusion(binary_v2, aten.relu)
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_unary_v2,
                computation_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=1,
                unary_attr=UnaryAttr("relu"),
            )
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_v2,
                computation_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=1,
            )

    def _register_binary_fusion():
        binary_ops = [aten.add, ops.add, aten.sub, ops.sub]
        fusion_ops = [
            mkldnn._convolution_pointwise.binary,
            mkldnn._linear_pointwise.binary,
        ]
        for computation_call, computation_op, fusion_op in zip(
            _computation_user_1[:-1], computation_ops[:-1], fusion_ops
        ):
            for binary_op in binary_ops:
                pattern = _binary_fusion_v2(computation_call, binary_op)
                _register_binary_unary_fusion_lowering(
                    pattern, computation_op, binary_op, fusion_op
                )

            for binary_op in [aten.add, ops.add]:
                pattern = _binary_fusion_v1(computation_call, binary_op)
                _register_binary_unary_fusion_lowering(
                    pattern, computation_op, binary_op, fusion_op
                )

    def _register_binary_unary_fusion():
        binary_ops = [aten.add, ops.add, aten.sub, ops.sub]
        fusion_ops = [mkldnn._convolution_pointwise.binary]
        for computation_call, computation_op, fusion_op in zip(
            _computation_user_1[:-1], computation_ops[:-1], fusion_ops
        ):
            for binary_op in binary_ops:
                pattern_v1 = _combined_fusion(
                    _binary_fusion_v2(computation_call, binary_op), aten.relu
                )
                _register_binary_unary_fusion_lowering(
                    pattern_v1,
                    computation_op,
                    binary_op,
                    fusion_op,
                    unary_attr=UnaryAttr("relu"),
                )
            for binary_op in [aten.add, ops.add]:
                pattern_v2 = _combined_fusion(
                    _binary_fusion_v1(computation_call, binary_op), aten.relu
                )
                _register_binary_unary_fusion_lowering(
                    pattern_v2,
                    computation_op,
                    binary_op,
                    fusion_op,
                    unary_attr=UnaryAttr("relu"),
                )

    @functools.lru_cache(None)
    def _mkldnn_fusion_init():
        if torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available():
            _register_unary_fusion()
            _register_inplace_fusion()
            _register_binary_unary_fusion()
            _register_binary_fusion()
