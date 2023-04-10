import torch

from .. import ir

from ..lowering import lowerings as L
from ..pattern_matcher import (
    Arg,
    CallFunction,
    filter_nodes,
    KeywordArg,
    register_lowering_pattern,
)

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
        if isinstance(min_value, ir.TensorBox) or isinstance(max_value, ir.TensorBox):
            matched = False
        else:  # inp is a Number
            matched = True
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
            return L[aten.clamp_max](L[aten.clamp_min](conv_out, min_value), max_value)

    return fn


def _register_unary_fusion():
    class UnaryAttr:
        def __init__(self, op_name: str, scalars_attr=None, algorithm_attr=None):
            self.op_name = op_name
            self.scalars_attr = scalars_attr if scalars_attr else []
            self.algorithm_attr = algorithm_attr if algorithm_attr else ""

    replacement_unary_fusion_patterns = {
        UnaryAttr("gelu", algorithm_attr="tanh"): [
            _gelu_fusion_2(u) for u in _computation_user_4
        ],
        UnaryAttr("gelu", algorithm_attr="none"): [
            _gelu_fusion_1(u) for u in _computation_user_2
        ],
        UnaryAttr("hardswish"): [_hardswish_fusion(u) for u in _computation_user_2],
        UnaryAttr("hardsigmoid"): [_hardsigmoid_fusion(u) for u in _computation_user_1],
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
    computation_ops = [
        mkldnn._convolution_pointwise.default,
        mkldnn._linear_pointwise.default,
        mkldnn._convolution_transpose_pointwise.default,
    ]

    for unary_attr, patterns in replacement_unary_fusion_patterns.items():
        _register_unary_fusion_lowering(patterns[0], unary_attr, computation_ops[0])
        _register_unary_fusion_lowering(patterns[1], unary_attr, computation_ops[1])
        _register_unary_fusion_lowering(patterns[2], unary_attr, computation_ops[2])

    _leaky_relu_patterns = [_leaky_relu_fusion(user) for user in _computation_user_3]
    _hardtanh_patterns = [_hardtanh_fusion(user) for user in _computation_user_1]
    for pattern, computation_op in zip(_leaky_relu_patterns, computation_ops):
        _register_leaky_relu_fusion_lowering(pattern, computation_op)
    for pattern, computation_op in zip(_hardtanh_patterns, computation_ops):
        _register_hardtanh_fusion_lowering(pattern, computation_op)


def _mkldnn_fusion_init():
    if torch._C.has_mkldnn:
        _register_unary_fusion()
