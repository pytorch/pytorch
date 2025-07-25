# mypy: allow-untyped-defs
import functools

import torch
from torch._inductor import config
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)

from .zendnn_utils import counters


pass_pattern = PatternMatcherPass()
at_ops = torch.ops.aten


# linear-relu
@register_graph_pattern(
    CallFunction(
        at_ops.relu,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.relu_,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
    ),
    pass_dict=pass_pattern,
)
def linear_relu_replacement(match, mat_1, mat_2, bias, *, is_weight_prepacked):
    def repl(mat_1, mat_2, bias, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_relu"] += 1
        return at_ops.zendnn_linear(
            mat_1, mat_2, bias, is_weight_prepacked=is_weight_prepacked, post_op="relu"
        )

    match.replace_by_example(repl, [mat_1, mat_2, bias, is_weight_prepacked])


@register_graph_pattern(
    CallFunction(
        at_ops.relu,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.relu_,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
    ),
    pass_dict=pass_pattern,
)
def linear_relu_replacement_no_bias(match, mat_1, mat_2, *, is_weight_prepacked):
    def repl(mat_1, mat_2, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_relu"] += 1
        return at_ops.zendnn_linear(
            mat_1, mat_2, is_weight_prepacked=is_weight_prepacked, post_op="relu"
        )

    match.replace_by_example(repl, [mat_1, mat_2, is_weight_prepacked])


# linear-silu
@register_graph_pattern(
    CallFunction(
        at_ops.silu,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.silu_,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
    ),
    pass_dict=pass_pattern,
)
def linear_silu_replacement(match, mat_1, mat_2, bias, *, is_weight_prepacked):
    def repl(mat_1, mat_2, bias, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_silu"] += 1
        return at_ops.zendnn_linear(
            mat_1, mat_2, bias, is_weight_prepacked=is_weight_prepacked, post_op="silu"
        )

    match.replace_by_example(repl, [mat_1, mat_2, bias, is_weight_prepacked])


@register_graph_pattern(
    CallFunction(
        at_ops.silu,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.silu_,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
    ),
    pass_dict=pass_pattern,
)
def linear_silu_replacement_no_bias(match, mat_1, mat_2, *, is_weight_prepacked):
    def repl(mat_1, mat_2, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_silu"] += 1
        return at_ops.zendnn_linear(
            mat_1, mat_2, is_weight_prepacked=is_weight_prepacked, post_op="silu"
        )

    match.replace_by_example(repl, [mat_1, mat_2, is_weight_prepacked])


# linear-silu-decomp (isometric patterns not needed for decomposed ops)
zendnn_linear_arg_users_2 = CallFunction(
    at_ops.zendnn_linear,
    Arg(),
    Arg(),
    Arg(),
    is_weight_prepacked=KeywordArg("is_weight_prepacked"),
    _users=2,
)


@register_graph_pattern(
    CallFunction(
        at_ops.mul,
        zendnn_linear_arg_users_2,
        CallFunction(at_ops.sigmoid, zendnn_linear_arg_users_2),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.mul,
        zendnn_linear_arg_users_2,
        CallFunction(at_ops.sigmoid_, zendnn_linear_arg_users_2),
    ),
    pass_dict=pass_pattern,
)
def linear_silu_decomp_replacement(match, mat_1, mat_2, bias, *, is_weight_prepacked):
    def repl(mat_1, mat_2, bias, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_silu"] += 1
        return at_ops.zendnn_linear(
            mat_1, mat_2, bias, is_weight_prepacked=is_weight_prepacked, post_op="silu"
        )

    match.replace_by_example(repl, [mat_1, mat_2, bias, is_weight_prepacked])


zendnn_linear_arg_no_bias_users_2 = CallFunction(
    at_ops.zendnn_linear,
    Arg(),
    Arg(),
    is_weight_prepacked=KeywordArg("is_weight_prepacked"),
    _users=2,
)


@register_graph_pattern(
    CallFunction(
        at_ops.mul,
        zendnn_linear_arg_no_bias_users_2,
        CallFunction(at_ops.sigmoid, zendnn_linear_arg_no_bias_users_2),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.mul,
        zendnn_linear_arg_no_bias_users_2,
        CallFunction(at_ops.sigmoid_, zendnn_linear_arg_no_bias_users_2),
    ),
    pass_dict=pass_pattern,
)
def linear_silu_decomp_replacement_no_bias(match, mat_1, mat_2, *, is_weight_prepacked):
    def repl(mat_1, mat_2, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_silu"] += 1
        return at_ops.zendnn_linear(
            mat_1, mat_2, is_weight_prepacked=is_weight_prepacked, post_op="silu"
        )

    match.replace_by_example(repl, [mat_1, mat_2, is_weight_prepacked])


zendnn_linear_arg_users_2_prims = CallFunction(
    torch.ops.prims.convert_element_type,
    CallFunction(
        at_ops.zendnn_linear,
        Arg(),
        Arg(),
        Arg(),
        is_weight_prepacked=KeywordArg("is_weight_prepacked"),
    ),
    torch.float32,
    _users=2,
)
zendnn_linear_arg_users_2_prims_no_bias = CallFunction(
    torch.ops.prims.convert_element_type,
    CallFunction(
        at_ops.zendnn_linear,
        Arg(),
        Arg(),
        is_weight_prepacked=KeywordArg("is_weight_prepacked"),
    ),
    torch.float32,
    _users=2,
)


@register_graph_pattern(
    CallFunction(
        torch.ops.prims.convert_element_type,
        CallFunction(
            at_ops.mul,
            zendnn_linear_arg_users_2_prims,
            CallFunction(at_ops.sigmoid, zendnn_linear_arg_users_2_prims),
        ),
        torch.bfloat16,
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        torch.ops.prims.convert_element_type,
        CallFunction(
            at_ops.mul,
            zendnn_linear_arg_users_2_prims,
            CallFunction(at_ops.sigmoid, zendnn_linear_arg_users_2_prims),
        ),
        torch.bfloat16,
    ),
    pass_dict=pass_pattern,
)
def linear_silu_decomp_replacement_1(match, mat_1, mat_2, bias, *, is_weight_prepacked):
    def repl(mat_1, mat_2, bias, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_silu"] += 1
        return at_ops.zendnn_linear(
            mat_1, mat_2, bias, is_weight_prepacked=is_weight_prepacked, post_op="silu"
        )

    match.replace_by_example(repl, [mat_1, mat_2, bias, is_weight_prepacked])


@register_graph_pattern(
    CallFunction(
        torch.ops.prims.convert_element_type,
        CallFunction(
            at_ops.mul,
            zendnn_linear_arg_users_2_prims_no_bias,
            CallFunction(at_ops.sigmoid, zendnn_linear_arg_users_2_prims_no_bias),
        ),
        torch.bfloat16,
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        torch.ops.prims.convert_element_type,
        CallFunction(
            at_ops.mul,
            zendnn_linear_arg_users_2_prims_no_bias,
            CallFunction(at_ops.sigmoid, zendnn_linear_arg_users_2_prims_no_bias),
        ),
        torch.bfloat16,
    ),
    pass_dict=pass_pattern,
)
def linear_silu_decomp_replacement_no_bias_1(
    match, mat_1, mat_2, *, is_weight_prepacked
):
    def repl(mat_1, mat_2, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_silu"] += 1
        return at_ops.zendnn_linear(
            mat_1, mat_2, is_weight_prepacked=is_weight_prepacked, post_op="silu"
        )

    match.replace_by_example(repl, [mat_1, mat_2, is_weight_prepacked])


# linear-sigmoid
@register_graph_pattern(
    CallFunction(
        at_ops.sigmoid,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.sigmoid_,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
    ),
    pass_dict=pass_pattern,
)
def linear_sigmoid_replacement(match, mat_1, mat_2, bias, *, is_weight_prepacked):
    def repl(mat_1, mat_2, bias, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_sigmoid"] += 1
        return at_ops.zendnn_linear(
            mat_1,
            mat_2,
            bias,
            is_weight_prepacked=is_weight_prepacked,
            post_op="sigmoid",
        )

    match.replace_by_example(repl, [mat_1, mat_2, bias, is_weight_prepacked])


@register_graph_pattern(
    CallFunction(
        at_ops.sigmoid,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.sigmoid_,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
    ),
    pass_dict=pass_pattern,
)
def linear_sigmoid_replacement_no_bias(match, mat_1, mat_2, *, is_weight_prepacked):
    def repl(mat_1, mat_2, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_sigmoid"] += 1
        return at_ops.zendnn_linear(
            mat_1, mat_2, is_weight_prepacked=is_weight_prepacked, post_op="sigmoid"
        )

    match.replace_by_example(repl, [mat_1, mat_2, is_weight_prepacked])


# linear-tanh
@register_graph_pattern(
    CallFunction(
        at_ops.tanh,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.tanh_,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
    ),
    pass_dict=pass_pattern,
)
def linear_tanh_replacement(match, mat_1, mat_2, bias, *, is_weight_prepacked):
    def repl(mat_1, mat_2, bias, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_tanh"] += 1
        return at_ops.zendnn_linear(
            mat_1, mat_2, bias, is_weight_prepacked=is_weight_prepacked, post_op="tanh"
        )

    match.replace_by_example(repl, [mat_1, mat_2, bias, is_weight_prepacked])


@register_graph_pattern(
    CallFunction(
        at_ops.tanh,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
    ),
    pass_dict=pass_pattern,
)
@register_graph_pattern(
    CallFunction(
        at_ops.tanh_,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
    ),
    pass_dict=pass_pattern,
)
def linear_tanh_replacement_no_bias(match, mat_1, mat_2, *, is_weight_prepacked):
    def repl(mat_1, mat_2, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_tanh"] += 1
        return at_ops.zendnn_linear(
            mat_1, mat_2, is_weight_prepacked=is_weight_prepacked, post_op="tanh"
        )

    match.replace_by_example(repl, [mat_1, mat_2, is_weight_prepacked])


# gelu-erf extra check
# def gelu_erf_check(match):
#     if (
#         match.kwargs.get("approximate") == "none"
#         or match.kwargs.get("approximate") is None
#     ):
#         return True
#     return False


# gelu-tanh extra check
def gelu_tanh_check(match):
    if match.kwargs.get("approximate") == "tanh":
        return True
    return False


# TODO: Enable this once the library has fixed the bug.
# linear-gelu-erf
# @register_graph_pattern(
#     CallFunction(
#         at_ops.gelu,
#         CallFunction(
#             at_ops.zendnn_linear,
#             Arg(),
#             Arg(),
#             Arg(),
#             is_weight_prepacked=KeywordArg("is_weight_prepacked"),
#         ),
#         approximate=KeywordArg("approximate"),
#     ),
#     pass_dict=pass_pattern,
#     extra_check=gelu_erf_check,
# )
# @register_graph_pattern(
#     CallFunction(
#         at_ops.gelu_,
#         CallFunction(
#             at_ops.zendnn_linear,
#             Arg(),
#             Arg(),
#             Arg(),
#             is_weight_prepacked=KeywordArg("is_weight_prepacked"),
#         ),
#         approximate=KeywordArg("approximate"),
#     ),
#     pass_dict=pass_pattern,
#     extra_check=gelu_erf_check,
# )
# def linear_gelu_replacement(
#     match, mat_1, mat_2, bias, *, is_weight_prepacked, approximate
# ):
#     def repl(mat_1, mat_2, bias, is_weight_prepacked, approximate):
#         counters["zendnn"]["zendnn_linear_gelu_erf"] += 1
#         return at_ops.zendnn_linear(
#             mat_1,
#             mat_2,
#             bias,
#             is_weight_prepacked=is_weight_prepacked,
#             post_op="gelu_erf",
#         )
#     match.replace_by_example(
#         repl, [mat_1, mat_2, bias, is_weight_prepacked, approximate]
#     )
# @register_graph_pattern(
#     CallFunction(
#         at_ops.gelu,
#         CallFunction(
#             at_ops.zendnn_linear,
#             Arg(),
#             Arg(),
#             is_weight_prepacked=KeywordArg("is_weight_prepacked"),
#         ),
#         approximate=KeywordArg("approximate"),
#     ),
#     pass_dict=pass_pattern,
#     extra_check=gelu_erf_check,
# )
# @register_graph_pattern(
#     CallFunction(
#         at_ops.gelu_,
#         CallFunction(
#             at_ops.zendnn_linear,
#             Arg(),
#             Arg(),
#             is_weight_prepacked=KeywordArg("is_weight_prepacked"),
#         ),
#         approximate=KeywordArg("approximate"),
#     ),
#     pass_dict=pass_pattern,
#     extra_check=gelu_erf_check,
# )
# def linear_gelu_replacement_no_bias(
#     match, mat_1, mat_2, *, is_weight_prepacked, approximate
# ):
#     def repl(mat_1, mat_2, is_weight_prepacked, approximate):
#         counters["zendnn"]["zendnn_linear_gelu_erf"] += 1
#         return at_ops.zendnn_linear(
#             mat_1, mat_2, is_weight_prepacked=is_weight_prepacked, post_op="gelu_erf"
#         )
#     match.replace_by_example(repl, [mat_1, mat_2, is_weight_prepacked, approximate])
# linear-gelu-erf-decomp
# @register_graph_pattern(
#     CallFunction(
#         at_ops.mul,
#         CallFunction(at_ops.mul, zendnn_linear_arg_users_2, 0.5),
#         CallFunction(
#             at_ops.add,
#             CallFunction(
#                 at_ops.erf,
#                 CallFunction(at_ops.mul, zendnn_linear_arg_users_2, 0.7071067811865476),
#             ),
#             1.0,
#         ),
#     ),
#     pass_dict=pass_pattern,
# )
# def linear_gelu_decomp_replacement(match, mat_1, mat_2, bias, *, is_weight_prepacked):
#     def repl(mat_1, mat_2, bias, is_weight_prepacked):
#         counters["zendnn"]["zendnn_linear_gelu_erf"] += 1
#         return at_ops.zendnn_linear(
#             mat_1,
#             mat_2,
#             bias,
#             is_weight_prepacked=is_weight_prepacked,
#             post_op="gelu_erf",
#         )
#     match.replace_by_example(repl, [mat_1, mat_2, bias, is_weight_prepacked])
# @register_graph_pattern(
#     CallFunction(
#         at_ops.mul,
#         CallFunction(at_ops.mul, zendnn_linear_arg_no_bias_users_2, 0.5),
#         CallFunction(
#             at_ops.add,
#             CallFunction(
#                 at_ops.erf,
#                 CallFunction(
#                     at_ops.mul, zendnn_linear_arg_no_bias_users_2, 0.7071067811865476
#                 ),
#             ),
#             1.0,
#         ),
#     ),
#     pass_dict=pass_pattern,
# )
# def linear_gelu_decomp_replacement_no_bias(match, mat_1, mat_2, *, is_weight_prepacked):
#     def repl(mat_1, mat_2, is_weight_prepacked):
#         counters["zendnn"]["zendnn_linear_gelu_erf"] += 1
#         return at_ops.zendnn_linear(
#             mat_1, mat_2, is_weight_prepacked=is_weight_prepacked, post_op="gelu_erf"
#         )
#     match.replace_by_example(repl, [mat_1, mat_2, is_weight_prepacked])
# @register_graph_pattern(
#     CallFunction(
#         torch.ops.prims.convert_element_type,
#         CallFunction(
#             at_ops.mul,
#             CallFunction(at_ops.mul, zendnn_linear_arg_users_2_prims, 0.5),
#             CallFunction(
#                 at_ops.add,
#                 CallFunction(
#                     at_ops.erf,
#                     CallFunction(
#                         at_ops.mul, zendnn_linear_arg_users_2_prims, 0.7071067811865476
#                     ),
#                 ),
#                 1.0,
#             ),
#         ),
#         torch.bfloat16,
#     ),
#     pass_dict=pass_pattern,
# )
# def linear_gelu_decomp_replacement_1(match, mat_1, mat_2, bias, *, is_weight_prepacked):
#     def repl(mat_1, mat_2, bias, is_weight_prepacked):
#         counters["zendnn"]["zendnn_linear_gelu_erf"] += 1
#         return at_ops.zendnn_linear(
#             mat_1,
#             mat_2,
#             bias,
#             is_weight_prepacked=is_weight_prepacked,
#             post_op="gelu_erf",
#         )
#     match.replace_by_example(repl, [mat_1, mat_2, bias, is_weight_prepacked])
# @register_graph_pattern(
#     CallFunction(
#         torch.ops.prims.convert_element_type,
#         CallFunction(
#             at_ops.mul,
#             CallFunction(at_ops.mul, zendnn_linear_arg_users_2_prims_no_bias, 0.5),
#             CallFunction(
#                 at_ops.add,
#                 CallFunction(
#                     at_ops.erf,
#                     CallFunction(
#                         at_ops.mul,
#                         zendnn_linear_arg_users_2_prims_no_bias,
#                         0.7071067811865476,
#                     ),
#                 ),
#                 1.0,
#             ),
#         ),
#         torch.bfloat16,
#     ),
#     pass_dict=pass_pattern,
# )
# def linear_gelu_decomp_replacement_no_bias_1(
#     match, mat_1, mat_2, *, is_weight_prepacked
# ):
#     def repl(mat_1, mat_2, is_weight_prepacked):
#         counters["zendnn"]["zendnn_linear_gelu_erf"] += 1
#         return at_ops.zendnn_linear(
#             mat_1, mat_2, is_weight_prepacked=is_weight_prepacked, post_op="gelu_erf"
#         )
#     match.replace_by_example(repl, [mat_1, mat_2, is_weight_prepacked])


# linear-gelu-tanh
@register_graph_pattern(
    CallFunction(
        at_ops.gelu,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_tanh_check,
)
@register_graph_pattern(
    CallFunction(
        at_ops.gelu_,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_tanh_check,
)
def linear_gelu_tanh_replacement(
    match, mat_1, mat_2, bias, *, is_weight_prepacked, approximate
):
    def repl(mat_1, mat_2, bias, is_weight_prepacked, approximate):
        counters["zendnn"]["zendnn_linear_gelu_tanh"] += 1
        return at_ops.zendnn_linear(
            mat_1,
            mat_2,
            bias,
            is_weight_prepacked=is_weight_prepacked,
            post_op="gelu_tanh",
        )

    match.replace_by_example(
        repl, [mat_1, mat_2, bias, is_weight_prepacked, approximate]
    )


@register_graph_pattern(
    CallFunction(
        at_ops.gelu,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_tanh_check,
)
@register_graph_pattern(
    CallFunction(
        at_ops.gelu_,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
        approximate=KeywordArg("approximate"),
    ),
    pass_dict=pass_pattern,
    extra_check=gelu_tanh_check,
)
def linear_gelu_tanh_replacement_no_bias(
    match, mat_1, mat_2, *, is_weight_prepacked, approximate
):
    def repl(mat_1, mat_2, is_weight_prepacked, approximate):
        counters["zendnn"]["zendnn_linear_gelu_tanh"] += 1
        return at_ops.zendnn_linear(
            mat_1, mat_2, is_weight_prepacked=is_weight_prepacked, post_op="gelu_tanh"
        )

    match.replace_by_example(repl, [mat_1, mat_2, is_weight_prepacked, approximate])


# linear-gelu-tanh-decomp
zendnn_linear_arg_users_4 = CallFunction(
    at_ops.zendnn_linear,
    Arg(),
    Arg(),
    Arg(),
    is_weight_prepacked=KeywordArg("is_weight_prepacked"),
    _users=4,
)


@register_graph_pattern(
    CallFunction(
        at_ops.mul,
        CallFunction(at_ops.mul, zendnn_linear_arg_users_4, 0.5),
        CallFunction(
            at_ops.add,
            CallFunction(
                at_ops.tanh,
                CallFunction(
                    at_ops.mul,
                    CallFunction(
                        at_ops.add,
                        zendnn_linear_arg_users_4,
                        CallFunction(
                            at_ops.mul,
                            CallFunction(
                                at_ops.mul,
                                CallFunction(
                                    at_ops.mul,
                                    zendnn_linear_arg_users_4,
                                    zendnn_linear_arg_users_4,
                                ),
                                zendnn_linear_arg_users_4,
                            ),
                            0.044715,
                        ),
                    ),
                    0.7978845608028654,
                ),
            ),
            1.0,
        ),
    ),
    pass_dict=pass_pattern,
)
def linear_gelu_tanh_decomp_replacement(
    match, mat_1, mat_2, bias, *, is_weight_prepacked
):
    def repl(mat_1, mat_2, bias, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_gelu_tanh"] += 1
        return at_ops.zendnn_linear(
            mat_1,
            mat_2,
            bias,
            is_weight_prepacked=is_weight_prepacked,
            post_op="gelu_tanh",
        )

    match.replace_by_example(repl, [mat_1, mat_2, bias, is_weight_prepacked])


zendnn_linear_arg_users_4_prims = CallFunction(
    torch.ops.prims.convert_element_type,
    CallFunction(
        at_ops.zendnn_linear,
        Arg(),
        Arg(),
        Arg(),
        is_weight_prepacked=KeywordArg("is_weight_prepacked"),
    ),
    torch.float32,
    _users=4,
)


@register_graph_pattern(
    CallFunction(
        torch.ops.prims.convert_element_type,
        CallFunction(
            at_ops.mul,
            CallFunction(at_ops.mul, zendnn_linear_arg_users_4_prims, 0.5),
            CallFunction(
                at_ops.add,
                CallFunction(
                    at_ops.tanh,
                    CallFunction(
                        at_ops.mul,
                        CallFunction(
                            at_ops.add,
                            zendnn_linear_arg_users_4_prims,
                            CallFunction(
                                at_ops.mul,
                                CallFunction(
                                    at_ops.mul,
                                    CallFunction(
                                        at_ops.mul,
                                        zendnn_linear_arg_users_4_prims,
                                        zendnn_linear_arg_users_4_prims,
                                    ),
                                    zendnn_linear_arg_users_4_prims,
                                ),
                                0.044715,
                            ),
                        ),
                        0.7978845608028654,
                    ),
                ),
                1.0,
            ),
        ),
        torch.bfloat16,
    ),
    pass_dict=pass_pattern,
)
def linear_gelu_tanh_decomp_replacement_1(
    match, mat_1, mat_2, bias, *, is_weight_prepacked
):
    def repl(mat_1, mat_2, bias, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_gelu_tanh"] += 1
        return at_ops.zendnn_linear(
            mat_1,
            mat_2,
            bias,
            is_weight_prepacked=is_weight_prepacked,
            post_op="gelu_tanh",
        )

    match.replace_by_example(repl, [mat_1, mat_2, bias, is_weight_prepacked])


zendnn_linear_arg_users_4_prims_no_bias = CallFunction(
    torch.ops.prims.convert_element_type,
    CallFunction(
        at_ops.zendnn_linear,
        Arg(),
        Arg(),
        is_weight_prepacked=KeywordArg("is_weight_prepacked"),
    ),
    torch.float32,
    _users=4,
)


@register_graph_pattern(
    CallFunction(
        torch.ops.prims.convert_element_type,
        CallFunction(
            at_ops.mul,
            CallFunction(at_ops.mul, zendnn_linear_arg_users_4_prims_no_bias, 0.5),
            CallFunction(
                at_ops.add,
                CallFunction(
                    at_ops.tanh,
                    CallFunction(
                        at_ops.mul,
                        CallFunction(
                            at_ops.add,
                            zendnn_linear_arg_users_4_prims_no_bias,
                            CallFunction(
                                at_ops.mul,
                                CallFunction(
                                    at_ops.mul,
                                    CallFunction(
                                        at_ops.mul,
                                        zendnn_linear_arg_users_4_prims_no_bias,
                                        zendnn_linear_arg_users_4_prims_no_bias,
                                    ),
                                    zendnn_linear_arg_users_4_prims_no_bias,
                                ),
                                0.044715,
                            ),
                        ),
                        0.7978845608028654,
                    ),
                ),
                1.0,
            ),
        ),
        torch.bfloat16,
    ),
    pass_dict=pass_pattern,
)
def linear_gelu_tanh_decomp_replacement_no_bias_1(
    match, mat_1, mat_2, *, is_weight_prepacked
):
    def repl(mat_1, mat_2, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_gelu_tanh"] += 1
        return at_ops.zendnn_linear(
            mat_1,
            mat_2,
            is_weight_prepacked=is_weight_prepacked,
            post_op="gelu_tanh",
        )

    match.replace_by_example(repl, [mat_1, mat_2, is_weight_prepacked])


zendnn_linear_arg_no_bias_users_4 = CallFunction(
    at_ops.zendnn_linear,
    Arg(),
    Arg(),
    is_weight_prepacked=KeywordArg("is_weight_prepacked"),
    _users=4,
)


@register_graph_pattern(
    CallFunction(
        at_ops.mul,
        CallFunction(at_ops.mul, zendnn_linear_arg_no_bias_users_4, 0.5),
        CallFunction(
            at_ops.add,
            CallFunction(
                at_ops.tanh,
                CallFunction(
                    at_ops.mul,
                    CallFunction(
                        at_ops.add,
                        zendnn_linear_arg_no_bias_users_4,
                        CallFunction(
                            at_ops.mul,
                            CallFunction(
                                at_ops.mul,
                                CallFunction(
                                    at_ops.mul,
                                    zendnn_linear_arg_no_bias_users_4,
                                    zendnn_linear_arg_no_bias_users_4,
                                ),
                                zendnn_linear_arg_no_bias_users_4,
                            ),
                            0.044715,
                        ),
                    ),
                    0.7978845608028654,
                ),
            ),
            1.0,
        ),
    ),
    pass_dict=pass_pattern,
)
def linear_gelu_tanh_decomp_replacement_no_bias(
    match, mat_1, mat_2, *, is_weight_prepacked
):
    def repl(mat_1, mat_2, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_gelu_tanh"] += 1
        return at_ops.zendnn_linear(
            mat_1, mat_2, is_weight_prepacked=is_weight_prepacked, post_op="gelu_tanh"
        )

    match.replace_by_example(repl, [mat_1, mat_2, is_weight_prepacked])


# addmm replacement check
def same_dtypes_check(match):
    if isinstance(match.args[0], torch.fx.node.Node):
        ref_dtype = match.args[0].meta["val"].dtype
    else:
        return False
    for arg in match.args:
        if not isinstance(arg, torch.fx.node.Node):
            return False
        elif arg.meta["val"].dtype != ref_dtype:
            return False
    return True


def binary_dim_check(match):
    if (
        isinstance(match.args[0], torch.fx.node.Node)
        and isinstance(match.args[1], torch.fx.node.Node)
        and isinstance(match.args[-1], torch.fx.node.Node)
    ):
        s1 = match.args[0].meta["val"].shape
        new_shape = s1[:-1] + (match.args[1].meta["val"].shape[-1],)
        if match.args[-1].meta["val"].shape != new_shape:
            return False
    return same_dtypes_check(match)


def binary_dim_check_iso(match):
    if (
        isinstance(match.args[0], torch.fx.node.Node)
        and isinstance(match.args[1], torch.fx.node.Node)
        and isinstance(match.args[2], torch.fx.node.Node)
    ):
        s1 = match.args[1].meta["val"].shape
        new_shape = s1[:-1] + (match.args[2].meta["val"].shape[-1],)
        if match.args[0].meta["val"].shape != new_shape:
            return False
    return same_dtypes_check(match)


# linear-add
@register_graph_pattern(
    CallFunction(
        at_ops.add.Tensor,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
        Arg(),
    ),
    pass_dict=pass_pattern,
    extra_check=binary_dim_check,
)
def linear_add_replacement(
    match: Match, mat_1, mat_2, bias, add, *, is_weight_prepacked
):
    def repl(mat_1, mat_2, bias, add, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_add"] += 1
        return at_ops.zendnn_linear_unary_binary(
            mat_1,
            mat_2,
            add,
            bias,
            is_weight_prepacked=is_weight_prepacked,
            post_op_1="",
            post_op_2="add",
        )

    match.replace_by_example(repl, [mat_1, mat_2, bias, add, is_weight_prepacked])


@register_graph_pattern(
    CallFunction(
        at_ops.add.Tensor,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
        Arg(),
    ),
    pass_dict=pass_pattern,
    extra_check=binary_dim_check,
)
def linear_add_replacement_no_bias(
    match: Match, mat_1, mat_2, add, *, is_weight_prepacked
):
    def repl(mat_1, mat_2, add, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_add"] += 1
        return at_ops.zendnn_linear_unary_binary(
            mat_1,
            mat_2,
            add,
            is_weight_prepacked=is_weight_prepacked,
            post_op_1="",
            post_op_2="add",
        )

    match.replace_by_example(repl, [mat_1, mat_2, add, is_weight_prepacked])


# linear-add-iso
@register_graph_pattern(
    CallFunction(
        at_ops.add.Tensor,
        Arg(),
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
    ),
    pass_dict=pass_pattern,
    extra_check=binary_dim_check_iso,
)
def linear_add_replacement_1(
    match: Match, add, mat_1, mat_2, bias, *, is_weight_prepacked
):
    def repl(add, mat_1, mat_2, bias, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_add"] += 1
        return at_ops.zendnn_linear_unary_binary(
            mat_1,
            mat_2,
            add,
            bias,
            is_weight_prepacked=is_weight_prepacked,
            post_op_1="",
            post_op_2="add",
        )

    match.replace_by_example(repl, [add, mat_1, mat_2, bias, is_weight_prepacked])


@register_graph_pattern(
    CallFunction(
        at_ops.add.Tensor,
        Arg(),
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
    ),
    pass_dict=pass_pattern,
    extra_check=binary_dim_check_iso,
)
def linear_add_replacement_no_bias_1(
    match: Match, add, mat_1, mat_2, *, is_weight_prepacked
):
    def repl(add, mat_1, mat_2, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_add"] += 1
        return at_ops.zendnn_linear_unary_binary(
            mat_1,
            mat_2,
            add,
            is_weight_prepacked=is_weight_prepacked,
            post_op_1="",
            post_op_2="add",
        )

    match.replace_by_example(repl, [add, mat_1, mat_2, is_weight_prepacked])


# linear-mul
@register_graph_pattern(
    CallFunction(
        at_ops.mul.Tensor,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
        Arg(),
    ),
    pass_dict=pass_pattern,
    extra_check=binary_dim_check,
)
def linear_mul_replacement(
    match: Match, mat_1, mat_2, bias, mul, *, is_weight_prepacked
):
    def repl(mat_1, mat_2, bias, mul, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_mul"] += 1
        return at_ops.zendnn_linear_unary_binary(
            mat_1,
            mat_2,
            mul,
            bias,
            is_weight_prepacked=is_weight_prepacked,
            post_op_1="",
            post_op_2="mul",
        )

    match.replace_by_example(repl, [mat_1, mat_2, bias, mul, is_weight_prepacked])


@register_graph_pattern(
    CallFunction(
        at_ops.mul.Tensor,
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
        Arg(),
    ),
    pass_dict=pass_pattern,
    extra_check=binary_dim_check,
)
def linear_mul_replacement_no_bias(
    match: Match, mat_1, mat_2, mul, *, is_weight_prepacked
):
    def repl(mat_1, mat_2, mul, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_mul"] += 1
        return at_ops.zendnn_linear_unary_binary(
            mat_1,
            mat_2,
            mul,
            is_weight_prepacked=is_weight_prepacked,
            post_op_1="",
            post_op_2="mul",
        )

    match.replace_by_example(repl, [mat_1, mat_2, mul, is_weight_prepacked])


# linear-mul-iso
@register_graph_pattern(
    CallFunction(
        at_ops.mul.Tensor,
        Arg(),
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
    ),
    pass_dict=pass_pattern,
    extra_check=binary_dim_check_iso,
)
def linear_mul_replacement_1(
    match: Match, mul, mat_1, mat_2, bias, *, is_weight_prepacked
):
    def repl(mul, mat_1, mat_2, bias, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_mul"] += 1
        return at_ops.zendnn_linear_unary_binary(
            mat_1,
            mat_2,
            mul,
            bias,
            is_weight_prepacked=is_weight_prepacked,
            post_op_1="",
            post_op_2="mul",
        )

    match.replace_by_example(repl, [mul, mat_1, mat_2, bias, is_weight_prepacked])


@register_graph_pattern(
    CallFunction(
        at_ops.mul.Tensor,
        Arg(),
        CallFunction(
            at_ops.zendnn_linear,
            Arg(),
            Arg(),
            is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        ),
    ),
    pass_dict=pass_pattern,
    extra_check=binary_dim_check_iso,
)
def linear_mul_replacement_no_bias_1(
    match: Match, mul, mat_1, mat_2, *, is_weight_prepacked
):
    def repl(mul, mat_1, mat_2, is_weight_prepacked):
        counters["zendnn"]["zendnn_linear_mul"] += 1
        return at_ops.zendnn_linear_unary_binary(
            mat_1,
            mat_2,
            mul,
            is_weight_prepacked=is_weight_prepacked,
            post_op_1="",
            post_op_2="mul",
        )

    match.replace_by_example(repl, [mul, mat_1, mat_2, is_weight_prepacked])


def zendnn_single_post_op_fusions(gm):
    GraphTransformObserver = functools.partial(
        torch.fx.passes.graph_transform_observer.GraphTransformObserver,
        subsystem="zendnn_single_post_op_fusions",
    )
    if config.pattern_matcher:
        GraphTransformObserver(gm, "pass_pattern").apply_graph_pass(pass_pattern.apply)
    gm.graph.lint()
    gm.recompile()
    return gm
