import functools
from typing import Any, Callable

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
    stable_topological_sort,
)
from torch.fx.graph_module import GraphModule

from .mkldnn_fusion import _gelu_fusion_1, _gelu_fusion_2, _silu_fusion


pass_pattern = PatternMatcherPass()
aten = torch.ops.aten
prims = torch.ops.prims


# helper function to create a generic pattern with linear call
# supports prims and bias as well
def create_pattern(
    compute_fn: CallFunction,
    unary_fusion: Callable[[CallFunction], CallFunction],
    with_prims: bool = False,
    users: int = 1,
) -> CallFunction:
    if with_prims:
        # linear->f32->unary->bf16
        upd_compute_fn = CallFunction(
            prims.convert_element_type,
            compute_fn,
            torch.float32,
            _users=users,
        )
        unary_op = unary_fusion(upd_compute_fn)
        # return node will always have a single user
        # we only support bf16 for now
        final_call_fn = CallFunction(
            prims.convert_element_type,
            unary_op,
            torch.bfloat16,
            _users=1,
        )
    else:
        final_call_fn = unary_fusion(compute_fn)
    return final_call_fn


# creates the linear_compute_fn based on args and kwargs
def create_linear_compute_fn(bias: bool = False, users: int = 1) -> CallFunction:
    linear_args = [Arg(), Arg()]  # no bias by default
    if bias:
        linear_args.append(Arg())
    return CallFunction(
        aten.zendnn_linear_unary,
        *linear_args,
        is_weight_prepacked=KeywordArg("is_weight_prepacked"),
        _users=users,
    )


# define a few other unary fusions
def _relu_fusion(compute_fn: CallFunction) -> CallFunction:
    return CallFunction(aten.relu, compute_fn)


def _silu_fusion_no_decomp(compute_fn: CallFunction) -> CallFunction:
    return CallFunction(aten.silu, compute_fn)


def _tanh_fusion(compute_fn: CallFunction) -> CallFunction:
    return CallFunction(aten.tanh, compute_fn)


def _sigmoid_fusion(compute_fn: CallFunction) -> CallFunction:
    return CallFunction(aten.sigmoid, compute_fn)


# this is the non-decomposed version of gelu fusion
def _gelu_fusion(compute_fn: CallFunction) -> CallFunction:
    return CallFunction(
        aten.gelu,
        compute_fn,
        approximate=KeywordArg("approximate"),
    )


# dummy extra check
def dummy_extra_check(match: Match) -> bool:
    return True


# gelu-erf extra check
def gelu_erf_check(match: Match) -> bool:
    if (
        match.kwargs.get("approximate") == "none"
        or match.kwargs.get("approximate") is None
    ):
        return True
    return False


# gelu-tanh extra check
def gelu_tanh_check(match: Match) -> bool:
    if match.kwargs.get("approximate") == "tanh":
        return True
    return False


# we need a generic registration function
def register_patterns(
    post_op_name: str,
    pattern: CallFunction,
    bias: bool,
    extra_check: Callable[[Any], bool] = dummy_extra_check,
) -> None:
    if bias:

        @register_graph_pattern(
            pattern, pass_dict=pass_pattern, extra_check=extra_check
        )
        def replacement_fn(
            match: Match, mat_1: Any, mat_2: Any, bias: Any, *, is_weight_prepacked: Any
        ) -> None:
            def repl(
                mat_1: Any, mat_2: Any, bias: Any, is_weight_prepacked: Any
            ) -> torch.Tensor:
                counters["zendnn"]["zendnn_linear_" + post_op_name] += 1
                return aten.zendnn_linear_unary(
                    mat_1,
                    mat_2,
                    bias,
                    is_weight_prepacked=is_weight_prepacked,
                    post_op=post_op_name,
                )

            match.replace_by_example(repl, [mat_1, mat_2, bias, is_weight_prepacked])
    else:

        @register_graph_pattern(
            pattern, pass_dict=pass_pattern, extra_check=extra_check
        )
        def replacement_fn(
            match: Match, mat_1: Any, mat_2: Any, *, is_weight_prepacked: Any
        ) -> None:
            def repl(mat_1: Any, mat_2: Any, is_weight_prepacked: Any) -> torch.Tensor:
                counters["zendnn"]["zendnn_linear_" + post_op_name] += 1
                return aten.zendnn_linear_unary(
                    mat_1,
                    mat_2,
                    is_weight_prepacked=is_weight_prepacked,
                    post_op=post_op_name,
                )

            match.replace_by_example(repl, [mat_1, mat_2, is_weight_prepacked])


# create a map to pass to unary_fusions_generator
fusions_mapper = {
    "gelu_erf": (_gelu_fusion_1, 2),
    "gelu_tanh": (_gelu_fusion_2, 4),
    "relu": (_relu_fusion, 1),
    "sigmoid": (_sigmoid_fusion, 1),
    "silu": (_silu_fusion, 2),
    "tanh": (_tanh_fusion, 1),
    "silu-no-decomp": (_silu_fusion_no_decomp, 1),
    "gelu-no-decomp": (_gelu_fusion, 1),
}


# function which creates and registers the patterns
@functools.cache
def register_unary_fusions() -> None:
    for bias in [True, False]:
        for post_op, (fusion, users) in fusions_mapper.items():
            if post_op in ("gelu_erf", "gelu_tanh", "silu"):
                # we will create and register prims patterns with these as well since
                # convert_element nodes appear in the graph for lower precision (bf16)
                prims_pattern = create_pattern(
                    create_linear_compute_fn(bias=bias),
                    fusion,
                    with_prims=True,
                    users=users,
                )
                register_patterns(post_op, prims_pattern, bias)
            pattern = create_pattern(
                create_linear_compute_fn(bias=bias, users=users), fusion, users=users
            )
            # if post-op has -no-decomp string we will remove that
            # additionally if gelu is present, we will add the extra_check as well
            # and register twice for erf and tanh
            if "-no-decomp" in post_op:
                post_op = post_op.removesuffix("-no-decomp")
                if post_op == "gelu":
                    register_patterns(
                        "gelu_erf", pattern, bias, extra_check=gelu_erf_check
                    )
                    register_patterns(
                        "gelu_tanh", pattern, bias, extra_check=gelu_tanh_check
                    )
            register_patterns(post_op, pattern, bias)


def zendnn_unary_post_op_fusions(gm: GraphModule) -> GraphModule:
    # call register first
    register_unary_fusions()  # type: ignore[arg-type]
    GraphTransformObserver = functools.partial(
        torch.fx.passes.graph_transform_observer.GraphTransformObserver,
        subsystem="zendnn_unary_post_op_fusions",
    )
    if config.pattern_matcher:
        GraphTransformObserver(gm, "pass_pattern").apply_graph_pass(pass_pattern.apply)
    stable_topological_sort(gm.graph)
    gm.graph.lint()
    gm.recompile()
    return gm
