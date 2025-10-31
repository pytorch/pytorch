import functools
from typing import Any, Callable

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
    stable_topological_sort,
)
from torch.fx.graph_module import GraphModule

from .zendnn_unary_post_op_fusions import (
    _silu_fusion,
    _silu_fusion_no_decomp,
    create_linear_compute_fn,
    create_pattern,
)


pass_pattern = PatternMatcherPass()
aten = torch.ops.aten


# addmm replacement check
def same_dtypes_check(match: Match) -> bool:
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


def calc_strides(new_shape: list[int]) -> list[int]:
    # calculate stride from new_shape
    last_stride = 1
    new_stride: list[int] = []
    for shape in reversed(new_shape):
        new_stride.insert(0, last_stride)
        last_stride *= shape
    return new_stride


def binary_dim_check(idx: int) -> Callable[[Match], bool]:
    def fn(match: Match) -> bool:
        if isinstance(match.args[idx - 1], torch.fx.node.Node):
            s1 = match.args[idx].meta["val"].shape
            new_shape = s1[:-1] + (match.args[idx + 1].meta["val"].shape[0],)
            if match.args[idx - 1].meta["val"].shape != new_shape:
                return False
            elif calc_strides(new_shape) != list(
                match.args[idx - 1].meta["val"].stride()
            ):
                return False
        return same_dtypes_check(match)

    return fn


# make an isometric patterns generator
def unary_binary_patterns_generator(
    aten_op: Any, unary_fusion: Any
) -> list[CallFunction]:
    # we will return a list of patterns, order of elements is important
    return [
        CallFunction(aten_op, unary_fusion, Arg()),
        CallFunction(aten_op, Arg(), unary_fusion),
    ]


# define registration functions
def register_unary_binary_patterns(
    unary_post_op_name: str,
    binary_post_op_name: str,
    pattern_lst: list[CallFunction],
    bias: bool,
) -> None:
    """Register unary-binary fusion patterns along with replacements."""
    for i in range(len(pattern_lst)):
        # Create a unique function for each pattern
        def create_replacement_fn_unary_binary(
            idx: int, unary_name: str, binary_name: str, has_bias: bool
        ) -> Callable[..., Any]:
            if has_bias:

                def replacement_fn(
                    match: Match,
                    arg_0: Any,
                    arg_1: Any,
                    arg_2: Any,
                    arg_3: Any,
                    *,
                    is_weight_prepacked: Any,
                ) -> None:
                    def repl(
                        arg_0: Any,
                        arg_1: Any,
                        arg_2: Any,
                        arg_3: Any,
                        is_weight_prepacked: Any,
                    ) -> torch.Tensor:
                        if unary_name == "none":
                            counters["zendnn"]["zendnn_linear_" + binary_name] += 1
                            new_unary_name = "none"
                        else:
                            if "_no_decomp" in unary_name:
                                new_unary_name = unary_name.removesuffix("_no_decomp")
                            else:
                                new_unary_name = unary_name
                            counters["zendnn"][
                                "zendnn_linear_" + new_unary_name + "_" + binary_name
                            ] += 1
                        # unwrap DFS arguments here:
                        # when i = 0 -> order is: mat_1, mat_2, bias, binary_op
                        # when i = 1 -> order is: binary_op, mat_1, mat_2, bias
                        if idx == 0:
                            return aten.zendnn_linear_unary_binary(
                                arg_0,
                                arg_1,
                                arg_3,
                                arg_2,
                                is_weight_prepacked=is_weight_prepacked,
                                post_op_1=new_unary_name,
                                post_op_2=binary_name,
                            )
                        else:
                            return aten.zendnn_linear_unary_binary(
                                arg_1,
                                arg_2,
                                arg_0,
                                arg_3,
                                is_weight_prepacked=is_weight_prepacked,
                                post_op_1=new_unary_name,
                                post_op_2=binary_name,
                            )

                    match.replace_by_example(
                        repl, [arg_0, arg_1, arg_2, arg_3, is_weight_prepacked]
                    )

                replacement_fn.__name__ = (
                    f"zendnn_linear_{unary_name}_{binary_name}_replacement_{idx}"
                )
                return replacement_fn
            else:

                def replacement_fn(  # type: ignore[misc]
                    match: Match,
                    arg_0: Any,
                    arg_1: Any,
                    arg_2: Any,
                    *,
                    is_weight_prepacked: Any,
                ) -> None:
                    def repl(
                        arg_0: Any, arg_1: Any, arg_2: Any, is_weight_prepacked: Any
                    ) -> torch.Tensor:
                        if unary_name == "none":
                            counters["zendnn"]["zendnn_linear_" + binary_name] += 1
                            new_unary_name = "none"
                        else:
                            if "_no_decomp" in unary_name:
                                new_unary_name = unary_name.removesuffix("_no_decomp")
                            else:
                                new_unary_name = unary_name
                            counters["zendnn"][
                                "zendnn_linear_" + new_unary_name + "_" + binary_name
                            ] += 1
                        if idx == 0:
                            return aten.zendnn_linear_unary_binary(
                                arg_0,
                                arg_1,
                                arg_2,
                                is_weight_prepacked=is_weight_prepacked,
                                post_op_1=new_unary_name,
                                post_op_2=binary_name,
                            )
                        else:
                            return aten.zendnn_linear_unary_binary(
                                arg_1,
                                arg_2,
                                arg_0,
                                is_weight_prepacked=is_weight_prepacked,
                                post_op_1=new_unary_name,
                                post_op_2=binary_name,
                            )

                    match.replace_by_example(
                        repl, [arg_0, arg_1, arg_2, is_weight_prepacked]
                    )

                replacement_fn.__name__ = f"zendnn_linear_{unary_name}_{binary_name}_no_bias_replacement_{idx}"
                return replacement_fn

        # Register the pattern with the created replacement function
        replacement_func = create_replacement_fn_unary_binary(
            i, unary_post_op_name, binary_post_op_name, bias
        )
        register_graph_pattern(
            pattern_lst[i], pass_dict=pass_pattern, extra_check=binary_dim_check(i)
        )(replacement_func)


binary_post_op_dict: dict[str, Any] = {"add": aten.add, "mul": aten.mul}


@functools.cache
def register_unary_binary_fusions() -> None:
    for bias in [True, False]:
        for unary_post_op in ["none", "silu"]:
            for binary_post_op, binary_post_op_fn in binary_post_op_dict.items():
                if unary_post_op != "silu" or binary_post_op != "add":
                    if unary_post_op == "silu":
                        patterns_lst_no_decomp = unary_binary_patterns_generator(
                            aten.mul,
                            _silu_fusion_no_decomp(create_linear_compute_fn(bias=bias)),
                        )
                        register_unary_binary_patterns(
                            "silu_no_decomp", "mul", patterns_lst_no_decomp, bias
                        )
                        for with_prims in [True, False]:
                            if with_prims:
                                patterns_lst = unary_binary_patterns_generator(
                                    aten.mul,
                                    create_pattern(
                                        create_linear_compute_fn(bias=bias),
                                        _silu_fusion,
                                        with_prims=with_prims,
                                        users=2,
                                    ),
                                )
                            else:
                                patterns_lst = unary_binary_patterns_generator(
                                    aten.mul,
                                    create_pattern(
                                        create_linear_compute_fn(bias=bias, users=2),
                                        _silu_fusion,
                                        with_prims=with_prims,
                                    ),
                                )
                            register_unary_binary_patterns(
                                "silu", "mul", patterns_lst, bias
                            )
                    else:
                        patterns_lst = unary_binary_patterns_generator(
                            binary_post_op_fn, create_linear_compute_fn(bias=bias)
                        )
                        register_unary_binary_patterns(
                            unary_post_op, binary_post_op, patterns_lst, bias
                        )


def zendnn_unary_binary_post_op_fusions(gm: GraphModule) -> GraphModule:
    # register the patterns
    register_unary_binary_fusions()  # type: ignore[arg-type]
    GraphTransformObserver = functools.partial(
        torch.fx.passes.graph_transform_observer.GraphTransformObserver,
        subsystem="zendnn_unary_binary_post_op_fusions",
    )
    if config.pattern_matcher:
        GraphTransformObserver(gm, "pass_pattern").apply_graph_pass(pass_pattern.apply)
    stable_topological_sort(gm.graph)
    gm.graph.lint()
    gm.recompile()
    return gm
