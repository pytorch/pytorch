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

from .zendnn_unary_binary_post_op_fusions import (
    binary_post_op_dict,
    calc_strides,
    same_dtypes_check,
)
from .zendnn_unary_post_op_fusions import create_linear_compute_fn


pass_pattern = PatternMatcherPass()
aten = torch.ops.aten


_mapping: dict[int, tuple[int, int]] = {
    0: (3, 4),
    1: (0, 4),
    2: (4, 0),
    3: (0, 1),
}
_mapping_no_bias: dict[int, tuple[int, int]] = {
    0: (2, 3),
    1: (0, 3),
    2: (3, 0),
    3: (0, 1),
}


def binary_binary_dim_check(
    idx: int, dfs_map: dict[int, tuple[int, int]]
) -> Callable[[Match], bool]:
    def fn(match: Match) -> bool:
        post_op_indices = dfs_map[idx]
        if isinstance(
            match.args[post_op_indices[0]], torch.fx.node.Node
        ) and isinstance(match.args[post_op_indices[1]], torch.fx.node.Node):
            s1 = match.args[idx].meta["val"].shape
            new_shape = s1[:-1] + (match.args[idx + 1].meta["val"].shape[0],)
            if (
                match.args[post_op_indices[0]].meta["val"].shape != new_shape
                or match.args[post_op_indices[1]].meta["val"].shape != new_shape
            ):
                return False
            elif calc_strides(new_shape) != list(
                match.args[post_op_indices[0]].meta["val"].stride()
            ) or calc_strides(new_shape) != list(
                match.args[post_op_indices[1]].meta["val"].stride()
            ):
                return False
        return same_dtypes_check(match)

    return fn


def binary_binary_patterns_generator(
    aten_op_outer: Any, aten_op_inner: Any, compute_fn: Any
) -> list[CallFunction]:
    # aten_op_outer is considered as the return node in the graph always
    # otherwise the combinations will increase in the list below
    # for example in linear-mul-add fusion, the add op will be aten_op_outer
    # and mul will be aten_op_inner
    # DFS orders are:
    # 0. *linear_args, aten_op_inner_arg, aten_op_outer_arg
    # 1. aten_op_inner_arg, *linear_args, aten_op_outer_arg
    # 2. aten_op_outer_arg, *linear_args, aten_op_inner_arg
    # 3. aten_op_inner_arg, aten_op_outer_arg, *linear_args
    return [
        CallFunction(
            aten_op_outer, CallFunction(aten_op_inner, compute_fn, Arg()), Arg()
        ),
        CallFunction(
            aten_op_outer, CallFunction(aten_op_inner, Arg(), compute_fn), Arg()
        ),
        CallFunction(
            aten_op_outer, Arg(), CallFunction(aten_op_inner, compute_fn, Arg())
        ),
        CallFunction(
            aten_op_outer, Arg(), CallFunction(aten_op_inner, Arg(), compute_fn)
        ),
    ]


def register_binary_binary_patterns(
    binary_post_op_name_1: str,
    binary_post_op_name_2: str,
    pattern_lst: list[CallFunction],
    bias: bool,
) -> None:
    """Register binary-binary fusion patterns along with replacements."""
    for i in range(len(pattern_lst)):
        # Create a unique function for each pattern
        def create_replacement_fn_binary_binary(
            idx: int, binary_name_1: str, binary_name_2: str, has_bias: bool
        ) -> Callable[..., Any]:
            if has_bias:

                def replacement_fn(
                    match: Match,
                    arg_0: Any,
                    arg_1: Any,
                    arg_2: Any,
                    arg_3: Any,
                    arg_4: Any,
                    *,
                    is_weight_prepacked: Any,
                ) -> None:
                    def repl(
                        arg_0: Any,
                        arg_1: Any,
                        arg_2: Any,
                        arg_3: Any,
                        arg_4: Any,
                        is_weight_prepacked: Any,
                    ) -> torch.Tensor:
                        counters["zendnn"][
                            "zendnn_linear_" + binary_name_1 + "_" + binary_name_2
                        ] += 1
                        if idx == 0:
                            return aten.zendnn_linear_binary_binary(
                                arg_0,
                                arg_1,
                                arg_3,
                                arg_4,
                                arg_2,
                                is_weight_prepacked=is_weight_prepacked,
                                post_op_1=binary_name_1,
                                post_op_2=binary_name_2,
                            )
                        elif idx == 1:
                            return aten.zendnn_linear_binary_binary(
                                arg_1,
                                arg_2,
                                arg_0,
                                arg_4,
                                arg_3,
                                is_weight_prepacked=is_weight_prepacked,
                                post_op_1=binary_name_1,
                                post_op_2=binary_name_2,
                            )
                        elif idx == 2:
                            return aten.zendnn_linear_binary_binary(
                                arg_1,
                                arg_2,
                                arg_4,
                                arg_0,
                                arg_3,
                                is_weight_prepacked=is_weight_prepacked,
                                post_op_1=binary_name_1,
                                post_op_2=binary_name_2,
                            )
                        else:
                            return aten.zendnn_linear_binary_binary(
                                arg_2,
                                arg_3,
                                arg_0,
                                arg_1,
                                arg_4,
                                is_weight_prepacked=is_weight_prepacked,
                                post_op_1=binary_name_1,
                                post_op_2=binary_name_2,
                            )

                    match.replace_by_example(
                        repl, [arg_0, arg_1, arg_2, arg_3, arg_4, is_weight_prepacked]
                    )

                replacement_fn.__name__ = (
                    f"{binary_name_1}_{binary_name_2}_linear_replacement_{idx}"
                )
                return replacement_fn
            else:

                def replacement_fn(  # type: ignore[misc]
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
                        counters["zendnn"][
                            "zendnn_linear_" + binary_name_1 + "_" + binary_name_2
                        ] += 1
                        if idx == 0:
                            return aten.zendnn_linear_binary_binary(
                                arg_0,
                                arg_1,
                                arg_2,
                                arg_3,
                                is_weight_prepacked=is_weight_prepacked,
                                post_op_1=binary_name_1,
                                post_op_2=binary_name_2,
                            )
                        elif idx == 1:
                            return aten.zendnn_linear_binary_binary(
                                arg_1,
                                arg_2,
                                arg_0,
                                arg_3,
                                is_weight_prepacked=is_weight_prepacked,
                                post_op_1=binary_name_1,
                                post_op_2=binary_name_2,
                            )
                        elif idx == 2:
                            return aten.zendnn_linear_binary_binary(
                                arg_1,
                                arg_2,
                                arg_3,
                                arg_0,
                                is_weight_prepacked=is_weight_prepacked,
                                post_op_1=binary_name_1,
                                post_op_2=binary_name_2,
                            )
                        else:
                            return aten.zendnn_linear_binary_binary(
                                arg_2,
                                arg_3,
                                arg_0,
                                arg_1,
                                is_weight_prepacked=is_weight_prepacked,
                                post_op_1=binary_name_1,
                                post_op_2=binary_name_2,
                            )

                    match.replace_by_example(
                        repl, [arg_0, arg_1, arg_2, arg_3, is_weight_prepacked]
                    )

                replacement_fn.__name__ = (
                    f"{binary_name_1}_{binary_name_2}_linear_no_bias_replacement_{idx}"
                )
                return replacement_fn

        # Register the pattern with the created replacement function
        replacement_func = create_replacement_fn_binary_binary(
            i, binary_post_op_name_1, binary_post_op_name_2, bias
        )
        if bias:
            register_graph_pattern(
                pattern_lst[i],
                pass_dict=pass_pattern,
                extra_check=binary_binary_dim_check(i, _mapping),
            )(replacement_func)
        else:
            register_graph_pattern(
                pattern_lst[i],
                pass_dict=pass_pattern,
                extra_check=binary_binary_dim_check(i, _mapping_no_bias),
            )(replacement_func)


@functools.cache
def register_binary_binary_fusions() -> None:
    for bias in [True, False]:
        for binary_post_op, binary_post_op_fn in binary_post_op_dict.items():
            # for now, we are only supporting the return node as add op
            # currently supported fusions: mul+add and add+add
            patterns_lst = binary_binary_patterns_generator(
                aten.add, binary_post_op_fn, create_linear_compute_fn(bias=bias)
            )
            register_binary_binary_patterns(binary_post_op, "add", patterns_lst, bias)


def zendnn_binary_binary_post_op_fusions(gm: GraphModule) -> GraphModule:
    # register the patterns
    register_binary_binary_fusions()  # type: ignore[arg-type]
    GraphTransformObserver = functools.partial(
        torch.fx.passes.graph_transform_observer.GraphTransformObserver,
        subsystem="zendnn_binary_binary_post_op_fusions",
    )
    if config.pattern_matcher:
        GraphTransformObserver(gm, "pass_pattern").apply_graph_pass(pass_pattern.apply)
    stable_topological_sort(gm.graph)
    gm.graph.lint()
    gm.recompile()
    return gm
