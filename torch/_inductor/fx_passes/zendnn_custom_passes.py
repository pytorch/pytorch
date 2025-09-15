import functools
from typing import Any

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


pass_pattern = PatternMatcherPass()
aten = torch.ops.aten


# zendnn_linear_unary replacement with weight prepacking
@register_graph_pattern(
    CallFunction(
        aten.zendnn_linear_unary,
        Arg(),
        Arg(),
    ),
    pass_dict=pass_pattern,
)
def zendnn_weight_prepack_for_linear_replacement_without_bias(
    match: Match, mat_1: Any, mat_2: Any
) -> None:
    def repl(mat_1: Any, mat_2: Any) -> torch.Tensor:
        counters["zendnn"]["zendnn_weight_prepack_for_linear"] += 1
        mat_2_prepacked = aten.zendnn_weight_prepack_for_linear(mat_2)
        return aten.zendnn_linear_unary(
            mat_1, mat_2_prepacked, is_weight_prepacked=True
        )

    match.replace_by_example(repl, [mat_1, mat_2])


@register_graph_pattern(
    CallFunction(
        aten.zendnn_linear_unary,
        Arg(),
        Arg(),
        Arg(),
    ),
    pass_dict=pass_pattern,
)
def zendnn_weight_prepack_for_linear_replacement_with_bias(
    match: Match, mat_1: Any, mat_2: Any, bias: Any
) -> None:
    def repl(mat_1: Any, mat_2: Any, bias: Any) -> torch.Tensor:
        counters["zendnn"]["zendnn_weight_prepack_for_linear"] += 1
        mat_2_prepacked = aten.zendnn_weight_prepack_for_linear(mat_2)
        return aten.zendnn_linear_unary(
            mat_1, mat_2_prepacked, bias, is_weight_prepacked=True
        )

    match.replace_by_example(repl, [mat_1, mat_2, bias])


def add_zendnn_weight_prepack_ops(gm: GraphModule) -> GraphModule:
    GraphTransformObserver = functools.partial(
        torch.fx.passes.graph_transform_observer.GraphTransformObserver,
        subsystem="add_zendnn_weight_prepack_ops",
    )

    if config.pattern_matcher:
        GraphTransformObserver(gm, "pass_pattern").apply_graph_pass(pass_pattern.apply)
    stable_topological_sort(gm.graph)
    gm.graph.lint()
    gm.recompile()
    return gm
