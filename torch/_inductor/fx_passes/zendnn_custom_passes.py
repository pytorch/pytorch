# mypy: allow-untyped-defs
import functools

import torch
from torch._inductor import config
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
    stable_topological_sort,
)

from .zendnn_utils import counters


pass_pattern = PatternMatcherPass()
at_ops = torch.ops.aten


# zendnn_linear replacement with weight prepacking
@register_graph_pattern(
    CallFunction(
        at_ops.zendnn_linear,
        Arg(),
        Arg(),
    ),
    pass_dict=pass_pattern,
)
def zendnn_weight_prepack_for_linear_replacement_without_bias(
    match: Match, mat_1, mat_2
):
    def repl(mat_1, mat_2):
        counters["zendnn"]["zendnn_weight_prepack_for_linear"] += 1
        mat_2_prepacked = at_ops.zendnn_weight_prepack_for_linear(mat_2)
        return at_ops.zendnn_linear(mat_1, mat_2_prepacked, is_weight_prepacked=True)

    match.replace_by_example(repl, [mat_1, mat_2])


@register_graph_pattern(
    CallFunction(
        at_ops.zendnn_linear,
        Arg(),
        Arg(),
        Arg(),
    ),
    pass_dict=pass_pattern,
)
def zendnn_weight_prepack_for_linear_replacement_with_bias(
    match: Match, mat_1, mat_2, bias
):
    def repl(mat_1, mat_2, bias):
        counters["zendnn"]["zendnn_weight_prepack_for_linear"] += 1
        mat_2_prepacked = at_ops.zendnn_weight_prepack_for_linear(mat_2)
        return at_ops.zendnn_linear(
            mat_1, mat_2_prepacked, bias, is_weight_prepacked=True
        )

    match.replace_by_example(repl, [mat_1, mat_2, bias])


def add_zendnn_weight_prepack_ops(gm):
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
