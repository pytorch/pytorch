import logging

import torch
import torch._guards
from .. import config
from ..pattern_matcher import (
    CallFunction,
    init_once_fakemode,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
    stable_topological_sort,
)

log = logging.getLogger(__name__)
patterns = PatternMatcherPass()


@init_once_fakemode
def lazy_init():
    from .fuse_attention import _sfdp_init

    _sfdp_init()


def joint_graph_passes(graph: torch.fx.GraphModule):
    """
    Run FX transformations on the joint forwards+backwards graph.
    """
    if not config.pattern_matcher:
        return

    lazy_init()

    count = patterns.apply(graph.graph)

    if count:
        stable_topological_sort(graph.graph)
        graph.graph.lint()
        graph.recompile()
    return graph


@register_graph_pattern(
    CallFunction(
        torch.ops.prims.convert_element_type.default,
        CallFunction(
            torch.ops.prims.convert_element_type.default,
            KeywordArg("arg"),
            KeywordArg("dtype1"),
        ),
        KeywordArg("dtype2"),
    ),
    pass_dict=patterns,
)
def pointless_convert(match: Match, arg, dtype1, dtype2):
    """Remove chain of dtype conversions often created by AMP"""
    graph = match.graph
    node = match.output_node()
    allowed = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    if dtype1 in allowed and dtype2 in allowed:
        repl = graph.call_function(
            torch.ops.prims.convert_element_type.default, (arg, dtype2)
        )
        repl.meta.update(node.meta)
        node.replace_all_uses_with(repl)
        match.erase_nodes(graph)


@register_graph_pattern(
    CallFunction(torch.ops.aten.view.default, KeywordArg("arg"), KeywordArg("size")),
    pass_dict=patterns,
)
def pointless_view(match: Match, arg, size):
    """Remove no-op view"""
    graph = match.graph
    node = match.output_node()
    arg_size = list(node.args[0].meta["val"].shape)
    if size == arg_size:
        node.replace_all_uses_with(node.args[0])
        match.erase_nodes(graph)
