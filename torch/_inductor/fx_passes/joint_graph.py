import functools
import logging

import torch
import torch._guards
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
from ..._subclasses import FakeTensorMode
from .. import config
from ..pattern_matcher import (
    Arg,
    CallFunction,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)

log = logging.getLogger(__name__)
patterns = PatternMatcherPass()


@functools.lru_cache(None)
def lazy_init():
    from .fuse_attention import _sfdp_init

    with torch._guards.tracing(
        None
    ), maybe_disable_fake_tensor_mode(), FakeTensorMode():
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
        graph.graph.lint()
        graph.recompile()
    return graph


@register_graph_pattern(
    CallFunction(
        torch.ops.prims.convert_element_type.default,
        CallFunction(
            torch.ops.prims.convert_element_type.default, Arg(), KeywordArg("dtype1")
        ),
        KeywordArg("dtype2"),
    ),
    pass_dict=patterns,
)
def pointless_convert(
    match: Match, graph: torch.fx.Graph, node: torch.fx.Node, arg, dtype1, dtype2
):
    """Remove chain of dtype conversions often created by AMP"""
    if dtype1.is_floating_point and dtype2.is_floating_point:
        repl = graph.call_function(
            torch.ops.prims.convert_element_type.default, (arg, dtype2)
        )
        repl.meta.update(node.meta)
        node.replace_all_uses_with(repl)
        match.erase_nodes(graph)
