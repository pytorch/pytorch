import functools
import logging

import torch
import torch._guards
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
from ..._subclasses import FakeTensorMode
from .. import config
from ..pattern_matcher import PatternMatcherPass

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

    if patterns.apply(graph.graph):
        graph.graph.lint()
        graph.recompile()

    return graph
