# mypy: allow-untyped-defs
import collections
import functools
import logging
from typing import Any

import torch
from torch import Tensor
from torch._dynamo.utils import counters
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch.fx.passes.shape_prop import _extract_tensor_metadata

from .. import config, inductor_prims
from ..pattern_matcher import (
    CallFunctionVarArgs,
    fwd_only,
    init_once_fakemode,
    joint_fwd_bwd,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
    register_replacement,
)
from ..virtualized import V


log = logging.getLogger(__name__)
patterns = PatternMatcherPass(subsystem="joint_graph_passes")
aten = torch.ops.aten


def replace_random_passes(gm: torch.fx.GraphModule):
    """Modify the given FX graph to use backend-native random ops"""
    if config.fallback_random:
        return 0

    lazy_init()
    count = patterns.apply(gm)
    with GraphTransformObserver(gm, "fuse_seed_creation_pass", "joint_graph_passes"):
        count += fuse_seed_creation_pass(gm.graph)

    return count


def fuse_seed_creation_pass(graph: torch.fx.Graph):
    """
    Horizontally fuse all the seed generation on each device

        a = inductor_seed(dev)
        b = inductor_seed(dev)

    Becomes:
        seeds = inductor_seeds(2, dev)
        a = inductor_lookup_seed(seeds, 0)
        b = inductor_lookup_seed(seeds, 1)

    We do this because seed creation is entirely launch overhead bound.
    """
    device_seeds = collections.defaultdict(list)
    for node in graph.nodes:
        if CallFunctionVarArgs(inductor_prims.seed).match(node):
            device_seeds[node.args[0]].append(node)

    if not device_seeds:
        return 0

    for device, seeds in device_seeds.items():
        with graph.inserting_before(seeds[0]):
            combined = graph.call_function(inductor_prims.seeds, (len(seeds), device))
            with V.fake_mode:
                combined.meta["val"] = torch.empty(
                    [len(seeds)], device=device, dtype=torch.int64
                )
                combined.meta["tensor_meta"] = _extract_tensor_metadata(
                    combined.meta["val"]
                )

        for idx, seed in enumerate(seeds):
            with graph.inserting_before(seed):
                new_seed = graph.call_function(
                    inductor_prims.lookup_seed, (combined, idx)
                )
            seed.replace_all_uses_with(new_seed)
            new_seed.meta.update(seed.meta)
            graph.erase_node(seed)

    return len(device_seeds)


def default_kwargs(device):
    return {}


def get_device(device):
    if device is not None:
        return device
    return torch.empty([]).device  # default device


# pyrefly: ignore [bad-argument-type]
@register_graph_pattern(CallFunctionVarArgs(aten.rand.default), pass_dict=patterns)
# pyrefly: ignore [bad-argument-type]
@register_graph_pattern(CallFunctionVarArgs(aten.rand.generator), pass_dict=patterns)
# pyrefly: ignore [bad-argument-type]
@register_graph_pattern(CallFunctionVarArgs(aten.randn.default), pass_dict=patterns)
# pyrefly: ignore [bad-argument-type]
@register_graph_pattern(CallFunctionVarArgs(aten.randn.generator), pass_dict=patterns)
def replace_random(
    match: Match,
    size,
    *,
    generator=None,
    dtype=None,
    device=None,
    layout=None,
    pin_memory=None,
):
    if generator is not None:
        return

    def replacement(size):
        result = inductor_prims.random(
            size, inductor_prims.seed(device), mode, **default_kwargs(device)
        )
        if dtype is not None:
            result = result.to(dtype)
        return result

    mode = {
        aten.rand: "rand",
        aten.randn: "randn",
    }[
        match.output_node().target.overloadpacket  # type: ignore[union-attr]
    ]  # type: ignore[union-attr]
    device = get_device(device)
    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(replacement, [size])


@init_once_fakemode
def lazy_init():
    if not torch.cuda.is_available():
        return

    # workaround https://github.com/pytorch/pytorch/issues/97894
    device = "cuda"
    # sizes/values don't actually matter for initial trace
    # once we get a possible match we re-trace with the actual values and verify the match still holds
    t = functools.partial(torch.empty, [1], device=device)
    # workaround https://github.com/pytorch/pytorch/issues/97894
    # 0.113377 is a "magic" value that lets us recover the lost input arg relationship
    workaround = {"dropout_p": 0.113377}

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        _dropout_pattern,
        # pyrefly: ignore [bad-argument-type]
        _dropout_replacement,
        [t(requires_grad=True), *workaround.values()],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        patterns,
        scalar_workaround=workaround,
    )
    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        _dropout_pattern,
        # pyrefly: ignore [bad-argument-type]
        _dropout_replacement,
        [t(requires_grad=True), *workaround.values()],
        # pyrefly: ignore [bad-argument-type]
        joint_fwd_bwd,
        # pyrefly: ignore [bad-argument-type]
        patterns,
        scalar_workaround=workaround,
    )


def _dropout_pattern(x: torch.Tensor, dropout_p: float):
    return torch.dropout(x, dropout_p, True)


def _dropout_replacement(x: torch.Tensor, dropout_p: float):
    assert 0 < dropout_p < 1, "should have been handled in decomps"
    counters["inductor"]["replace_random"] += 1
    seed = inductor_prims.seed(x.device)
    scale = float(1.0 / (1.0 - dropout_p))

    def get_bool_mask():
        return inductor_prims.random(x.size(), seed, "rand") > dropout_p

    class Dropout(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Tensor) -> Tensor:  # type: ignore[override]
            return get_bool_mask().to(x.dtype) * x * scale

        @staticmethod
        def backward(ctx: Any, grad_output: Tensor) -> Tensor:  # type: ignore[override]
            return get_bool_mask().to(grad_output.dtype) * grad_output * scale

    return Dropout.apply(x)


# pyrefly: ignore [bad-argument-type]
@register_graph_pattern(CallFunctionVarArgs(aten.randint.low), pass_dict=patterns)
def replace_randint(
    match: Match,
    low,
    high,
    size,
    *,
    dtype=torch.int64,
    device=None,
    layout=None,
    pin_memory=None,
):
    def replacement(low, high, size):
        result = inductor_prims.randint(low, high, size, inductor_prims.seed(device))
        return result.to(dtype)

    device = get_device(device)
    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(replacement, [low, high, size])
