# mypy: allow-untyped-defs
import collections
import logging

import torch
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch.fx.passes.shape_prop import _extract_tensor_metadata

from .. import config, inductor_prims
from ..pattern_matcher import (
    CallFunctionVarArgs,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from ..virtualized import V
from . import custom_philox_rand

log = logging.getLogger(__name__)
patterns = PatternMatcherPass()
aten = torch.ops.aten


def replace_random_passes(gm: torch.fx.GraphModule):
    """Modify the given FX graph to use backend-native random ops"""
    if config.fallback_random:
        return 0

    count = patterns.apply(gm)
    with GraphTransformObserver(gm, "fuse_seed_creation_pass"):
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


@register_graph_pattern(CallFunctionVarArgs(aten.rand.default), pass_dict=patterns)
@register_graph_pattern(CallFunctionVarArgs(aten.rand.generator), pass_dict=patterns)
@register_graph_pattern(CallFunctionVarArgs(aten.randn.default), pass_dict=patterns)
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

    mode = {
        aten.rand: "rand",
        aten.randn: "randn",
    }[
        match.output_node().target.overloadpacket  # type: ignore[union-attr]
    ]  # type: ignore[union-attr]
    device = get_device(device)
    # For uniform rand (e.g., used by dropout), call our custom Triton op directly
    
    if mode == "rand":
        def replacement(size):
            # dtype: keep caller's dtype if provided, else default fp32
            use_dtype = dtype if dtype is not None else torch.float32
            return torch.ops.my_triton_op.philox_rand(size, device, use_dtype)

        match.replace_by_example(replacement, [size])
        return

    # Fallback (e.g., randn) keeps existing inductor behavior
    def replacement(size):
        result = inductor_prims.random(
            size, inductor_prims.seed(device), mode, **default_kwargs(device)
        )
        if dtype is not None:
            result = result.to(dtype)
        return result

    match.replace_by_example(replacement, [size])


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
    match.replace_by_example(replacement, [low, high, size])
