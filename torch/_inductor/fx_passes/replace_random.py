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


log = logging.getLogger(__name__)
patterns = PatternMatcherPass(subsystem="joint_graph_passes")
aten = torch.ops.aten


def _shape_to_offset(shape, device: torch.device):
    # Modified from torch/_prims/rng_prims.py:philox_rand_offset
    nelem = 1
    for s in shape:
        nelem *= s

    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if device.type != "cuda":
        return 0

    block_size = 256
    unroll = 4
    curand4_engine_calls = 4

    device_property = torch.cuda.get_device_properties(device)

    blocks_per_sm = device_property.max_threads_per_multi_processor // block_size
    grid_size = (nelem + block_size - 1) // block_size
    grid_size = min(grid_size, device_property.multi_processor_count * blocks_per_sm)

    return ((nelem - 1) // (block_size * grid_size * unroll) + 1) * curand4_engine_calls


def replace_random_passes(gm: torch.fx.GraphModule):
    """Modify the given FX graph to use backend-native random ops"""
    if config.fallback_random:
        return 0

    count = patterns.apply(gm)
    with GraphTransformObserver(gm, "fuse_seed_creation_pass", "joint_graph_passes"):
        count += fuse_seed_creation_pass(gm.graph)
    if config.align_random_eager:
        with GraphTransformObserver(gm, "fuse_offset_creation_pass"):
            count += fuse_offset_creation_pass(gm.graph)

    return count


def fuse_offset_creation_pass(graph: torch.fx.Graph):
    """
    Here offset node means seed << 32 + offset, will unpacked in lowering.py:inductor_random()
    Horizontally fuse all the seed generation on each device
        a = inductor_prims.rand_eager_offset(offset, dev)
        b = inductor_prims.rand_eager_offset(offset, dev)
    Becomes:
        offsets = inductor_prims.rand_eager_offsets([offset1, offset2...], dev)
        a = inductor_lookup_seed(offsets, 0)
        b = inductor_lookup_seed(offsets, 1)
    We do this because seed creation is entirely launch overhead bound.
    """
    device_offsets = collections.defaultdict(list)
    for node in graph.nodes:
        if CallFunctionVarArgs(inductor_prims.rand_eager_offset).match(node):
            device_offsets[node.args[1]].append(node)

    if not device_offsets:
        return 0

    for device, offsets in device_offsets.items():
        with graph.inserting_before(offsets[0]):
            offs = [n.args[0] for n in offsets]
            combined = graph.call_function(
                inductor_prims.rand_eager_offsets, (offs, device)
            )
            with V.fake_mode:
                combined.meta["val"] = torch.empty(
                    [len(offsets)], device=device, dtype=torch.int64
                )
                combined.meta["tensor_meta"] = _extract_tensor_metadata(
                    combined.meta["val"]
                )

        for idx, offset in enumerate(offsets):
            with graph.inserting_before(offset):
                new_offset = graph.call_function(
                    inductor_prims.lookup_seed, (combined, idx)
                )
            offset.replace_all_uses_with(new_offset)
            new_offset.meta.update(offset.meta)
            graph.erase_node(offset)

    return len(device_offsets)


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
    replacement_fn = replacement

    if mode == "rand" and config.align_random_eager and device.type == "cuda":
        # Only enable when align_random_eager is on.
        def replacement_align(size):
            offset = _shape_to_offset(size, device)

            align_dtype = dtype
            if isinstance(align_dtype, (tuple, list)):
                align_dtype = align_dtype[0] if len(align_dtype) else None
            if align_dtype is None:
                align_dtype = torch.float32

            result = inductor_prims.random(
                size,
                inductor_prims.rand_eager_offset(offset, device),
                mode,
                **default_kwargs(device),
                align_dtype=align_dtype,
            )
            if dtype is not None:
                result = result.to(dtype)
            return result

        replacement_fn = replacement_align

    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(replacement_fn, [size])


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
