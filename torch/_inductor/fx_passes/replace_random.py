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

from torch.library import custom_op


# ---- host helpers ----
def _compute_grid_x(nelem: int, block: int, device_index: int) -> int:
    prop = torch.cuda.get_device_properties(device_index)
    blocks_per_sm = prop.max_threads_per_multi_processor // block
    max_blocks = prop.multi_processor_count * blocks_per_sm
    need_blocks = (nelem + block - 1) // block
    return min(max_blocks, need_blocks)


def _shape_to_offset(size, device: torch.device) -> int:
    nelem = 1
    for s in size:
        nelem *= int(s)

    UNROLL = 4
    prop = torch.cuda.get_device_properties(device)

    threads_per_round = (
        prop.multi_processor_count * prop.max_threads_per_multi_processor
    )
    rounds_per_thread = (nelem + threads_per_round * UNROLL - 1) // (
        threads_per_round * UNROLL
    )
    used_offset = rounds_per_thread * UNROLL
    return used_offset


def _reserve_offset(device: torch.device, used_offset: int) -> int:
    dev_index = device.index if isinstance(device, torch.device) else int(device)
    gen = torch.cuda.default_generators[dev_index]
    old_off = int(gen.get_offset())
    gen.set_offset(old_off + used_offset)
    return old_off // 4


@custom_op("custom_op::rand_eager_offset", mutates_args={})
def rand_eager_offset(offset: int, device: torch.device) -> torch.Tensor:
    base = _reserve_offset(device, int(offset))
    out = torch.empty(1, dtype=torch.int64, device=device)
    out.fill_(int(base))
    return out


@custom_op("custom_op::rand_eager_offsets", mutates_args={})
def rand_eager_offsets(offsets: list[int], device: torch.device) -> torch.Tensor:
    bases = [int(_reserve_offset(device, int(off))) for off in offsets]
    cpu = torch.tensor(bases, dtype=torch.int64).pin_memory()
    out = torch.empty_like(cpu, device=device)
    out.copy_(cpu, non_blocking=True)
    return out


@rand_eager_offset.register_fake
def _(offset: int, device: torch.device):
    return torch.empty((1,), dtype=torch.int64, device=device)


@rand_eager_offsets.register_fake
def _(offsets: list[int], device: torch.device):
    return torch.empty((len(offsets),), dtype=torch.int64, device=device)


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
    Horizontally fuse all the seed generation on each device

        a = custom_op.rand_eager_offset(offset, dev)
        b = custom_op.rand_eager_offset(offset, dev)

    Becomes:
        offsets = custom_op.rand_eager_offsets([offset1, offset2...], dev)
        a = inductor_lookup_seed(offsets, 0)
        b = inductor_lookup_seed(offsets, 1)

    We do this because seed creation is entirely launch overhead bound.
    """
    device_offsets = collections.defaultdict(list)
    for node in graph.nodes:
        if CallFunctionVarArgs(torch.ops.custom_op.rand_eager_offset).match(node):
            device_offsets[node.args[1]].append(node)

    if not device_offsets:
        return 0

    for device, offsets in device_offsets.items():
        with graph.inserting_before(offsets[0]):
            offs = [n.args[0] for n in offsets]
            combined = graph.call_function(
                torch.ops.custom_op.rand_eager_offsets.default, (offs, device)
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

    if mode == "rand" and config.align_random_eager:

        def replacement(size):
            offset = _shape_to_offset(size, device)

            align_dtype = dtype
            if isinstance(align_dtype, (tuple, list)):
                align_dtype = align_dtype[0] if len(align_dtype) else None
            if align_dtype is None:
                align_dtype = torch.float32

            result = inductor_prims.random(
                size,
                torch.ops.custom_op.rand_eager_offset(offset, device),
                mode,
                **default_kwargs(device),
                align_dtype=align_dtype,
            )
            if dtype is not None:
                result = result.to(dtype)
            return result

        match.replace_by_example(replacement, [size])
        return

    # Fallback (e.g., randn) keeps existing inductor behavior

    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(replacement, [size])


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
