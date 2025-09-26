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

import os
import math

from typing import Sequence, Optional
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton, register_fake, register_kernel

BLOCK  = 256

@triton.jit
def _pick_lane(u0, u1, u2, u3, lane):
    v = tl.where(lane == 0, u0, u1)
    v = tl.where(lane == 1, u1, v)
    v = tl.where(lane == 2, u2, v)
    v = tl.where(lane == 3, u3, v)
    return v

@triton.jit
def _philox_fill_uniform_gridstride(out_ptr, n_elements, seed, offset_blocks, lane_shift,threads_per_round,BLOCK: tl.constexpr = BLOCK):
    UNROLL = 4
    pid = tl.program_id(0)                       # [0, grid_x)
    tid = pid * BLOCK + tl.arange(0, BLOCK)      # [0, BLOCK*grid_x)
    inv  = 1.0 / 4294967296.0
    half = inv * 0.5

    # rounds_total = ceil(n / (threads_per_round * UNROLL))
    rounds_total = (n_elements + threads_per_round * UNROLL - 1) // (threads_per_round * UNROLL)

    # tl.device_print("rand_philox offset_blocks %d rounds_total %d\n", offset_blocks, rounds_total)

    for r in range(rounds_total):
        subseq = (tid).to(tl.uint64)
        lane = ((tid + lane_shift) % 4).to(tl.uint64)

        offblk = tl.full(subseq.shape, (offset_blocks + r), tl.uint64) 
        u0, u1, u2, u3 = tl.philox(
            seed,
            (offblk & 0xFFFFFFFF).to(tl.uint32),
            ((offblk >> 32) & 0xFFFFFFFF).to(tl.uint32),
            (subseq & 0xFFFFFFFF).to(tl.uint32),
            ((subseq >> 32) & 0xFFFFFFFF).to(tl.uint32),
        )
    
        inv  = 1.0 / 4294967296.0  # 2^-32
        half = inv * 0.5

        base   = tid * 4
        stride = threads_per_round
        
        # k=0
        i0 = base + (r * UNROLL) * stride
        m0 = i0 < n_elements
        lane0 = tl.full(tid.shape, (lane_shift + 0) % 4, tl.uint32)
        f0 = _pick_lane(u0, u1, u2, u3, lane0).to(tl.float32) * inv + half
        tl.store(out_ptr + i0, 1.0 - f0, mask=m0)

        # k=1
        i1 = base + 1 + (r * UNROLL) * stride
        m1 = i1 < n_elements
        lane1 = tl.full(tid.shape, (lane_shift + 1) % 4, tl.uint32)
        f1 = _pick_lane(u0, u1, u2, u3, lane1).to(tl.float32) * inv + half
        tl.store(out_ptr + i1, 1.0 - f1, mask=m1)

        # k=2
        i2 = base + 2 + (r * UNROLL) * stride
        m2 = i2 < n_elements
        lane2 = tl.full(tid.shape, (lane_shift + 2) % 4, tl.uint32)
        f2 = _pick_lane(u0, u1, u2, u3, lane2).to(tl.float32) * inv + half
        tl.store(out_ptr + i2, 1.0 - f2, mask=m2)

        # k=3
        i3 = base + 3 + (r * UNROLL) * stride
        m3 = i3 < n_elements
        lane3 = tl.full(tid.shape, (lane_shift + 3) % 4, tl.uint32)
        f3 = _pick_lane(u0, u1, u2, u3, lane3).to(tl.float32) * inv + half
        tl.store(out_ptr + i3, 1.0 - f3, mask=m3)


# ---- host helpers ----
def _compute_grid_x(nelem: int, block: int, device_index: int) -> int:
    prop = torch.cuda.get_device_properties(device_index)
    blocks_per_sm = prop.max_threads_per_multi_processor // block
    max_blocks = prop.multi_processor_count * blocks_per_sm
    need_blocks = (nelem + block - 1) // block
    return min(max_blocks, need_blocks)

def _reserve_seed_and_offset_gridstride(x_device_index: int, nelem: int, block: int):
    UNROLL = 4
    gen = torch.cuda.default_generators[x_device_index]
    seed = int(gen.initial_seed())
    grid_x = _compute_grid_x(nelem, block, x_device_index)
    rounds_per_thread = (nelem + (block * grid_x * UNROLL) - 1) // (block * grid_x * UNROLL)
    counter_offset_per_thread = rounds_per_thread * UNROLL
    used_32 = counter_offset_per_thread #* block * grid_x
    old_off = int(gen.get_offset())
    gen.set_offset(old_off + used_32)  
    return seed, (old_off // 4), (old_off % 4), grid_x

def get_and_acc_base_offset():
    pass

@triton.jit
def _write_offset(out_ptr, base_offset):
    # BLOCK = 64
    # idx = tl.arange(0, BLOCK)
    # offset_vec = tl.full(idx.shape, base_offset, tl.uint32)
    # mask = (idx == 0)
    # tl.store(out_ptr + idx, offset_vec, mask=mask)

    tl.store(out_ptr + 0, base_offset, mask=True)

@triton_op("triton_op::rand_eager_offset", mutates_args={})
def rand_eager_offset(
    shape: Sequence[int], #*,
    device: torch.device,
) -> torch.Tensor:
    out = torch.empty(1, dtype=torch.uint32, device=device)

    n = 1
    for d in shape:
        n *= d

    seed_val, base_offset, lane_shift, grid_x = _reserve_seed_and_offset_gridstride(device.index, n, BLOCK)

    grid = lambda meta: (1,)
    wrap_triton(_write_offset)[grid] (out, base_offset)
    return out


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

    mode = {
        aten.rand: "rand",
        aten.randn: "randn",
    }[
        match.output_node().target.overloadpacket  # type: ignore[union-attr]
    ]  # type: ignore[union-attr]
    device = get_device(device)

    if mode == "rand" and config.align_random_eager:
        def replacement(size):
            result = inductor_prims.random(
                size, inductor_prims.lookup_seed(torch.ops.my_triton_op.rand_eager_offset(size, device), 0), mode, **default_kwargs(device)
            )
            if dtype is not None:
                result = result.to(dtype)
            return result

        match.replace_by_example(replacement, [size])
        return

    # Fallback (e.g., randn) keeps existing inductor behavior
    
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
