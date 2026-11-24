# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T

from . import mxfp8_buffer_ops as buffer_ops


@flyc.jit
def for_each_grouped_tile_static(
    offsets,
    group_count: fx.Constexpr,
    n: fx.Constexpr,
    block_m: fx.Constexpr,
    block_n: fx.Constexpr,
    persistent: fx.Constexpr,
    tile_body,
):
    """N-fast grid-stride traversal with constexpr groups and column count.

    Load boundaries once per workgroup, and walk only actual tiles. The callback
    must finish LDS accesses and drain vector memory operations before returning.
    """
    num_pid_n = (n + block_n - 1) // block_n
    # This specialization is for power-of-two row tiles and monotone offsets.
    # Use a shift, avoiding signed floor-division fixups in the SALU prologue.
    if block_m <= 0 or (block_m & (block_m - 1)) != 0:
        raise AssertionError(f"block_m must be a positive power of two, got {block_m}")
    row_shift = block_m.bit_length() - 1
    resource = buffer_ops.create_buffer_resource(
        offsets, max_size=False, num_records_bytes=group_count * 4
    )
    ends = [
        ArithValue(
            buffer_ops.buffer_load(
                resource, fx.Int32(i), vec_width=1, dtype=T.i32, is_scalar=True
            )
        )
        for i in range(group_count)
    ]
    starts = [ArithValue(fx.Int32(0))] + ends[:-1]
    cumulative = [ArithValue(fx.Int32(0))]
    for i in range_constexpr(group_count):
        rows = ends[i] - starts[i]
        cumulative.append(cumulative[-1] + ((rows + block_m - 1) >> row_shift))

    def visit(tile):
        slot = tile // num_pid_n
        bid_n = tile % num_pid_n
        group = ArithValue(fx.Int32(0))
        bid_m = ArithValue(fx.Int32(0))
        row_base = ArithValue(fx.Int32(0))
        row_end = ArithValue(fx.Int32(0))
        for i in range_constexpr(group_count):
            hit = (slot >= cumulative[i]) & (slot < cumulative[i + 1])
            group = arith.select(hit, fx.Int32(i), group)
            bid_m = arith.select(hit, slot - cumulative[i], bid_m)
            row_base = arith.select(hit, starts[i], row_base)
            row_end = arith.select(hit, ends[i], row_end)
        tile_body(group, bid_m, bid_n, row_end - row_base, row_base)

    if const_expr(persistent):
        for tile in range(
            fx.Int32(fx.block_idx.x),
            cumulative[-1] * num_pid_n,
            fx.Int32(fx.grid_dim.x),
        ):
            visit(tile)
    else:
        tile = fx.Int32(fx.block_idx.x)
        if tile < cumulative[-1] * num_pid_n:
            visit(tile)
