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
    num_xcd: fx.Constexpr = 1,
    group_m: fx.Constexpr = 1,
):
    """Grid-stride traversal with optional XCD gathering and row grouping.

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
        if const_expr(group_m > 1):
            num_rows = (row_end - row_base + block_m - 1) >> row_shift
            first_m = (bid_m // group_m) * group_m
            remaining = num_rows - first_m
            group_size = arith.select(remaining < group_m, remaining, fx.Int32(group_m))
            within = (bid_m % group_m) * num_pid_n + bid_n
            bid_m = first_m + within % group_size
            bid_n = within // group_size
        tile_body(group, bid_m, bid_n, row_end - row_base, row_base)

    worker = fx.Int32(fx.block_idx.x)
    grid = fx.Int32(fx.grid_dim.x)
    if const_expr(num_xcd > 1):
        # Consecutive physical workgroups are distributed across XCDs. Give
        # each XCD consecutive logical workers, preserving the grid stride.
        worker = arith.select(
            grid % num_xcd == 0,
            (worker % num_xcd) * (grid // num_xcd) + worker // num_xcd,
            worker,
        )
    if const_expr(persistent):
        for tile in range(
            worker,
            cumulative[-1] * num_pid_n,
            grid,
        ):
            visit(tile)
    else:
        tile = worker
        if tile < cumulative[-1] * num_pid_n:
            visit(tile)
