# mypy: allow-untyped-defs
"""Inductor lowerings for flex_ep HOPs."""

from __future__ import annotations

import sys
from typing import Any, TYPE_CHECKING

import torch

from ...lowering import (
    lowerings,
    pointwise_cat,
    process_subgraph_nodes,
    register_lowering,
    to_dtype,
)


aten = torch.ops.aten


if TYPE_CHECKING:
    from ... import ir


@register_lowering(torch.ops.higher_order.flex_ep, type_promotion_kind=None)
def flex_ep(
    x,
    topk_idx,
    w13,
    w2,
    build_dispatch_plan_fn: ir.Subgraph,
    dispatch_fn: ir.Subgraph,
    combine_fn: ir.Subgraph,
    combine_bwd_fn: ir.Subgraph,
    dispatch_bwd_fn: ir.Subgraph,
    router_operands,
    num_experts: int,
    ep_rank: int,
    ep_size: int,
    max_tokens: int,
    topk: int,
    num_ctas: int = 152,
    _build_dispatch_plan_lifted_args: tuple[Any, ...] = (),
    _dispatch_lifted_args: tuple[Any, ...] = (),
    _combine_lifted_args: tuple[Any, ...] = (),
    _combine_bwd_lifted_args: tuple[Any, ...] = (),
    _dispatch_bwd_lifted_args: tuple[Any, ...] = (),
):
    del combine_bwd_fn
    del dispatch_bwd_fn
    del num_experts, ep_rank, ep_size, max_tokens, num_ctas
    del _combine_bwd_lifted_args, _dispatch_bwd_lifted_args

    x_expanded = lowerings[aten.unsqueeze](x, 1)
    x_size = x.get_size()
    x_expanded = lowerings[aten.expand](x_expanded, [x_size[0], topk, x_size[1]])
    x_expanded = lowerings[aten.clone](
        x_expanded, memory_format=torch.contiguous_format
    )

    plan = process_subgraph_nodes(
        build_dispatch_plan_fn.graph_module,
        [
            topk_idx,
            router_operands,
            *_build_dispatch_plan_lifted_args,
        ],
    )
    dispatch_out = process_subgraph_nodes(
        dispatch_fn.graph_module,
        [
            x_expanded,
            plan,
            router_operands,
            *_dispatch_lifted_args,
        ],
    )
    recv_x = (
        dispatch_out[0] if isinstance(dispatch_out, (tuple, list)) else dispatch_out
    )
    local_experts_start = plan.local_experts_start
    offs = lowerings[aten.slice](local_experts_start, 0, 1, sys.maxsize, 1)
    offs = to_dtype(offs, torch.int32)

    w13_t = lowerings[aten.permute](w13, [0, 2, 1])
    y1 = lowerings[aten._grouped_mm.default](recv_x, w13_t, offs, None, None)
    intermediate_dim = w2.get_size()[-1]
    gate = lowerings[aten.slice](y1, -1, 0, intermediate_dim, 1)
    up = lowerings[aten.slice](y1, -1, intermediate_dim, sys.maxsize, 1)
    y2 = lowerings[aten.mul](
        lowerings[aten.mul](gate, lowerings[aten.sigmoid](gate)), up
    )
    w2_t = lowerings[aten.permute](w2, [0, 2, 1])
    y3 = lowerings[aten._grouped_mm.default](y2, w2_t, offs, None, None)
    y3 = lowerings[aten.clone](y3, memory_format=torch.contiguous_format)
    y3.realize()
    return process_subgraph_nodes(
        combine_fn.graph_module,
        [
            y3,
            plan,
            router_operands,
            *_combine_lifted_args,
        ],
    )


@register_lowering(torch.ops.higher_order.flex_ep_backward, type_promotion_kind=None)
def flex_ep_backward(
    dy,
    recv_x,
    y1,
    y2,
    w13,
    w2,
    offs,
    token_end,
    combine_bwd_fn: ir.Subgraph,
    dispatch_bwd_fn: ir.Subgraph,
    plan,
    router_operands,
    _combine_bwd_lifted_args: tuple[Any, ...] = (),
    _dispatch_bwd_lifted_args: tuple[Any, ...] = (),
):
    del token_end

    dy3 = process_subgraph_nodes(
        combine_bwd_fn.graph_module,
        [
            dy,
            plan,
            router_operands,
            *_combine_bwd_lifted_args,
        ],
    )
    dy2 = lowerings[aten._grouped_mm.default](dy3, w2, offs, None, None)
    intermediate_dim = w2.get_size()[-1]
    gate = lowerings[aten.slice](y1, -1, 0, intermediate_dim, 1)
    up = lowerings[aten.slice](y1, -1, intermediate_dim, sys.maxsize, 1)
    sig = lowerings[aten.sigmoid](gate)
    gate_term = lowerings[aten.add](
        lowerings[aten.mul](
            gate,
            lowerings[aten.add](lowerings[aten.mul](sig, -1), 1),
        ),
        1,
    )
    dgate = lowerings[aten.mul](
        lowerings[aten.mul](lowerings[aten.mul](dy2, up), sig),
        gate_term,
    )
    dup = lowerings[aten.mul](
        dy2,
        lowerings[aten.mul](gate, sig),
    )
    dy1 = pointwise_cat([dgate, dup], -1)
    dx_recv = lowerings[aten._grouped_mm.default](dy1, w13, offs, None, None)
    dx_recv = lowerings[aten.clone](dx_recv, memory_format=torch.contiguous_format)
    dx_recv.realize()
    y2_t = lowerings[aten.permute](y2, [1, 0])
    dw2_t = lowerings[aten._grouped_mm.default](y2_t, dy3, offs, None, None)
    recv_x_t = lowerings[aten.permute](recv_x, [1, 0])
    dw13_t = lowerings[aten._grouped_mm.default](recv_x_t, dy1, offs, None, None)
    dxpn = process_subgraph_nodes(
        dispatch_bwd_fn.graph_module,
        [
            dx_recv,
            plan,
            router_operands,
            *_dispatch_bwd_lifted_args,
        ],
    )
    dx = lowerings[aten.sum](dxpn, axis=-2)
    dw13 = lowerings[aten.permute](dw13_t, [0, 2, 1])
    dw2 = lowerings[aten.permute](dw2_t, [0, 2, 1])
    return dx, dw13, dw2
