# mypy: allow-untyped-defs
"""Autograd-aware BF16 expert-parallel MoE block."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, cast, TYPE_CHECKING

import torch
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.utils import (
    _maybe_reenter_make_fx,
    autograd_not_implemented,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.fx.node import has_side_effect


if TYPE_CHECKING:
    from collections.abc import Callable


FLEX_EP_PLAN_FIELDS = (
    "recv_origin_global_token_id",
    "expert_begin_offset_per_ep",
    "dest_ranks",
    "dest_offsets",
    "local_experts_start",
    "max_recv_tokens",
    "recv_total_tokens",
    "overflow",
)


@dataclass(frozen=True)
class FlexEPDispatchPlan:
    recv_origin_global_token_id: torch.Tensor
    expert_begin_offset_per_ep: torch.Tensor
    dest_ranks: torch.Tensor
    dest_offsets: torch.Tensor
    local_experts_start: torch.Tensor
    max_recv_tokens: torch.Tensor
    recv_total_tokens: torch.Tensor
    overflow: torch.Tensor


pytree.register_dataclass(FlexEPDispatchPlan)


def validate_dispatch_plan(plan: Any) -> Any:
    for name in FLEX_EP_PLAN_FIELDS:
        if not hasattr(plan, name):
            raise RuntimeError(f"flex_ep dispatch plan missing field {name}")
        value = getattr(plan, name)
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"flex_ep dispatch-plan field {name} must be a tensor")
    return plan


def _validate_flex_ep_inputs(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    router_operands: Any,
    *,
    num_experts: int,
    ep_rank: int,
    ep_size: int,
    max_tokens: int,
    topk: int,
    num_ctas: int,
) -> None:
    if x.dim() != 2:
        raise ValueError(f"flex_ep expects x to be 2D, got {x.dim()}D")
    if topk_idx.dim() != 2:
        raise ValueError(f"flex_ep expects topk_idx to be 2D, got {topk_idx.dim()}D")
    if topk_idx.shape[0] != x.shape[0]:
        raise ValueError("flex_ep topk_idx batch dim must match x")
    if topk_idx.shape[1] != topk:
        raise ValueError("flex_ep topk_idx second dim must match topk")
    if topk_idx.dtype not in (torch.int32, torch.int64):
        raise ValueError("flex_ep topk_idx must be int32 or int64")
    if (
        x.dtype != torch.bfloat16
        or w13.dtype != torch.bfloat16
        or w2.dtype != torch.bfloat16
    ):
        raise ValueError("flex_ep v1 supports BF16 x, w13, and w2 only")
    if w13.dim() != 3 or w2.dim() != 3:
        raise ValueError("flex_ep expects w13 and w2 to be 3D expert weights")
    if w13.shape[0] != w2.shape[0]:
        raise ValueError("flex_ep w13 and w2 local expert dims must match")
    if w13.shape[2] != x.shape[1] or w2.shape[1] != x.shape[1]:
        raise ValueError("flex_ep expert weight hidden dims must match x")
    if w13.shape[1] != 2 * w2.shape[2]:
        raise ValueError("flex_ep w13 intermediate dim must be twice w2 input dim")
    if num_experts < 1:
        raise ValueError(f"flex_ep expects num_experts >= 1, got {num_experts}")
    if ep_size < 1:
        raise ValueError(f"flex_ep expects ep_size >= 1, got {ep_size}")
    if ep_rank < 0 or ep_rank >= ep_size:
        raise ValueError(f"flex_ep ep_rank must be in [0, {ep_size}), got {ep_rank}")
    if max_tokens < x.shape[0]:
        raise ValueError("flex_ep max_tokens must be at least the batch size")
    if topk < 1:
        raise ValueError(f"flex_ep expects topk >= 1, got {topk}")
    if num_ctas < 1:
        raise ValueError(f"flex_ep expects num_ctas >= 1, got {num_ctas}")


def _as_tuple(x: Any) -> tuple[Any, ...]:
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return (x,)


def _expand_x_for_topk(x: torch.Tensor, topk: int) -> torch.Tensor:
    return x.unsqueeze(1).expand(-1, topk, -1).contiguous()


def _grouped_mm_offsets(plan: Any) -> torch.Tensor:
    plan = validate_dispatch_plan(plan)
    return plan.local_experts_start[1:].to(torch.int32)


def _token_end_from_plan(plan: Any) -> torch.Tensor:
    plan = validate_dispatch_plan(plan)
    return plan.local_experts_start[-1:].to(torch.int64)


def _swiglu_reference(y1: torch.Tensor) -> torch.Tensor:
    gate, up = y1.chunk(2, dim=-1)
    return F.silu(gate) * up


def _swiglu_backward_reference(
    dy2: torch.Tensor,
    y1: torch.Tensor,
) -> torch.Tensor:
    gate, up = y1.chunk(2, dim=-1)
    sig = torch.sigmoid(gate)
    dgate = dy2 * up * sig * (1 + gate * (1 - sig))
    dup = dy2 * F.silu(gate)
    return torch.cat((dgate, dup), dim=-1)


def _swiglu_with_offsets(
    y1: torch.Tensor,
    token_end: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._flex_ep.swiglu_forward_with_offsets(y1, token_end)


def _swiglu_backward_with_offsets(
    dy2: torch.Tensor,
    y1: torch.Tensor,
    token_end: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._flex_ep.swiglu_backward_with_offsets(dy2, y1, token_end)


def _clone_valid_prefix(
    input: torch.Tensor,
    token_end: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._flex_ep.clone_valid_prefix(input, token_end)


def _moe_block_forward_bf16(
    recv_x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    offs: torch.Tensor,
    token_end: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y1 = torch._grouped_mm(recv_x, w13.transpose(-2, -1), offs=offs)
    y2 = _swiglu_with_offsets(y1, token_end)
    y3 = torch._grouped_mm(y2, w2.transpose(-2, -1), offs=offs)
    return y3, y1, y2


def _moe_block_backward_bf16(
    dy3: torch.Tensor,
    recv_x: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    offs: torch.Tensor,
    token_end: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dy2 = torch._grouped_mm(dy3, w2, offs=offs)
    dw2_t = torch._grouped_mm(y2.transpose(-2, -1), dy3, offs=offs)
    dy1 = _swiglu_backward_with_offsets(dy2, y1, token_end)
    dx_recv = torch._grouped_mm(dy1, w13, offs=offs)
    dw13_t = torch._grouped_mm(recv_x.transpose(-2, -1), dy1, offs=offs)
    return dx_recv, dw13_t.transpose(-2, -1), dw2_t.transpose(-2, -1)


def _flex_ep_forward(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    build_dispatch_plan_fn: Callable[..., Any],
    dispatch_fn: Callable[..., Any],
    combine_fn: Callable[..., Any],
    router_operands: Any,
    *,
    topk: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Any,
    torch.Tensor,
]:
    x_expanded = _expand_x_for_topk(x, topk)
    plan = validate_dispatch_plan(build_dispatch_plan_fn(topk_idx, router_operands))
    dispatch_out = _as_tuple(dispatch_fn(x_expanded, plan, router_operands))
    if len(dispatch_out) != 1:
        raise RuntimeError(
            "flex_ep dispatch_fn must return recv_x or a single-item tuple/list"
        )
    recv_x = dispatch_out[0]
    if not isinstance(recv_x, torch.Tensor):
        raise RuntimeError("flex_ep dispatch_fn first return must be recv_x tensor")
    offs = _grouped_mm_offsets(plan)
    token_end = _token_end_from_plan(plan)
    y3, y1, y2 = _moe_block_forward_bf16(recv_x, w13, w2, offs, token_end)
    y = combine_fn(y3, plan, router_operands)
    if not isinstance(y, torch.Tensor):
        raise RuntimeError("flex_ep combine_fn must return a tensor")
    return y, recv_x, y1, y2, plan, offs


def _flex_ep_backward_impl(
    dy: torch.Tensor,
    recv_x: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    offs: torch.Tensor,
    token_end: torch.Tensor,
    combine_bwd_fn: Callable[..., Any],
    dispatch_bwd_fn: Callable[..., Any],
    plan: Any,
    router_operands: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    plan = validate_dispatch_plan(plan)
    dy3 = combine_bwd_fn(dy, plan, router_operands)
    if not isinstance(dy3, torch.Tensor):
        raise RuntimeError("flex_ep combine_bwd_fn must return a tensor")
    dx_recv, dw13, dw2 = _moe_block_backward_bf16(
        dy3,
        recv_x,
        y1,
        y2,
        w13,
        w2,
        offs,
        token_end,
    )
    dxpn = dispatch_bwd_fn(dx_recv, plan, router_operands)
    if not isinstance(dxpn, torch.Tensor):
        raise RuntimeError("flex_ep dispatch_bwd_fn must return a tensor")
    return dxpn.sum(-2), dw13, dw2


class _FlexEpAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        build_dispatch_plan_fn: Callable[..., Any],
        dispatch_fn: Callable[..., Any],
        combine_fn: Callable[..., Any],
        combine_bwd_fn: Callable[..., Any],
        dispatch_bwd_fn: Callable[..., Any],
        router_operands: Any,
        topk: int,
    ) -> torch.Tensor:
        with torch.no_grad():
            y, recv_x, y1, y2, plan, offs = _flex_ep_forward(
                x,
                topk_idx,
                w13,
                w2,
                build_dispatch_plan_fn,
                dispatch_fn,
                combine_fn,
                router_operands,
                topk=topk,
            )
        ctx.dispatch_bwd_fn = dispatch_bwd_fn
        ctx.combine_bwd_fn = combine_bwd_fn
        ctx.router_operands = router_operands
        ctx.plan = plan
        token_end = _token_end_from_plan(plan)
        ctx.save_for_backward(
            _clone_valid_prefix(recv_x, token_end),
            y1,
            y2,
            w13,
            w2,
            offs,
            token_end,
        )
        return y

    @staticmethod
    def backward(ctx, *grad_outputs: Any) -> Any:
        (dy,) = grad_outputs
        recv_x, y1, y2, w13, w2, offs, token_end = ctx.saved_tensors
        plan = ctx.plan
        router_operands = ctx.router_operands
        dx, dw13, dw2 = flex_ep_backward(
            dy,
            recv_x,
            y1,
            y2,
            w13,
            w2,
            offs,
            token_end,
            ctx.combine_bwd_fn,
            ctx.dispatch_bwd_fn,
            plan,
            router_operands,
        )
        return (
            dx,
            None,
            dw13,
            dw2,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _flex_ep_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    build_dispatch_plan_fn: Callable[..., Any],
    dispatch_fn: Callable[..., Any],
    combine_fn: Callable[..., Any],
    combine_bwd_fn: Callable[..., Any],
    dispatch_bwd_fn: Callable[..., Any],
    router_operands: Any,
    num_experts: int,
    ep_rank: int,
    ep_size: int,
    max_tokens: int,
    topk: int,
    num_ctas: int = 152,
) -> torch.Tensor:
    _validate_flex_ep_inputs(
        x,
        topk_idx,
        w13,
        w2,
        router_operands,
        num_experts=num_experts,
        ep_rank=ep_rank,
        ep_size=ep_size,
        max_tokens=max_tokens,
        topk=topk,
        num_ctas=num_ctas,
    )
    return _FlexEpAutogradFunction.apply(
        x,
        topk_idx,
        w13,
        w2,
        build_dispatch_plan_fn,
        dispatch_fn,
        combine_fn,
        combine_bwd_fn,
        dispatch_bwd_fn,
        router_operands,
        topk,
    )


class FlexEpHOP(HigherOrderOperator):
    """Autograd-aware BF16 expert-parallel MoE block.

    This higher order operator expands each local token across its selected
    experts, dispatches those token/expert pairs with ``dispatch_fn``, evaluates
    the local expert MLP with grouped matmuls and fused SwiGLU, then combines the
    expert outputs with ``combine_fn``. TorchTitan's FlexEP backend returns
    unweighted expert outputs of shape ``[num_tokens, topk, hidden]``; callers
    apply the top-k weights outside this HOP.

    Args:
        x: Local token activations with shape ``[num_tokens, hidden]`` and BF16
            dtype.
        topk_idx: Global expert id for each selected expert, with shape
            ``[num_tokens, topk]``.
        w13: Local expert gate/up projection weights with shape
            ``[num_local_experts, 2 * intermediate, hidden]`` and BF16 dtype.
        w2: Local expert down projection weights with shape
            ``[num_local_experts, hidden, intermediate]`` and BF16 dtype.
        build_dispatch_plan_fn: Router subgraph called as
            ``build_dispatch_plan_fn(topk_idx, router_operands)``. It must
            return a ``FlexEPDispatchPlan``-compatible dataclass.
        dispatch_fn: Router subgraph called as
            ``dispatch_fn(x_expanded, plan, router_operands)``. It must return
            ``recv_x``.
        combine_fn: Router subgraph called as
            ``combine_fn(y3, plan, router_operands)``. It must return the HOP
            output tensor.
        combine_bwd_fn: Backward router subgraph called as
            ``combine_bwd_fn(dy, plan, router_operands)``. It must return the
            gradient for the local expert output ``y3``.
        dispatch_bwd_fn: Backward router subgraph called as
            ``dispatch_bwd_fn(dx_recv, plan, router_operands)``. It must
            return expanded input gradients with shape
            ``[num_tokens, topk, hidden]``.
        router_operands: Python object captured by the router subgraphs, such
            as a dataclass containing workspace tensors, offsets, and EP rank
            metadata. All tensors needed by the subgraphs must flow through
            here; closure-captured tensors are not supported.
        num_experts: Total number of global experts.
        ep_rank: This rank within the expert-parallel group.
        ep_size: Number of ranks in the expert-parallel group.
        max_tokens: Maximum local token count used for router buffer sizing.
        topk: Number of selected experts per token.
        num_ctas: CTA count used by backend router kernels.
    """

    def __init__(self) -> None:
        super().__init__("flex_ep", cacheable=True)
        self.subgraph_indexes = [4, 5, 6, 7, 8]

    def __call__(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        build_dispatch_plan_fn: Callable[..., Any],
        dispatch_fn: Callable[..., Any],
        combine_fn: Callable[..., Any],
        combine_bwd_fn: Callable[..., Any],
        dispatch_bwd_fn: Callable[..., Any],
        router_operands: Any,
        num_experts: int,
        ep_rank: int,
        ep_size: int,
        max_tokens: int,
        topk: int,
        num_ctas: int = 152,
    ) -> torch.Tensor:
        return super().__call__(
            x,
            topk_idx,
            w13,
            w2,
            build_dispatch_plan_fn,
            dispatch_fn,
            combine_fn,
            combine_bwd_fn,
            dispatch_bwd_fn,
            router_operands,
            num_experts=num_experts,
            ep_rank=ep_rank,
            ep_size=ep_size,
            max_tokens=max_tokens,
            topk=topk,
            num_ctas=num_ctas,
        )


flex_ep = FlexEpHOP()


class FlexEpBackwardHOP(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flex_ep_backward", cacheable=True)
        self.subgraph_indexes = [8, 9]

    def __call__(
        self,
        dy: torch.Tensor,
        recv_x: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        offs: torch.Tensor,
        token_end: torch.Tensor,
        combine_bwd_fn: Callable[..., Any],
        dispatch_bwd_fn: Callable[..., Any],
        plan: Any,
        router_operands: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().__call__(
            dy,
            recv_x,
            y1,
            y2,
            w13,
            w2,
            offs,
            token_end,
            combine_bwd_fn,
            dispatch_bwd_fn,
            plan,
            router_operands,
        )


flex_ep_backward = FlexEpBackwardHOP()


@flex_ep_backward.py_impl(DispatchKey.CompositeExplicitAutograd)
def flex_ep_backward_dense(
    dy: torch.Tensor,
    recv_x: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    offs: torch.Tensor,
    token_end: torch.Tensor,
    combine_bwd_fn: Callable[..., Any],
    dispatch_bwd_fn: Callable[..., Any],
    plan: Any,
    router_operands: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _flex_ep_backward_impl(
        dy,
        recv_x,
        y1,
        y2,
        w13,
        w2,
        offs,
        token_end,
        combine_bwd_fn,
        dispatch_bwd_fn,
        plan,
        router_operands,
    )


def _trace_flex_ep_backward_proxy(
    mode: ProxyTorchDispatchMode,
    dy: torch.Tensor,
    recv_x: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    offs: torch.Tensor,
    token_end: torch.Tensor,
    combine_bwd_fn: Callable[..., Any],
    dispatch_bwd_fn: Callable[..., Any],
    plan: Any,
    router_operands: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    plan = validate_dispatch_plan(plan)
    if not isinstance(combine_bwd_fn, torch.fx.GraphModule):
        combine_bwd_fn = _maybe_reenter_make_fx(combine_bwd_fn)(
            dy,
            plan,
            router_operands,
        )
    if not isinstance(dispatch_bwd_fn, torch.fx.GraphModule):
        dispatch_bwd_fn = _maybe_reenter_make_fx(dispatch_bwd_fn)(
            recv_x,
            plan,
            router_operands,
        )
    example_out = flex_ep_backward(
        dy,
        recv_x,
        y1,
        y2,
        w13,
        w2,
        offs,
        token_end,
        combine_bwd_fn,
        dispatch_bwd_fn,
        plan,
        router_operands,
    )
    if not isinstance(mode.tracer, torch.fx.Tracer):
        raise AssertionError(
            f"expected proxy_mode.tracer to be torch.fx.Tracer, got {type(mode.tracer)}"
        )
    for name, fn in (
        ("flex_ep_combine_bwd", combine_bwd_fn),
        ("flex_ep_dispatch_bwd", dispatch_bwd_fn),
    ):
        if isinstance(fn, torch.fx.GraphModule):
            qualname = mode.tracer.get_fresh_qualname(name)
            mode.tracer.root.register_module(qualname, fn)
    node_args = (
        dy,
        recv_x,
        y1,
        y2,
        w13,
        w2,
        offs,
        token_end,
        combine_bwd_fn,
        dispatch_bwd_fn,
        plan,
        router_operands,
    )
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, node_args)  # type: ignore[attr-defined]
    with torch.fx.experimental.proxy_tensor.set_original_aten_op(flex_ep_backward):
        out_proxy = mode.tracer.create_proxy(
            "call_function",
            flex_ep_backward,
            proxy_args,
            {},
        )
    return track_tensor_tree(
        example_out,
        out_proxy,
        constant=None,
        tracer=mode.tracer,
    )


@flex_ep_backward.py_impl(ProxyTorchDispatchMode)
def flex_ep_backward_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    dy: torch.Tensor,
    recv_x: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    offs: torch.Tensor,
    token_end: torch.Tensor,
    combine_bwd_fn: Callable[..., Any],
    dispatch_bwd_fn: Callable[..., Any],
    plan: Any,
    router_operands: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _trace_flex_ep_backward_proxy(
        mode,
        dy,
        recv_x,
        y1,
        y2,
        w13,
        w2,
        offs,
        token_end,
        combine_bwd_fn,
        dispatch_bwd_fn,
        plan,
        router_operands,
    )


@flex_ep_backward.py_impl(FakeTensorMode)
def flex_ep_backward_fake_tensor_mode(
    mode: FakeTensorMode,
    dy: torch.Tensor,
    recv_x: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    offs: torch.Tensor,
    token_end: torch.Tensor,
    combine_bwd_fn: Callable[..., Any],
    dispatch_bwd_fn: Callable[..., Any],
    plan: Any,
    router_operands: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with mode:
        return _flex_ep_backward_impl(
            dy,
            recv_x,
            y1,
            y2,
            w13,
            w2,
            offs,
            token_end,
            combine_bwd_fn,
            dispatch_bwd_fn,
            plan,
            router_operands,
        )


@flex_ep_backward.py_functionalize_impl
def flex_ep_backward_functionalize(
    ctx,
    dy: torch.Tensor,
    recv_x: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    offs: torch.Tensor,
    token_end: torch.Tensor,
    combine_bwd_fn: Callable[..., Any],
    dispatch_bwd_fn: Callable[..., Any],
    plan: Any,
    router_operands: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    unwrapped = pytree.tree_map_only(
        torch.Tensor,
        ctx.unwrap_tensors,
        (
            dy,
            recv_x,
            y1,
            y2,
            w13,
            w2,
            offs,
            token_end,
            plan,
            router_operands,
        ),
    )
    (
        dy,
        recv_x,
        y1,
        y2,
        w13,
        w2,
        offs,
        token_end,
        plan,
        router_operands,
    ) = unwrapped
    with ctx.redispatch_to_next():
        with suspend_functionalization(), disable_proxy_modes_tracing():
            functional_combine_bwd_fn = ctx.functionalize(combine_bwd_fn)
            functional_dispatch_bwd_fn = ctx.functionalize(dispatch_bwd_fn)
        out = flex_ep_backward(
            dy,
            recv_x,
            y1,
            y2,
            w13,
            w2,
            offs,
            token_end,
            functional_combine_bwd_fn,
            functional_dispatch_bwd_fn,
            plan,
            router_operands,
        )
    return ctx.wrap_tensors(out)


flex_ep_backward.py_autograd_impl(
    autograd_not_implemented(flex_ep_backward, deferred_error=True)
)


@flex_ep.py_impl(DispatchKey.CompositeExplicitAutograd)
def flex_ep_dense(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    build_dispatch_plan_fn: Callable[..., Any],
    dispatch_fn: Callable[..., Any],
    combine_fn: Callable[..., Any],
    combine_bwd_fn: Callable[..., Any],
    dispatch_bwd_fn: Callable[..., Any],
    router_operands: Any,
    num_experts: int,
    ep_rank: int,
    ep_size: int,
    max_tokens: int,
    topk: int,
    num_ctas: int = 152,
) -> torch.Tensor:
    return _flex_ep_impl(
        x,
        topk_idx,
        w13,
        w2,
        build_dispatch_plan_fn,
        dispatch_fn,
        combine_fn,
        combine_bwd_fn,
        dispatch_bwd_fn,
        router_operands,
        num_experts=num_experts,
        ep_rank=ep_rank,
        ep_size=ep_size,
        max_tokens=max_tokens,
        topk=topk,
        num_ctas=num_ctas,
    )


@flex_ep.py_impl(DispatchKey.Autograd)
def flex_ep_autograd(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    build_dispatch_plan_fn: Callable[..., Any],
    dispatch_fn: Callable[..., Any],
    combine_fn: Callable[..., Any],
    combine_bwd_fn: Callable[..., Any],
    dispatch_bwd_fn: Callable[..., Any],
    router_operands: Any,
    num_experts: int,
    ep_rank: int,
    ep_size: int,
    max_tokens: int,
    topk: int,
    num_ctas: int = 152,
) -> torch.Tensor:
    return _flex_ep_impl(
        x,
        topk_idx,
        w13,
        w2,
        build_dispatch_plan_fn,
        dispatch_fn,
        combine_fn,
        combine_bwd_fn,
        dispatch_bwd_fn,
        router_operands,
        num_experts=num_experts,
        ep_rank=ep_rank,
        ep_size=ep_size,
        max_tokens=max_tokens,
        topk=topk,
        num_ctas=num_ctas,
    )


def _trace_flex_ep_proxy(
    mode: ProxyTorchDispatchMode,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    build_dispatch_plan_fn: Callable[..., Any],
    dispatch_fn: Callable[..., Any],
    combine_fn: Callable[..., Any],
    combine_bwd_fn: Callable[..., Any],
    dispatch_bwd_fn: Callable[..., Any],
    router_operands: Any,
    kwargs: dict[str, Any],
) -> torch.Tensor:
    example_out = flex_ep(
        x,
        topk_idx,
        w13,
        w2,
        build_dispatch_plan_fn,
        dispatch_fn,
        combine_fn,
        combine_bwd_fn,
        dispatch_bwd_fn,
        router_operands,
        **kwargs,
    )
    if not isinstance(mode.tracer, torch.fx.Tracer):
        raise AssertionError(
            f"expected proxy_mode.tracer to be torch.fx.Tracer, got {type(mode.tracer)}"
        )
    for name, fn in (
        ("flex_ep_build_dispatch_plan", build_dispatch_plan_fn),
        ("flex_ep_dispatch", dispatch_fn),
        ("flex_ep_combine", combine_fn),
        ("flex_ep_combine_bwd", combine_bwd_fn),
        ("flex_ep_dispatch_bwd", dispatch_bwd_fn),
    ):
        if isinstance(fn, torch.fx.GraphModule):
            qualname = mode.tracer.get_fresh_qualname(name)
            mode.tracer.root.register_module(qualname, fn)
    node_args = (
        x,
        topk_idx,
        w13,
        w2,
        build_dispatch_plan_fn,
        dispatch_fn,
        combine_fn,
        combine_bwd_fn,
        dispatch_bwd_fn,
        router_operands,
    )
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, node_args)  # type: ignore[attr-defined]
    proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)  # type: ignore[attr-defined]
    with torch.fx.experimental.proxy_tensor.set_original_aten_op(flex_ep):
        out_proxy = mode.tracer.create_proxy(
            "call_function",
            flex_ep,
            proxy_args,
            proxy_kwargs,
        )
    return track_tensor_tree(
        example_out,
        out_proxy,
        constant=None,
        tracer=mode.tracer,
    )


@flex_ep.py_impl(ProxyTorchDispatchMode)
def flex_ep_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    build_dispatch_plan_fn: Callable[..., Any],
    dispatch_fn: Callable[..., Any],
    combine_fn: Callable[..., Any],
    combine_bwd_fn: Callable[..., Any],
    dispatch_bwd_fn: Callable[..., Any],
    router_operands: Any,
    num_experts: int,
    ep_rank: int,
    ep_size: int,
    max_tokens: int,
    topk: int,
    num_ctas: int = 152,
) -> torch.Tensor:
    return _trace_flex_ep_proxy(
        mode,
        x,
        topk_idx,
        w13,
        w2,
        build_dispatch_plan_fn,
        dispatch_fn,
        combine_fn,
        combine_bwd_fn,
        dispatch_bwd_fn,
        router_operands,
        {
            "num_experts": num_experts,
            "ep_rank": ep_rank,
            "ep_size": ep_size,
            "max_tokens": max_tokens,
            "topk": topk,
            "num_ctas": num_ctas,
        },
    )


@flex_ep.py_impl(FakeTensorMode)
def flex_ep_fake_tensor_mode(
    mode: FakeTensorMode,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    build_dispatch_plan_fn: Callable[..., Any],
    dispatch_fn: Callable[..., Any],
    combine_fn: Callable[..., Any],
    combine_bwd_fn: Callable[..., Any],
    dispatch_bwd_fn: Callable[..., Any],
    router_operands: Any,
    num_experts: int,
    ep_rank: int,
    ep_size: int,
    max_tokens: int,
    topk: int,
    num_ctas: int = 152,
) -> torch.Tensor:
    with mode:
        _validate_flex_ep_inputs(
            x,
            topk_idx,
            w13,
            w2,
            router_operands,
            num_experts=num_experts,
            ep_rank=ep_rank,
            ep_size=ep_size,
            max_tokens=max_tokens,
            topk=topk,
            num_ctas=num_ctas,
        )
        y, _, _, _, _, _ = _flex_ep_forward(
            x,
            topk_idx,
            w13,
            w2,
            build_dispatch_plan_fn,
            dispatch_fn,
            combine_fn,
            router_operands,
            topk=topk,
        )
        return y


@flex_ep.py_functionalize_impl
def flex_ep_functionalize(
    ctx,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    build_dispatch_plan_fn: Callable[..., Any],
    dispatch_fn: Callable[..., Any],
    combine_fn: Callable[..., Any],
    combine_bwd_fn: Callable[..., Any],
    dispatch_bwd_fn: Callable[..., Any],
    router_operands: Any,
    num_experts: int,
    ep_rank: int,
    ep_size: int,
    max_tokens: int,
    topk: int,
    num_ctas: int = 152,
) -> torch.Tensor:
    unwrapped = pytree.tree_map_only(
        torch.Tensor,
        ctx.unwrap_tensors,
        (
            x,
            topk_idx,
            w13,
            w2,
            router_operands,
        ),
    )
    (
        x,
        topk_idx,
        w13,
        w2,
        router_operands,
    ) = unwrapped
    with ctx.redispatch_to_next():
        with suspend_functionalization(), disable_proxy_modes_tracing():
            functional_build_dispatch_plan_fn = ctx.functionalize(
                build_dispatch_plan_fn
            )
            functional_dispatch_fn = ctx.functionalize(dispatch_fn)
            functional_combine_fn = ctx.functionalize(combine_fn)
            functional_combine_bwd_fn = ctx.functionalize(combine_bwd_fn)
            functional_dispatch_bwd_fn = ctx.functionalize(dispatch_bwd_fn)
        out = flex_ep(
            x,
            topk_idx,
            w13,
            w2,
            functional_build_dispatch_plan_fn,
            functional_dispatch_fn,
            functional_combine_fn,
            functional_combine_bwd_fn,
            functional_dispatch_bwd_fn,
            router_operands,
            num_experts=num_experts,
            ep_rank=ep_rank,
            ep_size=ep_size,
            max_tokens=max_tokens,
            topk=topk,
            num_ctas=num_ctas,
        )
    return ctx.wrap_tensors(out)


_flex_ep_lib = torch.library.Library("_flex_ep", "DEF")
_flex_ep_lib.define(
    "ep_allgather(Tensor output, Tensor input, Tensor buffers_cuda_ptrs, "
    "int offs_output, int ep_rank) -> Tensor"
)
_flex_ep_lib.define(
    "router_compute_all_expert_offsets(Tensor all_expert_counts, int ep_rank, "
    "int local_experts, int token_alignment) -> (Tensor, Tensor, Tensor)"
)
_flex_ep_lib.define(
    "router_compute_dest_offsets(Tensor topk_idx, Tensor recv_ofs, int ep_size) "
    "-> (Tensor, Tensor)"
)
_flex_ep_lib.define(
    "router_dispatch(Tensor my_tokens, Tensor? my_scaling_factors, "
    "Tensor? my_topk_weights, Tensor dest_ranks, Tensor dest_offsets, "
    "Tensor buffers_cuda_ptrs, Tensor dispatch_recv_buffer, "
    "Tensor dispatch_recv_buffer_scaling_factors, "
    "Tensor dispatch_recv_origin_global_token_id, "
    "Tensor dispatch_recv_weights, int offs_recv_tokens, "
    "int offs_recv_scaling_factors, int offs_recv_weights, "
    "int offs_recv_origin_global_token_id, int ep_rank, int num_ctas, "
    "int max_B) -> (Tensor, Tensor, Tensor, Tensor)"
)
_flex_ep_lib.define(
    "router_combine(Tensor send_tokens, Tensor? send_scale_factors, "
    "Tensor? send_weights, Tensor expert_begin_offset_per_ep, "
    "Tensor token_send_end, Tensor send_origin_global_token_id, "
    "Tensor buffers_cuda_ptrs, Tensor combine_recv_buffer, "
    "Tensor combine_recv_scale_factors, Tensor combine_recv_weights, "
    "int offs_combine_recv_tokens, int offs_combine_recv_scale_factors, "
    "int offs_combine_recv_weights, int ep_rank, int B, int topk, "
    "int num_ctas, int max_B) -> (Tensor, Tensor, Tensor)"
)
_flex_ep_lib.define(
    "barrier_arrive(Tensor flag, Tensor dependency, int nonce=0) -> Tensor"
)
_flex_ep_lib.define(
    "barrier_wait(Tensor input, Tensor cuda_ptrs, int offs_flag, Tensor expected, "
    "float timeout_s=5.0) -> Tensor"
)
_flex_ep_lib.define(
    "barrier_wait_no_clone(Tensor(a) input, Tensor cuda_ptrs, int offs_flag, "
    "Tensor expected, float timeout_s=5.0) -> Tensor(a)"
)
_flex_ep_lib.define(
    "swiglu_forward_with_offsets(Tensor y1, Tensor token_end) -> Tensor"
)
_flex_ep_lib.define(
    "swiglu_backward_with_offsets(Tensor dy2, Tensor y1, Tensor token_end) -> Tensor"
)
_flex_ep_lib.define("clone_valid_prefix(Tensor input, Tensor token_end) -> Tensor")
_flex_ep_lib.define(
    "zfill_ranges_inplace(Tensor input, Tensor begin_ofs, Tensor end_ofs, "
    "int max_values_per_batch) -> Tensor"
)
_flex_ep_lib.define("fill_i64_inplace(Tensor input, int value) -> Tensor")

for _name in (
    "ep_allgather",
    "router_dispatch",
    "router_combine",
    "barrier_arrive",
    "barrier_wait",
    "barrier_wait_no_clone",
    "zfill_ranges_inplace",
    "fill_i64_inplace",
):
    has_side_effect(getattr(torch.ops._flex_ep, _name).default)
del _name


MAX_TLX_NUM_CTAS = 152
TLX_NUM_STAGES = 8

_REQUIRED_FLEX_EP_BACKEND_OPS = (
    "flex_ep_allgather",
    "flex_ep_router_compute_all_expert_offsets",
    "flex_ep_router_compute_dest_offsets",
    "flex_ep_router_dispatch",
    "flex_ep_router_combine",
    "flex_ep_barrier_arrive",
    "flex_ep_barrier_wait",
    "flex_ep_swiglu_forward",
    "flex_ep_swiglu_backward",
    "flex_ep_swiglu_forward_with_offsets",
    "flex_ep_swiglu_backward_with_offsets",
    "flex_ep_clone_valid_prefix",
    "flex_ep_weighted_sum_forward",
    "flex_ep_weighted_sum_backward",
    "flex_ep_zfill_ranges_inplace",
)


def _missing_flex_ep_backend_ops() -> list[str]:
    return [
        op_name
        for op_name in _REQUIRED_FLEX_EP_BACKEND_OPS
        if not hasattr(torch.ops.inductor, op_name)
    ]


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def register_flex_ep_backend_ops() -> None:
    if not _missing_flex_ep_backend_ops():
        return

    try:
        import torch.distributed._symmetric_memory._shmem_triton as shmem_triton
        import triton
        import triton.language as tl

        from torch.distributed._symmetric_memory._shmem_triton import requires_shmem
    except ImportError as e:
        raise ImportError(
            "flex_ep backend requires Triton symmetric-memory kernels."
        ) from e

    try:
        tlx = cast(Any, importlib.import_module("triton.language.extra.tlx"))
    except ImportError:
        tlx = None

    shmem_backend = shmem_triton.get_shmem_backend_module()

    @triton.jit
    def _barrier_arrive_kernel(flag_ptr, out_ptr):
        old = tl.atomic_add(
            flag_ptr,
            tl.full([], 1, dtype=tl.int32),
            sem="release",
            scope="sys",
        )
        tl.store(out_ptr, old + 1)

    @triton.jit
    def _barrier_wait_kernel(
        cuda_ptrs_ptr,
        expected_ptr,
        n_flags,
        offs_flag,
        timeout_ns,
        BLOCK: tl.constexpr,
    ):
        offs = tl.arange(0, BLOCK)
        mask = offs < n_flags
        expected = tl.load(expected_ptr)
        base_ptrs = tl.load(cuda_ptrs_ptr + offs, mask=mask, other=0)
        flag_ptrs = (base_ptrs + offs_flag).to(tl.pointer_type(tl.int32))
        zero = tl.zeros([BLOCK], dtype=tl.int32)
        start = tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;",
            "=l",
            [],
            dtype=tl.int64,
            is_pure=False,
            pack=1,
        )
        done = tl.zeros([], dtype=tl.int32)
        while done == 0:
            vals = tl.atomic_add(
                flag_ptrs,
                zero,
                mask=mask,
                sem="acquire",
                scope="sys",
            )
            waiting = tl.sum(tl.where(mask & (vals < expected), 1, 0))
            if waiting == 0:
                done = tl.full([], 1, dtype=tl.int32)
            else:
                now = tl.inline_asm_elementwise(
                    "mov.u64 $0, %globaltimer;",
                    "=l",
                    [],
                    dtype=tl.int64,
                    is_pure=False,
                    pack=1,
                )
                if (now - start) > timeout_ns:  # pyrefly: ignore[unsupported-operation]
                    tl.device_print("flex_ep barrier_wait timed out")
                    tl.inline_asm_elementwise(
                        "trap;",
                        "=r",
                        [],
                        dtype=tl.int32,
                        is_pure=False,
                        pack=1,
                    )
                    done = tl.full([], 1, dtype=tl.int32)

    @requires_shmem
    @triton.jit
    def _ep_allgather_kernel(
        input_ptr,
        buffers_cuda_ptrs_ptr,
        offs_output,
        ep_rank,
        input_num_bytes: tl.constexpr,
    ):
        peer = tl.program_id(0)
        base_ptr = tl.load(buffers_cuda_ptrs_ptr + ep_rank)
        dst_ptr = (base_ptr + offs_output + ep_rank * input_num_bytes).to(
            tl.pointer_type(tl.uint8)
        )
        shmem_backend.put(dst_ptr, input_ptr, input_num_bytes, peer)
        shmem_backend.quiet()

    @triton.jit
    def _compute_expert_offsets_kernel(
        counts_ptr,
        offsets_ptr,
        expert_start_ptr,
        grand_total_ptr,
        LOCAL_EXPERTS: tl.constexpr,
        EP_SIZE: tl.constexpr,
        TOKEN_ALIGNMENT: tl.constexpr,
    ):
        dest = tl.program_id(0)
        num_experts = EP_SIZE * LOCAL_EXPERTS

        src_idx = tl.arange(0, EP_SIZE)
        expert_idx = tl.arange(0, LOCAL_EXPERTS)
        counts = tl.load(
            counts_ptr
            + src_idx[:, None] * num_experts
            + dest * LOCAL_EXPERTS
            + expert_idx[None, :]
        )

        total_per_expert = tl.sum(counts, axis=0)
        tl.store(grand_total_ptr + dest, tl.sum(total_per_expert))

        aligned = (
            (total_per_expert + TOKEN_ALIGNMENT - 1) // TOKEN_ALIGNMENT
        ) * TOKEN_ALIGNMENT
        aligned_inclusive = tl.cumsum(aligned, axis=0)
        aligned_exclusive = aligned_inclusive - aligned

        tl.store(
            expert_start_ptr + dest * (LOCAL_EXPERTS + 1) + expert_idx,
            aligned_exclusive.to(tl.int32),
        )
        tl.store(
            expert_start_ptr + dest * (LOCAL_EXPERTS + 1) + LOCAL_EXPERTS,
            tl.sum(aligned).to(tl.int32),
        )

        counts_inclusive = tl.cumsum(counts, axis=0)
        tl.store(
            offsets_ptr
            + dest * (LOCAL_EXPERTS * (EP_SIZE + 1))
            + expert_idx * (EP_SIZE + 1),
            aligned_exclusive.to(tl.int32),
        )
        tl.store(
            offsets_ptr
            + dest * (LOCAL_EXPERTS * (EP_SIZE + 1))
            + expert_idx[None, :] * (EP_SIZE + 1)
            + (src_idx[:, None] + 1),
            (aligned_exclusive[None, :] + counts_inclusive).to(tl.int32),
        )

    @triton.jit
    def _compute_dest_offsets_kernel(
        topk_idx_ptr,
        recv_ofs_ptr,
        recv_ofs_stride,
        dest_offsets_ptr,
        batch,
        TOPK: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        expert = tl.program_id(0)
        flat_offsets = tl.arange(0, BLOCK)
        total = batch * TOPK
        base = tl.load(recv_ofs_ptr + expert * recv_ofs_stride).to(tl.int64)
        count = tl.zeros([], dtype=tl.int64)

        for flat_begin in range(0, total, BLOCK):
            flat_idx = flat_begin + flat_offsets
            valid = flat_idx < total
            routed_expert = tl.load(topk_idx_ptr + flat_idx, mask=valid, other=-1)
            is_mine = (routed_expert == expert) & valid
            match = is_mine.to(tl.int64)  # pyrefly: ignore[missing-attribute]
            inclusive = tl.cumsum(match, axis=0)
            exclusive = inclusive - match

            tl.store(
                dest_offsets_ptr + flat_idx,
                base + count + exclusive,
                mask=is_mine,
            )
            count += tl.sum(match)

    @triton.jit
    def _swiglu_forward_kernel(
        y1_ptr,
        y2_ptr,
        total,
        hidden_dim: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total
        col = offs % hidden_dim
        row = offs // hidden_dim
        gate = tl.load(y1_ptr + row * hidden_dim * 2 + col, mask=mask).to(tl.float32)
        up = tl.load(
            y1_ptr + row * hidden_dim * 2 + hidden_dim + col,
            mask=mask,
        ).to(tl.float32)
        y2 = gate * tl.sigmoid(gate) * up
        tl.store(y2_ptr + offs, y2, mask=mask)

    @triton.jit
    def _swiglu_backward_kernel(
        dy2_ptr,
        y1_ptr,
        dy1_ptr,
        total,
        hidden_dim: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total
        col = offs % hidden_dim
        row = offs // hidden_dim
        y1_base = row * hidden_dim * 2
        gate = tl.load(y1_ptr + y1_base + col, mask=mask).to(tl.float32)
        up = tl.load(y1_ptr + y1_base + hidden_dim + col, mask=mask).to(tl.float32)
        dy2 = tl.load(dy2_ptr + offs, mask=mask).to(tl.float32)
        sig = tl.sigmoid(gate)
        silu_gate = gate * sig
        dgate = dy2 * up * sig * (1.0 + gate * (1.0 - sig))
        dup = dy2 * silu_gate
        tl.store(dy1_ptr + y1_base + col, dgate, mask=mask)
        tl.store(dy1_ptr + y1_base + hidden_dim + col, dup, mask=mask)

    @triton.jit
    def _swiglu_forward_with_offsets_kernel(
        y1_ptr,
        token_end_ptr,
        y2_ptr,
        total,
        hidden_dim: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        token_end = tl.load(token_end_ptr)
        mask = (offs < total) & (offs < token_end * hidden_dim)
        col = offs % hidden_dim
        row = offs // hidden_dim
        gate = tl.load(y1_ptr + row * hidden_dim * 2 + col, mask=mask).to(tl.float32)
        up = tl.load(
            y1_ptr + row * hidden_dim * 2 + hidden_dim + col,
            mask=mask,
        ).to(tl.float32)
        y2 = gate * tl.sigmoid(gate) * up
        tl.store(y2_ptr + offs, y2, mask=mask)

    @triton.jit
    def _swiglu_backward_with_offsets_kernel(
        dy2_ptr,
        y1_ptr,
        token_end_ptr,
        dy1_ptr,
        total,
        hidden_dim: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        token_end = tl.load(token_end_ptr)
        mask = (offs < total) & (offs < token_end * hidden_dim)
        col = offs % hidden_dim
        row = offs // hidden_dim
        y1_base = row * hidden_dim * 2
        gate = tl.load(y1_ptr + y1_base + col, mask=mask).to(tl.float32)
        up = tl.load(y1_ptr + y1_base + hidden_dim + col, mask=mask).to(tl.float32)
        dy2 = tl.load(dy2_ptr + offs, mask=mask).to(tl.float32)
        sig = tl.sigmoid(gate)
        silu_gate = gate * sig
        dgate = dy2 * up * sig * (1.0 + gate * (1.0 - sig))
        dup = dy2 * silu_gate
        tl.store(dy1_ptr + y1_base + col, dgate, mask=mask)
        tl.store(dy1_ptr + y1_base + hidden_dim + col, dup, mask=mask)

    @triton.jit
    def _clone_valid_prefix_kernel(
        input_ptr,
        token_end_ptr,
        out_ptr,
        total,
        row_width: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        token_end = tl.load(token_end_ptr)
        mask = (offs < total) & (offs < token_end * row_width)
        values = tl.load(input_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, values, mask=mask)

    @triton.jit
    def _weighted_sum_forward_kernel(
        y_partial_ptr,
        scores_ptr,
        out_ptr,
        batch,
        dim: tl.constexpr,
        TOPK: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        dim_block = tl.program_id(1)
        offs_d = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = (token < batch) & (offs_d < dim)
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for expert_slot in tl.static_range(0, TOPK):
            y = tl.load(
                y_partial_ptr + (token * TOPK + expert_slot) * dim + offs_d,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            score = tl.load(scores_ptr + token * TOPK + expert_slot).to(tl.float32)
            acc += y * score
        tl.store(out_ptr + token * dim + offs_d, acc, mask=mask)

    @triton.jit
    def _weighted_sum_backward_kernel(
        grad_out_ptr,
        y_partial_ptr,
        scores_ptr,
        grad_y_partial_ptr,
        grad_scores_ptr,
        batch,
        dim: tl.constexpr,
        TOPK: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        expert_slot = tl.program_id(1)
        dim_block = tl.program_id(2)
        offs_d = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = (token < batch) & (offs_d < dim)
        grad_out = tl.load(
            grad_out_ptr + token * dim + offs_d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        y = tl.load(
            y_partial_ptr + (token * TOPK + expert_slot) * dim + offs_d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        score = tl.load(scores_ptr + token * TOPK + expert_slot).to(tl.float32)
        tl.store(
            grad_y_partial_ptr + (token * TOPK + expert_slot) * dim + offs_d,
            grad_out * score,
            mask=mask,
        )
        partial_score_grad = tl.sum(grad_out * y, axis=0)
        tl.atomic_add(
            grad_scores_ptr + token * TOPK + expert_slot,
            partial_score_grad,
            sem="relaxed",
        )

    @requires_shmem
    @triton.jit
    def _router_dispatch_kernel(
        my_tokens_ptr,
        dest_ranks_ptr,
        dest_offsets_ptr,
        buffers_cuda_ptrs_ptr,
        dispatch_recv_origin_global_token_id_ptr,
        dispatch_recv_weights_ptr,
        offs_recv_tokens,
        offs_recv_origin_global_token_id,
        ep_rank,
        max_B,
        total_copies,
        D_BYTES: tl.constexpr,
        TOPK: tl.constexpr,
        WRITE_MAPPING: tl.constexpr,
    ):
        copy_id = tl.program_id(0)
        num_programs = tl.num_programs(0)
        base_ptr = tl.load(buffers_cuda_ptrs_ptr + ep_rank)

        while copy_id < total_copies:
            token_id = copy_id // TOPK
            expert_slot = copy_id - token_id * TOPK
            dest_rank = tl.load(dest_ranks_ptr + copy_id)
            dest_offset = tl.load(dest_offsets_ptr + copy_id)
            valid = dest_rank >= 0

            src_ptr = my_tokens_ptr + copy_id * D_BYTES
            dst_ptr = (base_ptr + offs_recv_tokens + dest_offset * D_BYTES).to(
                tl.pointer_type(tl.uint8)
            )

            if valid:
                shmem_backend.put(dst_ptr, src_ptr, D_BYTES, dest_rank)

            if valid and WRITE_MAPPING:
                id_ptr = (base_ptr + offs_recv_origin_global_token_id).to(
                    tl.pointer_type(tl.int64)
                )
                global_token_id = ep_rank * max_B * TOPK + token_id * TOPK + expert_slot
                scratch_id_ptr = dispatch_recv_weights_ptr.to(tl.pointer_type(tl.int64))
                scratch_id_ptr += copy_id
                tl.store(scratch_id_ptr, global_token_id)
                shmem_backend.put(id_ptr + dest_offset, scratch_id_ptr, 1, dest_rank)

            copy_id += num_programs
        shmem_backend.quiet()

    @requires_shmem
    @triton.jit
    def _router_combine_kernel(
        send_tokens_ptr,
        token_send_end_ptr,
        send_origin_global_token_id_ptr,
        buffers_cuda_ptrs_ptr,
        offs_combine_recv_tokens,
        ep_rank,
        ep_size,
        max_B,
        total_copies,
        D_BYTES: tl.constexpr,
        TOPK: tl.constexpr,
    ):
        copy_id = tl.program_id(0)
        num_programs = tl.num_programs(0)
        token_send_end = tl.load(token_send_end_ptr)
        base_ptr = tl.load(buffers_cuda_ptrs_ptr + ep_rank)
        max_B_topk = max_B * TOPK

        while copy_id < total_copies:
            valid = copy_id < token_send_end
            origin_id = tl.load(
                send_origin_global_token_id_ptr + copy_id,
                mask=valid,
                other=-1,
            )
            valid = valid & (origin_id >= 0)

            from_ep_rank = origin_id // max_B_topk
            dest_idx = origin_id - from_ep_rank * max_B_topk
            valid = valid & (from_ep_rank >= 0) & (from_ep_rank < ep_size)

            src_ptr = send_tokens_ptr + copy_id * D_BYTES
            dst_ptr = (base_ptr + offs_combine_recv_tokens + dest_idx * D_BYTES).to(
                tl.pointer_type(tl.uint8)
            )
            if valid:
                shmem_backend.put(dst_ptr, src_ptr, D_BYTES, from_ep_rank.to(tl.int32))

            copy_id += num_programs
        shmem_backend.quiet()

    _router_dispatch_tlx_kernel_untyped: Any = None
    _router_combine_tlx_kernel_untyped: Any = None

    if tlx is not None:
        tlx_mod = cast(Any, tlx)

        @triton.jit
        def _threadfence_system():
            tl.inline_asm_elementwise(
                "fence.sc.sys; // $0",
                "=r",
                [],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )

        @triton.jit
        def _router_dispatch_tlx_kernel(
            my_tokens_ptr,
            dest_ranks_ptr,
            dest_offsets_ptr,
            buffers_cuda_ptrs_ptr,
            dispatch_recv_origin_global_token_id_ptr,
            dispatch_recv_weights_ptr,
            offs_recv_tokens,
            offs_recv_origin_global_token_id,
            ep_rank,
            max_B,
            total_copies,
            D_BYTES: tl.constexpr,
            SMEM_SIZE: tl.constexpr,
            NUM_STAGES: tl.constexpr,
            TOPK: tl.constexpr,
            EP_SIZE: tl.constexpr,
            WRITE_MAPPING: tl.constexpr,
        ):
            pid = tl.program_id(0)
            smem = tlx_mod.local_alloc((SMEM_SIZE,), tl.uint8, num=NUM_STAGES)
            bars_full = tlx_mod.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
            bars_empty = tlx_mod.alloc_barriers(
                num_barriers=NUM_STAGES,
                arrive_count=1,
            )

            num_programs = tl.num_programs(0)
            copy_begin = pid * total_copies // num_programs
            copy_end = (pid + 1) * total_copies // num_programs
            num_copies = copy_end - copy_begin

            with tlx_mod.async_tasks():
                with tlx_mod.async_task("default"):
                    p = 1
                    for i in range(num_copies):
                        copy_id = copy_begin + i
                        buf = i % NUM_STAGES  # pyrefly: ignore[unsupported-operation]

                        empty_bar = tlx_mod.local_view(bars_empty, buf)
                        tlx_mod.barrier_wait(empty_bar, p)

                        src_base = my_tokens_ptr + copy_id * D_BYTES
                        full_bar = tlx_mod.local_view(bars_full, buf)
                        smem_buf = tlx_mod.local_view(smem, buf)
                        tlx_mod.barrier_expect_bytes(full_bar, D_BYTES)
                        tlx_mod.async_load(
                            src_base,
                            smem_buf,
                            bulk=True,
                            bulk_size=D_BYTES,
                            barrier=full_bar,
                        )
                        p = p ^ (buf == (NUM_STAGES - 1))

                with tlx_mod.async_task(num_warps=1, replicate=NUM_STAGES):
                    replica_id = tlx_mod.async_task_replica_id()
                    p = 0
                    num_consumer_iters = (
                        num_copies - replica_id + NUM_STAGES - 1
                    ) // NUM_STAGES

                    for j in range(num_consumer_iters):
                        copy_id = copy_begin + j * NUM_STAGES + replica_id
                        token_id = copy_id // TOPK
                        expert_slot = copy_id - token_id * TOPK
                        flat_idx = token_id * TOPK + expert_slot
                        dest_rank = tl.load(dest_ranks_ptr + flat_idx)
                        dest_offset = tl.load(dest_offsets_ptr + flat_idx)
                        valid = (dest_rank >= 0) & (dest_rank < EP_SIZE)

                        full_bar = tlx_mod.local_view(bars_full, replica_id)
                        tlx_mod.barrier_wait(full_bar, p)

                        if valid:
                            dest_buf_addr = tl.load(
                                buffers_cuda_ptrs_ptr + dest_rank.to(tl.int64)
                            )
                            dst_base = (
                                dest_buf_addr + offs_recv_tokens + dest_offset * D_BYTES
                            ).to(tl.pointer_type(tl.uint8))
                            smem_buf = tlx_mod.local_view(smem, replica_id)
                            tlx_mod.async_store(dst_base, smem_buf, D_BYTES)
                            tlx_mod.async_descriptor_store_wait(0)

                            if WRITE_MAPPING:
                                id_ptr = (
                                    dest_buf_addr + offs_recv_origin_global_token_id
                                ).to(tl.pointer_type(tl.int64))
                                ep_rank_i64 = ep_rank + tl.zeros([], dtype=tl.int64)
                                max_B_i64 = max_B + tl.zeros([], dtype=tl.int64)
                                token_id_i64 = token_id + tl.zeros([], dtype=tl.int64)
                                expert_slot_i64 = expert_slot + tl.zeros(
                                    [],
                                    dtype=tl.int64,
                                )
                                topk_i64 = tl.constexpr(TOPK) + tl.zeros(
                                    [],
                                    dtype=tl.int64,
                                )
                                global_token_id = (
                                    ep_rank_i64 * max_B_i64 * topk_i64
                                    + token_id_i64 * topk_i64
                                    + expert_slot_i64
                                )
                                tl.store(id_ptr + dest_offset, global_token_id)

                        empty_bar = tlx_mod.local_view(bars_empty, replica_id)
                        tlx_mod.barrier_arrive(empty_bar)
                        p = p ^ 1

                    _threadfence_system()

        @triton.jit
        def _router_combine_tlx_kernel(
            send_tokens_ptr,
            token_send_end_ptr,
            send_origin_global_token_id_ptr,
            buffers_cuda_ptrs_ptr,
            offs_combine_recv_tokens,
            max_B,
            D_BYTES: tl.constexpr,
            SMEM_SIZE: tl.constexpr,
            NUM_STAGES: tl.constexpr,
            TOPK: tl.constexpr,
            EP_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            smem = tlx_mod.local_alloc((SMEM_SIZE,), tl.uint8, num=NUM_STAGES)
            bars_full = tlx_mod.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
            bars_empty = tlx_mod.alloc_barriers(
                num_barriers=NUM_STAGES,
                arrive_count=1,
            )

            token_send_end = tl.load(token_send_end_ptr)
            num_programs = tl.num_programs(0)
            copy_begin = pid * token_send_end // num_programs
            copy_end = (pid + 1) * token_send_end // num_programs
            num_copies = (copy_end - copy_begin).to(tl.int32)

            with tlx_mod.async_tasks():
                with tlx_mod.async_task("default"):
                    p = 1
                    for i in range(num_copies):
                        copy_id = (copy_begin + i).to(tl.int64)
                        buf = i % NUM_STAGES  # pyrefly: ignore[unsupported-operation]

                        empty_bar = tlx_mod.local_view(bars_empty, buf)
                        tlx_mod.barrier_wait(empty_bar, p)

                        src_base = send_tokens_ptr + copy_id * D_BYTES
                        full_bar = tlx_mod.local_view(bars_full, buf)
                        smem_buf = tlx_mod.local_view(smem, buf)
                        tlx_mod.barrier_expect_bytes(full_bar, D_BYTES)
                        tlx_mod.async_load(
                            src_base,
                            smem_buf,
                            bulk=True,
                            bulk_size=D_BYTES,
                            barrier=full_bar,
                        )
                        p = p ^ (buf == (NUM_STAGES - 1))

                with tlx_mod.async_task(num_warps=1, replicate=NUM_STAGES):
                    replica_id = tlx_mod.async_task_replica_id()
                    p = 0
                    num_consumer_iters = (
                        num_copies - replica_id + NUM_STAGES - 1
                    ) // NUM_STAGES
                    max_B_i64 = max_B + tl.zeros([], dtype=tl.int64)
                    topk_i64 = tl.constexpr(TOPK) + tl.zeros([], dtype=tl.int64)
                    max_B_topk = max_B_i64 * topk_i64

                    for j in range(num_consumer_iters):
                        copy_id = (copy_begin + j * NUM_STAGES + replica_id).to(
                            tl.int64
                        )
                        origin_id = tl.load(send_origin_global_token_id_ptr + copy_id)
                        from_ep_rank = origin_id // max_B_topk
                        dest_idx = origin_id - from_ep_rank * max_B_topk
                        valid = (
                            (origin_id >= 0)
                            & (from_ep_rank >= 0)
                            & (from_ep_rank < EP_SIZE)
                        )

                        full_bar = tlx_mod.local_view(bars_full, replica_id)
                        tlx_mod.barrier_wait(full_bar, p)

                        if valid:
                            dest_buf_addr = tl.load(
                                buffers_cuda_ptrs_ptr + from_ep_rank
                            )
                            dst_base = (
                                dest_buf_addr
                                + offs_combine_recv_tokens
                                + dest_idx * D_BYTES
                            ).to(tl.pointer_type(tl.uint8))
                            smem_buf = tlx_mod.local_view(smem, replica_id)
                            tlx_mod.async_store(dst_base, smem_buf, D_BYTES)
                            tlx_mod.async_descriptor_store_wait(0)

                        empty_bar = tlx_mod.local_view(bars_empty, replica_id)
                        tlx_mod.barrier_arrive(empty_bar)
                        p = p ^ 1

                    _threadfence_system()

        _router_dispatch_tlx_kernel_untyped = cast(
            Any,
            _router_dispatch_tlx_kernel,
        )
        _router_combine_tlx_kernel_untyped = cast(
            Any,
            _router_combine_tlx_kernel,
        )

    @triton.jit
    def _zfill_ranges_kernel(
        input_ptr,
        begin_ofs_ptr,
        end_ofs_ptr,
        row_num_bytes: tl.constexpr,
        BLOCK_BYTES: tl.constexpr,
    ):
        range_idx = tl.program_id(0)
        row_in_range = tl.program_id(1)
        begin = tl.load(begin_ofs_ptr + range_idx)
        end = tl.load(end_ofs_ptr + range_idx)
        row = begin + row_in_range
        byte_offsets = tl.arange(0, BLOCK_BYTES)
        mask = (row < end) & (byte_offsets < row_num_bytes)
        tl.store(
            input_ptr + row * row_num_bytes + byte_offsets,
            tl.zeros([BLOCK_BYTES], dtype=tl.uint8),
            mask=mask,
        )

    _barrier_arrive_kernel_untyped = cast(Any, _barrier_arrive_kernel)
    _barrier_wait_kernel_untyped = cast(Any, _barrier_wait_kernel)
    _ep_allgather_kernel_untyped = cast(Any, _ep_allgather_kernel)
    _compute_expert_offsets_kernel_untyped = cast(Any, _compute_expert_offsets_kernel)
    _compute_dest_offsets_kernel_untyped = cast(Any, _compute_dest_offsets_kernel)
    _swiglu_forward_kernel_untyped = cast(Any, _swiglu_forward_kernel)
    _swiglu_backward_kernel_untyped = cast(Any, _swiglu_backward_kernel)
    _swiglu_forward_with_offsets_kernel_untyped = cast(
        Any,
        _swiglu_forward_with_offsets_kernel,
    )
    _swiglu_backward_with_offsets_kernel_untyped = cast(
        Any,
        _swiglu_backward_with_offsets_kernel,
    )
    _clone_valid_prefix_kernel_untyped = cast(Any, _clone_valid_prefix_kernel)
    _weighted_sum_forward_kernel_untyped = cast(Any, _weighted_sum_forward_kernel)
    _weighted_sum_backward_kernel_untyped = cast(Any, _weighted_sum_backward_kernel)
    _router_dispatch_kernel_untyped = cast(Any, _router_dispatch_kernel)
    _router_combine_kernel_untyped = cast(Any, _router_combine_kernel)
    _zfill_ranges_kernel_untyped = cast(Any, _zfill_ranges_kernel)

    if not hasattr(torch.ops.inductor, "flex_ep_barrier_arrive"):

        @torch.library.custom_op(
            "inductor::flex_ep_barrier_arrive",
            mutates_args=("flag",),
        )
        def _flex_ep_barrier_arrive(flag: torch.Tensor) -> torch.Tensor:
            out = torch.empty(1, dtype=torch.int32, device=flag.device)
            _barrier_arrive_kernel_untyped[(1,)](flag, out, num_warps=1)
            return out

    if not hasattr(torch.ops.inductor, "flex_ep_barrier_wait"):

        @torch.library.custom_op("inductor::flex_ep_barrier_wait", mutates_args=())
        def _flex_ep_barrier_wait(
            cuda_ptrs: torch.Tensor,
            offs_flag: int,
            expected: torch.Tensor,
            timeout_s: float = 5.0,
        ) -> None:
            block = triton.next_power_of_2(cuda_ptrs.numel())
            timeout_ns = int(timeout_s * 1e9)
            _barrier_wait_kernel_untyped[(1,)](
                cuda_ptrs,
                expected,
                cuda_ptrs.numel(),
                offs_flag,
                timeout_ns,
                BLOCK=block,
                num_warps=1,
            )

    if not hasattr(torch.ops.inductor, "flex_ep_allgather"):

        @torch.library.custom_op(
            "inductor::flex_ep_allgather",
            mutates_args=("output",),
        )
        def _flex_ep_allgather(
            output: torch.Tensor,
            input: torch.Tensor,
            buffers_cuda_ptrs: torch.Tensor,
            offs_output: int,
            ep_rank: int,
        ) -> None:
            input_u8 = input.contiguous().view(torch.uint8)
            _ep_allgather_kernel_untyped[(buffers_cuda_ptrs.numel(),)](
                input_u8,
                buffers_cuda_ptrs,
                offs_output,
                ep_rank,
                input_num_bytes=input_u8.numel(),
                num_warps=1,
            )

    if not hasattr(torch.ops.inductor, "flex_ep_router_compute_all_expert_offsets"):

        @torch.library.custom_op(
            "inductor::flex_ep_router_compute_all_expert_offsets",
            mutates_args=(),
        )
        def _flex_ep_router_compute_all_expert_offsets(
            all_expert_counts: torch.Tensor,
            ep_rank: int,
            local_experts: int,
            token_alignment: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if all_expert_counts.dim() != 2:
                raise ValueError(
                    "flex_ep_router_compute_all_expert_offsets expects "
                    f"[ep_size, num_experts], got {all_expert_counts.shape}."
                )
            ep_size, num_experts = all_expert_counts.shape
            if local_experts <= 0 or num_experts != ep_size * local_experts:
                raise ValueError(
                    "flex_ep_router_compute_all_expert_offsets expects "
                    f"num_experts == ep_size * local_experts, got "
                    f"{num_experts=} {ep_size=} {local_experts=}."
                )
            if not 0 <= ep_rank < ep_size:
                raise ValueError(f"ep_rank must be in [0, {ep_size}), got {ep_rank}.")
            if all_expert_counts.dtype != torch.int64:
                raise ValueError(
                    "flex_ep_router_compute_all_expert_offsets requires int64 "
                    f"all_expert_counts, got {all_expert_counts.dtype}."
                )
            if not all_expert_counts.is_contiguous():
                raise ValueError(
                    "flex_ep_router_compute_all_expert_offsets requires "
                    "contiguous all_expert_counts."
                )
            if not _is_power_of_2(ep_size) or not _is_power_of_2(local_experts):
                raise ValueError(
                    "flex_ep_router_compute_all_expert_offsets requires ep_size "
                    f"and local_experts to be powers of 2, got {ep_size=} "
                    f"{local_experts=}."
                )

            offsets = torch.empty(
                (ep_size, local_experts, ep_size + 1),
                dtype=torch.int32,
                device=all_expert_counts.device,
            )
            expert_start = torch.empty(
                (ep_size, local_experts + 1),
                dtype=torch.int32,
                device=all_expert_counts.device,
            )
            grand_total = torch.empty(
                (ep_size,),
                dtype=torch.int64,
                device=all_expert_counts.device,
            )
            _compute_expert_offsets_kernel_untyped[(ep_size,)](
                all_expert_counts,
                offsets,
                expert_start,
                grand_total,
                LOCAL_EXPERTS=local_experts,
                EP_SIZE=ep_size,
                TOKEN_ALIGNMENT=token_alignment,
                num_warps=4,
            )
            return offsets, grand_total[ep_rank].clone(), expert_start[ep_rank].clone()

        @_flex_ep_router_compute_all_expert_offsets.register_fake
        def _flex_ep_router_compute_all_expert_offsets_fake(
            all_expert_counts: torch.Tensor,
            ep_rank: int,
            local_experts: int,
            token_alignment: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            del ep_rank, token_alignment
            ep_size = all_expert_counts.shape[0]
            return (
                torch.empty(
                    (ep_size, local_experts, ep_size + 1),
                    dtype=torch.int32,
                    device=all_expert_counts.device,
                ),
                torch.empty((), dtype=torch.int64, device=all_expert_counts.device),
                torch.empty(
                    (local_experts + 1,),
                    dtype=torch.int32,
                    device=all_expert_counts.device,
                ),
            )

    if not hasattr(torch.ops.inductor, "flex_ep_router_compute_dest_offsets"):

        @torch.library.custom_op(
            "inductor::flex_ep_router_compute_dest_offsets",
            mutates_args=(),
        )
        def _flex_ep_router_compute_dest_offsets(
            topk_idx: torch.Tensor,
            recv_ofs: torch.Tensor,
            ep_size: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if topk_idx.dim() != 2 or recv_ofs.dim() != 1:
                raise ValueError(
                    "flex_ep_router_compute_dest_offsets expects topk_idx [batch, top_k] "
                    f"and recv_ofs [num_experts], got {topk_idx.shape} and {recv_ofs.shape}."
                )
            if ep_size <= 0 or recv_ofs.numel() % ep_size != 0:
                raise ValueError(
                    "flex_ep_router_compute_dest_offsets expects num_experts "
                    f"to be divisible by ep_size, got num_experts={recv_ofs.numel()} "
                    f"and {ep_size=}."
                )
            if topk_idx.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    "flex_ep_router_compute_dest_offsets expects int32 or int64 topk_idx."
                )
            if recv_ofs.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    "flex_ep_router_compute_dest_offsets expects int32 or int64 recv_ofs."
                )

            batch, top_k = topk_idx.shape
            num_experts = recv_ofs.numel()
            local_experts = num_experts // ep_size
            if topk_idx.numel() == 0:
                return (
                    torch.empty_like(topk_idx, dtype=torch.int32),
                    torch.empty_like(topk_idx, dtype=torch.int64),
                )
            if not topk_idx.is_contiguous():
                raise ValueError(
                    "flex_ep_router_compute_dest_offsets requires contiguous "
                    "topk_idx."
                )
            if not _is_power_of_2(num_experts):
                raise ValueError(
                    "flex_ep_router_compute_dest_offsets requires num_experts "
                    f"to be a power of 2, got {num_experts}."
                )

            dest_ranks = torch.div(
                topk_idx,
                local_experts,
                rounding_mode="floor",
            ).to(torch.int32)
            dest_offsets = torch.empty(
                topk_idx.shape,
                dtype=torch.int64,
                device=topk_idx.device,
            )
            block = 512
            _compute_dest_offsets_kernel_untyped[(num_experts,)](
                topk_idx,
                recv_ofs,
                recv_ofs.stride(0),
                dest_offsets,
                batch,
                TOPK=top_k,
                BLOCK=block,
                num_warps=4,
            )
            return dest_ranks, dest_offsets

        @_flex_ep_router_compute_dest_offsets.register_fake
        def _flex_ep_router_compute_dest_offsets_fake(
            topk_idx: torch.Tensor,
            recv_ofs: torch.Tensor,
            ep_size: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del recv_ofs, ep_size
            return (
                torch.empty_like(topk_idx, dtype=torch.int32),
                torch.empty_like(topk_idx, dtype=torch.int64),
            )

    if not hasattr(torch.ops.inductor, "flex_ep_swiglu_forward"):

        @torch.library.custom_op(
            "inductor::flex_ep_swiglu_forward",
            mutates_args=(),
        )
        def _flex_ep_swiglu_forward(y1: torch.Tensor) -> torch.Tensor:
            if y1.dtype != torch.bfloat16:
                raise ValueError("flex_ep_swiglu_forward supports BF16 input only.")
            if y1.dim() != 2 or y1.shape[-1] % 2 != 0:
                raise ValueError(
                    f"flex_ep_swiglu_forward expects [tokens, 2 * hidden], got {y1.shape}."
                )
            if not y1.is_contiguous():
                raise ValueError("flex_ep_swiglu_forward requires contiguous input.")
            hidden_dim = y1.shape[-1] // 2
            y2 = torch.empty(
                (y1.shape[0], hidden_dim),
                device=y1.device,
                dtype=y1.dtype,
            )
            total = y2.numel()
            if total == 0:
                return y2
            block = 1024
            _swiglu_forward_kernel_untyped[(triton.cdiv(total, block),)](
                y1,
                y2,
                total,
                hidden_dim,
                BLOCK=block,
                num_warps=4,
            )
            return y2

        @_flex_ep_swiglu_forward.register_fake
        def _flex_ep_swiglu_forward_fake(y1: torch.Tensor) -> torch.Tensor:
            return torch.empty(
                (y1.shape[0], y1.shape[-1] // 2),
                device=y1.device,
                dtype=y1.dtype,
            )

    if not hasattr(torch.ops.inductor, "flex_ep_swiglu_backward"):

        @torch.library.custom_op(
            "inductor::flex_ep_swiglu_backward",
            mutates_args=(),
        )
        def _flex_ep_swiglu_backward(
            dy2: torch.Tensor,
            y1: torch.Tensor,
        ) -> torch.Tensor:
            if dy2.dtype != torch.bfloat16 or y1.dtype != torch.bfloat16:
                raise ValueError("flex_ep_swiglu_backward supports BF16 inputs only.")
            if y1.dim() != 2 or y1.shape[-1] % 2 != 0:
                raise ValueError(
                    f"flex_ep_swiglu_backward expects y1 [tokens, 2 * hidden], got {y1.shape}."
                )
            if dy2.shape != (y1.shape[0], y1.shape[-1] // 2):
                raise ValueError(
                    "flex_ep_swiglu_backward dy2 shape must match "
                    f"[tokens, hidden], got dy2={dy2.shape}, y1={y1.shape}."
                )
            if not dy2.is_contiguous() or not y1.is_contiguous():
                raise ValueError("flex_ep_swiglu_backward requires contiguous inputs.")
            hidden_dim = dy2.shape[-1]
            dy1 = torch.empty_like(y1)
            total = dy2.numel()
            if total == 0:
                return dy1
            block = 1024
            _swiglu_backward_kernel_untyped[(triton.cdiv(total, block),)](
                dy2,
                y1,
                dy1,
                total,
                hidden_dim,
                BLOCK=block,
                num_warps=4,
            )
            return dy1

        @_flex_ep_swiglu_backward.register_fake
        def _flex_ep_swiglu_backward_fake(
            dy2: torch.Tensor,
            y1: torch.Tensor,
        ) -> torch.Tensor:
            del dy2
            return torch.empty_like(y1)

    if not hasattr(torch.ops.inductor, "flex_ep_swiglu_forward_with_offsets"):

        @torch.library.custom_op(
            "inductor::flex_ep_swiglu_forward_with_offsets",
            mutates_args=(),
        )
        def _flex_ep_swiglu_forward_with_offsets(
            y1: torch.Tensor,
            token_end: torch.Tensor,
        ) -> torch.Tensor:
            if y1.dtype != torch.bfloat16:
                raise ValueError(
                    "flex_ep_swiglu_forward_with_offsets supports BF16 input only."
                )
            if token_end.dtype != torch.int64 or token_end.numel() != 1:
                raise ValueError(
                    "flex_ep_swiglu_forward_with_offsets expects token_end "
                    "to be a single int64 tensor."
                )
            if y1.dim() != 2 or y1.shape[-1] % 2 != 0:
                raise ValueError(
                    "flex_ep_swiglu_forward_with_offsets expects "
                    f"[tokens, 2 * hidden], got {y1.shape}."
                )
            if not y1.is_contiguous() or not token_end.is_contiguous():
                raise ValueError(
                    "flex_ep_swiglu_forward_with_offsets requires contiguous inputs."
                )
            hidden_dim = y1.shape[-1] // 2
            y2 = torch.empty(
                (y1.shape[0], hidden_dim),
                device=y1.device,
                dtype=y1.dtype,
            )
            total = y2.numel()
            if total == 0:
                return y2
            block = 1024
            _swiglu_forward_with_offsets_kernel_untyped[(triton.cdiv(total, block),)](
                y1,
                token_end,
                y2,
                total,
                hidden_dim,
                BLOCK=block,
                num_warps=4,
            )
            return y2

        @_flex_ep_swiglu_forward_with_offsets.register_fake
        def _flex_ep_swiglu_forward_with_offsets_fake(
            y1: torch.Tensor,
            token_end: torch.Tensor,
        ) -> torch.Tensor:
            del token_end
            return torch.empty(
                (y1.shape[0], y1.shape[-1] // 2),
                device=y1.device,
                dtype=y1.dtype,
            )

    if not hasattr(torch.ops.inductor, "flex_ep_swiglu_backward_with_offsets"):

        @torch.library.custom_op(
            "inductor::flex_ep_swiglu_backward_with_offsets",
            mutates_args=(),
        )
        def _flex_ep_swiglu_backward_with_offsets(
            dy2: torch.Tensor,
            y1: torch.Tensor,
            token_end: torch.Tensor,
        ) -> torch.Tensor:
            if dy2.dtype != torch.bfloat16 or y1.dtype != torch.bfloat16:
                raise ValueError(
                    "flex_ep_swiglu_backward_with_offsets supports BF16 inputs only."
                )
            if token_end.dtype != torch.int64 or token_end.numel() != 1:
                raise ValueError(
                    "flex_ep_swiglu_backward_with_offsets expects token_end "
                    "to be a single int64 tensor."
                )
            if y1.dim() != 2 or y1.shape[-1] % 2 != 0:
                raise ValueError(
                    "flex_ep_swiglu_backward_with_offsets expects y1 "
                    f"[tokens, 2 * hidden], got {y1.shape}."
                )
            if dy2.shape != (y1.shape[0], y1.shape[-1] // 2):
                raise ValueError(
                    "flex_ep_swiglu_backward_with_offsets dy2 shape must match "
                    f"[tokens, hidden], got dy2={dy2.shape}, y1={y1.shape}."
                )
            if (
                not dy2.is_contiguous()
                or not y1.is_contiguous()
                or not token_end.is_contiguous()
            ):
                raise ValueError(
                    "flex_ep_swiglu_backward_with_offsets requires contiguous inputs."
                )
            hidden_dim = dy2.shape[-1]
            dy1 = torch.empty_like(y1)
            total = dy2.numel()
            if total == 0:
                return dy1
            block = 1024
            _swiglu_backward_with_offsets_kernel_untyped[(triton.cdiv(total, block),)](
                dy2,
                y1,
                token_end,
                dy1,
                total,
                hidden_dim,
                BLOCK=block,
                num_warps=4,
            )
            return dy1

        @_flex_ep_swiglu_backward_with_offsets.register_fake
        def _flex_ep_swiglu_backward_with_offsets_fake(
            dy2: torch.Tensor,
            y1: torch.Tensor,
            token_end: torch.Tensor,
        ) -> torch.Tensor:
            del dy2, token_end
            return torch.empty_like(y1)

    if not hasattr(torch.ops.inductor, "flex_ep_clone_valid_prefix"):

        @torch.library.custom_op(
            "inductor::flex_ep_clone_valid_prefix",
            mutates_args=(),
        )
        def _flex_ep_clone_valid_prefix(
            input: torch.Tensor,
            token_end: torch.Tensor,
        ) -> torch.Tensor:
            if input.dim() < 1:
                raise ValueError(
                    "flex_ep_clone_valid_prefix expects at least 1D input."
                )
            if token_end.dtype != torch.int64 or token_end.numel() != 1:
                raise ValueError(
                    "flex_ep_clone_valid_prefix expects token_end to be a "
                    "single int64 tensor."
                )
            if not input.is_contiguous() or not token_end.is_contiguous():
                raise ValueError(
                    "flex_ep_clone_valid_prefix requires contiguous inputs."
                )
            out = torch.empty_like(input)
            total = input.numel()
            if total == 0:
                return out
            row_width = total // input.shape[0]
            block = 1024
            _clone_valid_prefix_kernel_untyped[(triton.cdiv(total, block),)](
                input,
                token_end,
                out,
                total,
                row_width,
                BLOCK=block,
                num_warps=4,
            )
            return out

        @_flex_ep_clone_valid_prefix.register_fake
        def _flex_ep_clone_valid_prefix_fake(
            input: torch.Tensor,
            token_end: torch.Tensor,
        ) -> torch.Tensor:
            del token_end
            return torch.empty_like(input)

    if not hasattr(torch.ops.inductor, "flex_ep_weighted_sum_forward"):

        @torch.library.custom_op(
            "inductor::flex_ep_weighted_sum_forward",
            mutates_args=(),
        )
        def _flex_ep_weighted_sum_forward(
            y_partial: torch.Tensor,
            top_scores: torch.Tensor,
        ) -> torch.Tensor:
            if y_partial.dtype != torch.bfloat16 or top_scores.dtype != torch.float32:
                raise ValueError(
                    "flex_ep_weighted_sum_forward expects BF16 y_partial and "
                    "FP32 top_scores."
                )
            if y_partial.dim() != 3 or top_scores.dim() != 2:
                raise ValueError(
                    "flex_ep_weighted_sum_forward expects y_partial [tokens, topk, dim] "
                    f"and top_scores [tokens, topk], got {y_partial.shape} and {top_scores.shape}."
                )
            if y_partial.shape[:2] != top_scores.shape:
                raise ValueError(
                    "flex_ep_weighted_sum_forward top_scores shape must match "
                    f"y_partial leading dims, got {top_scores.shape} and {y_partial.shape[:2]}."
                )
            if not y_partial.is_contiguous() or not top_scores.is_contiguous():
                raise ValueError(
                    "flex_ep_weighted_sum_forward requires contiguous inputs."
                )
            out = torch.empty(
                (y_partial.shape[0], y_partial.shape[2]),
                device=y_partial.device,
                dtype=y_partial.dtype,
            )
            if out.numel() == 0:
                return out
            block_d = 256
            _weighted_sum_forward_kernel_untyped[
                (y_partial.shape[0], triton.cdiv(y_partial.shape[2], block_d))
            ](
                y_partial,
                top_scores,
                out,
                y_partial.shape[0],
                dim=y_partial.shape[2],
                TOPK=y_partial.shape[1],
                BLOCK_D=block_d,
                num_warps=4,
            )
            return out

        @_flex_ep_weighted_sum_forward.register_fake
        def _flex_ep_weighted_sum_forward_fake(
            y_partial: torch.Tensor,
            top_scores: torch.Tensor,
        ) -> torch.Tensor:
            del top_scores
            return torch.empty(
                (y_partial.shape[0], y_partial.shape[2]),
                device=y_partial.device,
                dtype=y_partial.dtype,
            )

    if not hasattr(torch.ops.inductor, "flex_ep_weighted_sum_backward"):

        @torch.library.custom_op(
            "inductor::flex_ep_weighted_sum_backward",
            mutates_args=(),
        )
        def _flex_ep_weighted_sum_backward(
            grad_out: torch.Tensor,
            y_partial: torch.Tensor,
            top_scores: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if (
                grad_out.dtype != torch.bfloat16
                or y_partial.dtype != torch.bfloat16
                or top_scores.dtype != torch.float32
            ):
                raise ValueError(
                    "flex_ep_weighted_sum_backward expects BF16 grad/y_partial "
                    "and FP32 top_scores."
                )
            if y_partial.dim() != 3 or top_scores.dim() != 2 or grad_out.dim() != 2:
                raise ValueError(
                    "flex_ep_weighted_sum_backward expects grad_out [tokens, dim], "
                    "y_partial [tokens, topk, dim], and top_scores [tokens, topk]."
                )
            if grad_out.shape != (y_partial.shape[0], y_partial.shape[2]):
                raise ValueError(
                    "flex_ep_weighted_sum_backward grad_out shape must match "
                    f"[tokens, dim], got {grad_out.shape} and {y_partial.shape}."
                )
            if y_partial.shape[:2] != top_scores.shape:
                raise ValueError(
                    "flex_ep_weighted_sum_backward top_scores shape must match "
                    f"y_partial leading dims, got {top_scores.shape} and {y_partial.shape[:2]}."
                )
            if (
                not grad_out.is_contiguous()
                or not y_partial.is_contiguous()
                or not top_scores.is_contiguous()
            ):
                raise ValueError(
                    "flex_ep_weighted_sum_backward requires contiguous inputs."
                )
            grad_y_partial = torch.empty_like(y_partial)
            grad_scores = torch.zeros_like(top_scores)
            if grad_y_partial.numel() == 0:
                return grad_y_partial, grad_scores
            block_d = 1024
            _weighted_sum_backward_kernel_untyped[
                (
                    y_partial.shape[0],
                    y_partial.shape[1],
                    triton.cdiv(y_partial.shape[2], block_d),
                )
            ](
                grad_out,
                y_partial,
                top_scores,
                grad_y_partial,
                grad_scores,
                y_partial.shape[0],
                dim=y_partial.shape[2],
                TOPK=y_partial.shape[1],
                BLOCK_D=block_d,
                num_warps=8,
            )
            return grad_y_partial, grad_scores

        @_flex_ep_weighted_sum_backward.register_fake
        def _flex_ep_weighted_sum_backward_fake(
            grad_out: torch.Tensor,
            y_partial: torch.Tensor,
            top_scores: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del grad_out
            return torch.empty_like(y_partial), torch.empty_like(top_scores)

    if not hasattr(torch.ops.inductor, "flex_ep_router_dispatch"):

        @torch.library.custom_op(
            "inductor::flex_ep_router_dispatch",
            mutates_args=(
                "dispatch_recv_buffer",
                "dispatch_recv_buffer_scaling_factors",
                "dispatch_recv_origin_global_token_id",
                "dispatch_recv_weights",
            ),
            schema="(Tensor my_tokens, Tensor? my_scaling_factors, "
            "Tensor? my_topk_weights, Tensor dest_ranks, Tensor dest_offsets, "
            "Tensor buffers_cuda_ptrs, Tensor(a!) dispatch_recv_buffer, "
            "Tensor(b!) dispatch_recv_buffer_scaling_factors, "
            "Tensor(c!) dispatch_recv_origin_global_token_id, "
            "Tensor(d!) dispatch_recv_weights, int offs_recv_tokens, "
            "int offs_recv_scaling_factors, int offs_recv_weights, "
            "int offs_recv_origin_global_token_id, int ep_rank, int num_ctas, "
            "int max_B) -> ()",
        )
        def _flex_ep_router_dispatch(
            my_tokens: torch.Tensor,
            my_scaling_factors: torch.Tensor | None,
            my_topk_weights: torch.Tensor | None,
            dest_ranks: torch.Tensor,
            dest_offsets: torch.Tensor,
            buffers_cuda_ptrs: torch.Tensor,
            dispatch_recv_buffer: torch.Tensor,
            dispatch_recv_buffer_scaling_factors: torch.Tensor,
            dispatch_recv_origin_global_token_id: torch.Tensor,
            dispatch_recv_weights: torch.Tensor,
            offs_recv_tokens: int,
            offs_recv_scaling_factors: int,
            offs_recv_weights: int,
            offs_recv_origin_global_token_id: int,
            ep_rank: int,
            num_ctas: int,
            max_B: int,
        ) -> None:
            del my_scaling_factors, my_topk_weights
            del dispatch_recv_buffer, dispatch_recv_buffer_scaling_factors
            del offs_recv_scaling_factors, offs_recv_weights
            if my_tokens.dtype != torch.bfloat16:
                raise ValueError("flex_ep_router_dispatch supports BF16 tokens only.")
            if not my_tokens.is_contiguous():
                raise ValueError("flex_ep_router_dispatch requires contiguous tokens.")
            if num_ctas < 1:
                raise ValueError(f"num_ctas must be positive, got {num_ctas}.")
            batch, top_k, dim = my_tokens.shape
            my_tokens_u8 = my_tokens.view(torch.uint8)
            dim_bytes = dim * my_tokens.element_size()
            total_copies = batch * top_k
            if total_copies == 0:
                return
            if _router_dispatch_tlx_kernel_untyped is not None:
                if dim_bytes % 16 != 0:
                    raise ValueError(
                        "Token row size must be a multiple of 16 bytes, "
                        f"got {dim_bytes}."
                    )
                launch_ctas = min(num_ctas, MAX_TLX_NUM_CTAS, total_copies)
                _router_dispatch_tlx_kernel_untyped[(launch_ctas,)](
                    my_tokens_u8,
                    dest_ranks,
                    dest_offsets,
                    buffers_cuda_ptrs,
                    dispatch_recv_origin_global_token_id,
                    dispatch_recv_weights,
                    offs_recv_tokens,
                    offs_recv_origin_global_token_id,
                    ep_rank,
                    max_B,
                    total_copies,
                    D_BYTES=dim_bytes,
                    SMEM_SIZE=triton.next_power_of_2(dim_bytes),
                    NUM_STAGES=TLX_NUM_STAGES,
                    TOPK=top_k,
                    EP_SIZE=buffers_cuda_ptrs.numel(),
                    WRITE_MAPPING=offs_recv_origin_global_token_id >= 0,
                    num_warps=4,
                    num_stages=1,
                    ctas_per_cga=(4, 1, 1),
                )
                return
            _router_dispatch_kernel_untyped[(min(num_ctas, total_copies),)](
                my_tokens_u8,
                dest_ranks,
                dest_offsets,
                buffers_cuda_ptrs,
                dispatch_recv_origin_global_token_id,
                dispatch_recv_weights,
                offs_recv_tokens,
                offs_recv_origin_global_token_id,
                ep_rank,
                max_B,
                total_copies,
                D_BYTES=dim_bytes,
                TOPK=top_k,
                WRITE_MAPPING=offs_recv_origin_global_token_id >= 0,
                num_warps=1,
            )

    if not hasattr(torch.ops.inductor, "flex_ep_router_combine"):

        @torch.library.custom_op(
            "inductor::flex_ep_router_combine",
            mutates_args=(
                "combine_recv_buffer",
                "combine_recv_scale_factors",
                "combine_recv_weights",
            ),
            schema="(Tensor send_tokens, Tensor? send_scale_factors, "
            "Tensor? send_weights, Tensor expert_begin_offset_per_ep, "
            "Tensor token_send_end, Tensor send_origin_global_token_id, "
            "Tensor buffers_cuda_ptrs, Tensor(a!) combine_recv_buffer, "
            "Tensor(b!) combine_recv_scale_factors, Tensor(c!) combine_recv_weights, "
            "int offs_combine_recv_tokens, int offs_combine_recv_scale_factors, "
            "int offs_combine_recv_weights, int ep_rank, int B, int TOPK, "
            "int num_ctas, int max_B) -> ()",
        )
        def _flex_ep_router_combine(
            send_tokens: torch.Tensor,
            send_scale_factors: torch.Tensor | None,
            send_weights: torch.Tensor | None,
            expert_begin_offset_per_ep: torch.Tensor,
            token_send_end: torch.Tensor,
            send_origin_global_token_id: torch.Tensor,
            buffers_cuda_ptrs: torch.Tensor,
            combine_recv_buffer: torch.Tensor,
            combine_recv_scale_factors: torch.Tensor,
            combine_recv_weights: torch.Tensor,
            offs_combine_recv_tokens: int,
            offs_combine_recv_scale_factors: int,
            offs_combine_recv_weights: int,
            ep_rank: int,
            batch: int,
            top_k: int,
            num_ctas: int,
            max_B: int,
        ) -> None:
            del send_scale_factors, send_weights, expert_begin_offset_per_ep
            del combine_recv_buffer, combine_recv_scale_factors, combine_recv_weights
            del offs_combine_recv_scale_factors, offs_combine_recv_weights
            del batch
            if send_tokens.dtype != torch.bfloat16:
                raise ValueError("flex_ep_router_combine supports BF16 tokens only.")
            if not send_tokens.is_contiguous():
                raise ValueError("flex_ep_router_combine requires contiguous tokens.")
            if num_ctas < 1:
                raise ValueError(f"num_ctas must be positive, got {num_ctas}.")
            dim = send_tokens.shape[-1]
            send_tokens_u8 = send_tokens.view(torch.uint8)
            dim_bytes = dim * send_tokens.element_size()
            total_copies = send_tokens.shape[0]
            if total_copies == 0:
                return
            if _router_combine_tlx_kernel_untyped is not None:
                if dim_bytes % 16 != 0:
                    raise ValueError(
                        "Token row size must be a multiple of 16 bytes, "
                        f"got {dim_bytes}."
                    )
                launch_ctas = min(num_ctas, MAX_TLX_NUM_CTAS, total_copies)
                _router_combine_tlx_kernel_untyped[(launch_ctas,)](
                    send_tokens_u8,
                    token_send_end,
                    send_origin_global_token_id,
                    buffers_cuda_ptrs,
                    offs_combine_recv_tokens,
                    max_B,
                    D_BYTES=dim_bytes,
                    SMEM_SIZE=triton.next_power_of_2(dim_bytes),
                    NUM_STAGES=TLX_NUM_STAGES,
                    TOPK=top_k,
                    EP_SIZE=buffers_cuda_ptrs.numel(),
                    num_warps=4,
                    num_stages=1,
                    ctas_per_cga=(4, 1, 1),
                )
                return
            _router_combine_kernel_untyped[(min(num_ctas, total_copies),)](
                send_tokens_u8,
                token_send_end,
                send_origin_global_token_id,
                buffers_cuda_ptrs,
                offs_combine_recv_tokens,
                ep_rank,
                buffers_cuda_ptrs.numel(),
                max_B,
                total_copies,
                D_BYTES=dim_bytes,
                TOPK=top_k,
                num_warps=1,
            )

    if not hasattr(torch.ops.inductor, "flex_ep_zfill_ranges_inplace"):

        @torch.library.custom_op(
            "inductor::flex_ep_zfill_ranges_inplace",
            mutates_args=("input",),
        )
        def _flex_ep_zfill_ranges_inplace(
            input: torch.Tensor,
            begin_ofs: torch.Tensor,
            end_ofs: torch.Tensor,
            max_values_per_batch: int,
        ) -> None:
            if not input.is_contiguous() or input.ndim != 2:
                raise ValueError(
                    "flex_ep_zfill_ranges_inplace requires a contiguous 2D tensor."
                )
            block_bytes = triton.next_power_of_2(input.shape[1])
            _zfill_ranges_kernel_untyped[(begin_ofs.numel(), max_values_per_batch)](
                input,
                begin_ofs,
                end_ofs,
                row_num_bytes=input.shape[1],
                BLOCK_BYTES=block_bytes,
                num_warps=1,
            )


_INDUCTOR_ROUTER_OP_NAMES = {
    "ep_allgather": "flex_ep_allgather",
    "router_compute_all_expert_offsets": "flex_ep_router_compute_all_expert_offsets",
    "router_compute_dest_offsets": "flex_ep_router_compute_dest_offsets",
    "router_dispatch": "flex_ep_router_dispatch",
    "router_combine": "flex_ep_router_combine",
    "barrier_arrive": "flex_ep_barrier_arrive",
    "barrier_wait": "flex_ep_barrier_wait",
    "swiglu_forward_with_offsets": "flex_ep_swiglu_forward_with_offsets",
    "swiglu_backward_with_offsets": "flex_ep_swiglu_backward_with_offsets",
    "clone_valid_prefix": "flex_ep_clone_valid_prefix",
    "zfill_ranges_inplace": "flex_ep_zfill_ranges_inplace",
}


def _require_inductor_router_op(name: str) -> Any:
    op_name = _INDUCTOR_ROUTER_OP_NAMES[name]
    op = getattr(torch.ops.inductor, op_name, None)
    if op is not None:
        return op
    register_flex_ep_backend_ops()
    op = getattr(torch.ops.inductor, op_name, None)
    if op is None:
        raise RuntimeError(
            f"flex_ep requires torch.ops.inductor.{op_name} to be registered."
        )
    return op


@torch.library.impl(_flex_ep_lib, "ep_allgather", "CompositeExplicitAutograd")
def _ep_allgather(
    output: torch.Tensor,
    input: torch.Tensor,
    buffers_cuda_ptrs: torch.Tensor,
    offs_output: int,
    ep_rank: int,
) -> torch.Tensor:
    op = _require_inductor_router_op("ep_allgather")
    op(output, input, buffers_cuda_ptrs, offs_output, ep_rank)
    return output


@torch.library.register_fake("_flex_ep::ep_allgather")
def _ep_allgather_fake(
    output: torch.Tensor,
    input: torch.Tensor,
    buffers_cuda_ptrs: torch.Tensor,
    offs_output: int,
    ep_rank: int,
) -> torch.Tensor:
    del input, buffers_cuda_ptrs, offs_output, ep_rank
    return torch.empty_like(output)


@torch.library.impl(
    _flex_ep_lib, "router_compute_all_expert_offsets", "CompositeExplicitAutograd"
)
def _router_compute_all_expert_offsets(
    all_expert_counts: torch.Tensor,
    ep_rank: int,
    local_experts: int,
    token_alignment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    op = _require_inductor_router_op("router_compute_all_expert_offsets")
    return op(all_expert_counts, ep_rank, local_experts, token_alignment)


@torch.library.register_fake("_flex_ep::router_compute_all_expert_offsets")
def _router_compute_all_expert_offsets_fake(
    all_expert_counts: torch.Tensor,
    ep_rank: int,
    local_experts: int,
    token_alignment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del ep_rank, token_alignment
    ep_size = all_expert_counts.shape[0]
    return (
        all_expert_counts.new_empty(
            (ep_size, local_experts, ep_size + 1),
            dtype=torch.int32,
        ),
        all_expert_counts.new_empty((), dtype=torch.int64),
        all_expert_counts.new_empty((local_experts + 1,), dtype=torch.int32),
    )


@torch.library.impl(
    _flex_ep_lib, "router_compute_dest_offsets", "CompositeExplicitAutograd"
)
def _router_compute_dest_offsets(
    topk_idx: torch.Tensor,
    recv_ofs: torch.Tensor,
    ep_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    op = _require_inductor_router_op("router_compute_dest_offsets")
    return op(topk_idx, recv_ofs, ep_size)


@torch.library.register_fake("_flex_ep::router_compute_dest_offsets")
def _router_compute_dest_offsets_fake(
    topk_idx: torch.Tensor,
    recv_ofs: torch.Tensor,
    ep_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del recv_ofs, ep_size
    return (
        topk_idx.new_empty(topk_idx.shape, dtype=torch.int32),
        topk_idx.new_empty(topk_idx.shape, dtype=torch.int64),
    )


@torch.library.impl(_flex_ep_lib, "router_dispatch", "CompositeExplicitAutograd")
def _router_dispatch(
    my_tokens: torch.Tensor,
    my_scaling_factors: torch.Tensor | None,
    my_topk_weights: torch.Tensor | None,
    dest_ranks: torch.Tensor,
    dest_offsets: torch.Tensor,
    buffers_cuda_ptrs: torch.Tensor,
    dispatch_recv_buffer: torch.Tensor,
    dispatch_recv_buffer_scaling_factors: torch.Tensor,
    dispatch_recv_origin_global_token_id: torch.Tensor,
    dispatch_recv_weights: torch.Tensor,
    offs_recv_tokens: int,
    offs_recv_scaling_factors: int,
    offs_recv_weights: int,
    offs_recv_origin_global_token_id: int,
    ep_rank: int,
    num_ctas: int,
    max_B: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    op = _require_inductor_router_op("router_dispatch")
    op(
        my_tokens,
        my_scaling_factors,
        my_topk_weights,
        dest_ranks,
        dest_offsets,
        buffers_cuda_ptrs,
        dispatch_recv_buffer,
        dispatch_recv_buffer_scaling_factors,
        dispatch_recv_origin_global_token_id,
        dispatch_recv_weights,
        offs_recv_tokens,
        offs_recv_scaling_factors,
        offs_recv_weights,
        offs_recv_origin_global_token_id,
        ep_rank,
        num_ctas,
        max_B,
    )
    return (
        dispatch_recv_buffer,
        dispatch_recv_buffer_scaling_factors,
        dispatch_recv_origin_global_token_id,
        dispatch_recv_weights,
    )


@torch.library.register_fake("_flex_ep::router_dispatch")
def _router_dispatch_fake(
    my_tokens: torch.Tensor,
    my_scaling_factors: torch.Tensor | None,
    my_topk_weights: torch.Tensor | None,
    dest_ranks: torch.Tensor,
    dest_offsets: torch.Tensor,
    buffers_cuda_ptrs: torch.Tensor,
    dispatch_recv_buffer: torch.Tensor,
    dispatch_recv_buffer_scaling_factors: torch.Tensor,
    dispatch_recv_origin_global_token_id: torch.Tensor,
    dispatch_recv_weights: torch.Tensor,
    offs_recv_tokens: int,
    offs_recv_scaling_factors: int,
    offs_recv_weights: int,
    offs_recv_origin_global_token_id: int,
    ep_rank: int,
    num_ctas: int,
    max_B: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(dispatch_recv_buffer),
        torch.empty_like(dispatch_recv_buffer_scaling_factors),
        torch.empty_like(dispatch_recv_origin_global_token_id),
        torch.empty_like(dispatch_recv_weights),
    )


@torch.library.impl(_flex_ep_lib, "router_combine", "CompositeExplicitAutograd")
def _router_combine(
    send_tokens: torch.Tensor,
    send_scale_factors: torch.Tensor | None,
    send_weights: torch.Tensor | None,
    expert_begin_offset_per_ep: torch.Tensor,
    token_send_end: torch.Tensor,
    send_origin_global_token_id: torch.Tensor,
    buffers_cuda_ptrs: torch.Tensor,
    combine_recv_buffer: torch.Tensor,
    combine_recv_scale_factors: torch.Tensor,
    combine_recv_weights: torch.Tensor,
    offs_combine_recv_tokens: int,
    offs_combine_recv_scale_factors: int,
    offs_combine_recv_weights: int,
    ep_rank: int,
    B: int,
    topk: int,
    num_ctas: int,
    max_B: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    op = _require_inductor_router_op("router_combine")
    op(
        send_tokens,
        send_scale_factors,
        send_weights,
        expert_begin_offset_per_ep,
        token_send_end,
        send_origin_global_token_id,
        buffers_cuda_ptrs,
        combine_recv_buffer,
        combine_recv_scale_factors,
        combine_recv_weights,
        offs_combine_recv_tokens,
        offs_combine_recv_scale_factors,
        offs_combine_recv_weights,
        ep_rank,
        B,
        topk,
        num_ctas,
        max_B,
    )
    return combine_recv_buffer, combine_recv_scale_factors, combine_recv_weights


@torch.library.register_fake("_flex_ep::router_combine")
def _router_combine_fake(
    send_tokens: torch.Tensor,
    send_scale_factors: torch.Tensor | None,
    send_weights: torch.Tensor | None,
    expert_begin_offset_per_ep: torch.Tensor,
    token_send_end: torch.Tensor,
    send_origin_global_token_id: torch.Tensor,
    buffers_cuda_ptrs: torch.Tensor,
    combine_recv_buffer: torch.Tensor,
    combine_recv_scale_factors: torch.Tensor,
    combine_recv_weights: torch.Tensor,
    offs_combine_recv_tokens: int,
    offs_combine_recv_scale_factors: int,
    offs_combine_recv_weights: int,
    ep_rank: int,
    B: int,
    topk: int,
    num_ctas: int,
    max_B: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(combine_recv_buffer),
        torch.empty_like(combine_recv_scale_factors),
        torch.empty_like(combine_recv_weights),
    )


@torch.library.impl(_flex_ep_lib, "barrier_arrive", "CompositeExplicitAutograd")
def _barrier_arrive(
    flag: torch.Tensor,
    dependency: torch.Tensor,
    nonce: int = 0,
) -> torch.Tensor:
    del dependency, nonce
    op = _require_inductor_router_op("barrier_arrive")
    return op(flag)


@torch.library.impl(_flex_ep_lib, "barrier_wait", "CompositeExplicitAutograd")
def _barrier_wait(
    input: torch.Tensor,
    cuda_ptrs: torch.Tensor,
    offs_flag: int,
    expected: torch.Tensor,
    timeout_s: float = 5.0,
) -> torch.Tensor:
    op = _require_inductor_router_op("barrier_wait")
    op(cuda_ptrs, offs_flag, expected, timeout_s)
    return input.clone()


@torch.library.impl(_flex_ep_lib, "barrier_wait_no_clone", "CompositeExplicitAutograd")
def _barrier_wait_no_clone(
    input: torch.Tensor,
    cuda_ptrs: torch.Tensor,
    offs_flag: int,
    expected: torch.Tensor,
    timeout_s: float = 5.0,
) -> torch.Tensor:
    op = _require_inductor_router_op("barrier_wait")
    op(cuda_ptrs, offs_flag, expected, timeout_s)
    return input


@torch.library.register_fake("_flex_ep::barrier_arrive")
def _barrier_arrive_fake(
    flag: torch.Tensor,
    dependency: torch.Tensor,
    nonce: int = 0,
) -> torch.Tensor:
    del dependency, nonce
    return flag.new_empty((1,), dtype=torch.int32)


@torch.library.register_fake("_flex_ep::barrier_wait")
def _barrier_wait_fake(
    input: torch.Tensor,
    cuda_ptrs: torch.Tensor,
    offs_flag: int,
    expected: torch.Tensor,
    timeout_s: float = 5.0,
) -> torch.Tensor:
    del cuda_ptrs, offs_flag, expected, timeout_s
    return torch.empty_like(input)


@torch.library.register_fake("_flex_ep::barrier_wait_no_clone")
def _barrier_wait_no_clone_fake(
    input: torch.Tensor,
    cuda_ptrs: torch.Tensor,
    offs_flag: int,
    expected: torch.Tensor,
    timeout_s: float = 5.0,
) -> torch.Tensor:
    del cuda_ptrs, offs_flag, expected, timeout_s
    return input


@torch.library.impl(
    _flex_ep_lib, "swiglu_forward_with_offsets", "CompositeExplicitAutograd"
)
def _swiglu_forward_with_offsets_impl(
    y1: torch.Tensor,
    token_end: torch.Tensor,
) -> torch.Tensor:
    op = _require_inductor_router_op("swiglu_forward_with_offsets")
    return op(y1, token_end)


@torch.library.register_fake("_flex_ep::swiglu_forward_with_offsets")
def _swiglu_forward_with_offsets_fake(
    y1: torch.Tensor,
    token_end: torch.Tensor,
) -> torch.Tensor:
    del token_end
    return _swiglu_reference(y1)


@torch.library.impl(
    _flex_ep_lib, "swiglu_backward_with_offsets", "CompositeExplicitAutograd"
)
def _swiglu_backward_with_offsets_impl(
    dy2: torch.Tensor,
    y1: torch.Tensor,
    token_end: torch.Tensor,
) -> torch.Tensor:
    op = _require_inductor_router_op("swiglu_backward_with_offsets")
    return op(dy2, y1, token_end)


@torch.library.register_fake("_flex_ep::swiglu_backward_with_offsets")
def _swiglu_backward_with_offsets_fake(
    dy2: torch.Tensor,
    y1: torch.Tensor,
    token_end: torch.Tensor,
) -> torch.Tensor:
    del token_end
    return _swiglu_backward_reference(dy2, y1)


@torch.library.impl(_flex_ep_lib, "clone_valid_prefix", "CompositeExplicitAutograd")
def _clone_valid_prefix_impl(
    input: torch.Tensor,
    token_end: torch.Tensor,
) -> torch.Tensor:
    op = _require_inductor_router_op("clone_valid_prefix")
    return op(input, token_end)


@torch.library.register_fake("_flex_ep::clone_valid_prefix")
def _clone_valid_prefix_fake(
    input: torch.Tensor,
    token_end: torch.Tensor,
) -> torch.Tensor:
    del token_end
    return torch.empty_like(input)


@torch.library.impl(_flex_ep_lib, "zfill_ranges_inplace", "CompositeExplicitAutograd")
def _zfill_ranges_inplace_impl(
    input: torch.Tensor,
    begin_ofs: torch.Tensor,
    end_ofs: torch.Tensor,
    max_values_per_batch: int,
) -> torch.Tensor:
    op = _require_inductor_router_op("zfill_ranges_inplace")
    op(input, begin_ofs, end_ofs, max_values_per_batch)
    return input


@torch.library.register_fake("_flex_ep::zfill_ranges_inplace")
def _zfill_ranges_inplace_fake(
    input: torch.Tensor,
    begin_ofs: torch.Tensor,
    end_ofs: torch.Tensor,
    max_values_per_batch: int,
) -> torch.Tensor:
    del begin_ofs, end_ofs, max_values_per_batch
    return torch.empty_like(input)


@torch.library.impl(_flex_ep_lib, "fill_i64_inplace", "CompositeExplicitAutograd")
def _fill_i64_inplace_impl(input: torch.Tensor, value: int) -> torch.Tensor:
    if input.dtype != torch.int64:
        raise ValueError("fill_i64_inplace expects an int64 tensor.")
    input.fill_(value)
    return input


@torch.library.register_fake("_flex_ep::fill_i64_inplace")
def _fill_i64_inplace_fake(input: torch.Tensor, value: int) -> torch.Tensor:
    del value
    return torch.empty_like(input)
