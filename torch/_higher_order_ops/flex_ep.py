# mypy: allow-untyped-defs
"""Autograd-aware BF16 expert-parallel MoE block."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

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


TMI_FLAT_LEN = 8
LOCAL_EXPERTS_START_INDEX = 6


def _is_int_like(x: Any) -> bool:
    return isinstance(x, (int, torch.SymInt))


def _validate_router_operands(router_operands: tuple[Any, ...]) -> None:
    for i, operand in enumerate(router_operands):
        if not (isinstance(operand, torch.Tensor) or _is_int_like(operand)):
            raise TypeError(
                "flex_ep router_operands must be flat tensors/ints, "
                f"got {type(operand).__name__} at index {i}"
            )


def _validate_flex_ep_inputs(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    router_operands: tuple[Any, ...],
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
    _validate_router_operands(router_operands)


def _call_flat_fn(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    lifted_args: tuple[Any, ...],
) -> Any:
    return fn(*args, *lifted_args)


def _as_tuple(x: Any) -> tuple[Any, ...]:
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return (x,)


def _expand_x_for_topk(x: torch.Tensor, topk: int) -> torch.Tensor:
    return x.unsqueeze(1).expand(-1, topk, -1).contiguous()


def _grouped_mm_offsets(tmi_flat: tuple[Any, ...]) -> torch.Tensor:
    if len(tmi_flat) != TMI_FLAT_LEN:
        raise RuntimeError(
            f"flex_ep expected {TMI_FLAT_LEN} TokenMappingInfo values, "
            f"got {len(tmi_flat)}"
        )
    local_experts_start = tmi_flat[LOCAL_EXPERTS_START_INDEX]
    if not isinstance(local_experts_start, torch.Tensor):
        raise RuntimeError("flex_ep local_experts_start must be a tensor")
    return local_experts_start[1:].to(torch.int32)


def _token_end_from_tmi(tmi_flat: tuple[Any, ...]) -> torch.Tensor:
    if len(tmi_flat) != TMI_FLAT_LEN:
        raise RuntimeError(
            f"flex_ep expected {TMI_FLAT_LEN} TokenMappingInfo values, "
            f"got {len(tmi_flat)}"
        )
    local_experts_start = tmi_flat[LOCAL_EXPERTS_START_INDEX]
    if not isinstance(local_experts_start, torch.Tensor):
        raise RuntimeError("flex_ep local_experts_start must be a tensor")
    return local_experts_start[-1:].to(torch.int64)


def _split_tmi_and_router_operands(
    tmi_and_router_operands: tuple[Any, ...],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    if len(tmi_and_router_operands) < TMI_FLAT_LEN:
        raise RuntimeError(
            f"flex_ep_backward expected at least {TMI_FLAT_LEN} TokenMappingInfo values"
        )
    return (
        tuple(tmi_and_router_operands[:TMI_FLAT_LEN]),
        tuple(tmi_and_router_operands[TMI_FLAT_LEN:]),
    )


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


def _swiglu(y1: torch.Tensor) -> torch.Tensor:
    return torch.ops._flex_ep.swiglu_forward(y1)


def _swiglu_with_offsets(
    y1: torch.Tensor,
    token_end: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._flex_ep.swiglu_forward_with_offsets(y1, token_end)


def _swiglu_backward(
    dy2: torch.Tensor,
    y1: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._flex_ep.swiglu_backward(dy2, y1)


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
    dispatch_fn: Callable[..., Any],
    combine_fn: Callable[..., Any],
    router_operands: tuple[Any, ...],
    *,
    topk: int,
    dispatch_lifted_args: tuple[Any, ...] = (),
    combine_lifted_args: tuple[Any, ...] = (),
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[Any, ...],
    torch.Tensor,
]:
    x_expanded = _expand_x_for_topk(x, topk)
    dispatch_out = _as_tuple(
        _call_flat_fn(
            dispatch_fn,
            (x_expanded, topk_idx, *router_operands),
            dispatch_lifted_args,
        )
    )
    if len(dispatch_out) != 1 + TMI_FLAT_LEN:
        raise RuntimeError(
            "flex_ep dispatch_fn must return (recv_x, *tmi_flat) with "
            f"{TMI_FLAT_LEN} TokenMappingInfo values"
        )
    recv_x = dispatch_out[0]
    if not isinstance(recv_x, torch.Tensor):
        raise RuntimeError("flex_ep dispatch_fn first return must be recv_x tensor")
    tmi_flat = tuple(dispatch_out[1:])
    offs = _grouped_mm_offsets(tmi_flat)
    token_end = _token_end_from_tmi(tmi_flat)
    y3, y1, y2 = _moe_block_forward_bf16(recv_x, w13, w2, offs, token_end)
    y = _call_flat_fn(
        combine_fn,
        (y3, *tmi_flat, *router_operands),
        combine_lifted_args,
    )
    if not isinstance(y, torch.Tensor):
        raise RuntimeError("flex_ep combine_fn must return a tensor")
    return y, recv_x, y1, y2, tmi_flat, offs


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
    *tmi_and_router_operands: Any,
    _combine_bwd_lifted_args: tuple[Any, ...] = (),
    _dispatch_bwd_lifted_args: tuple[Any, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tmi_flat, router_operands = _split_tmi_and_router_operands(tmi_and_router_operands)
    _validate_router_operands(router_operands)
    dy3 = _call_flat_fn(
        combine_bwd_fn,
        (dy, *tmi_flat, *router_operands),
        _combine_bwd_lifted_args,
    )
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
    dxpn = _call_flat_fn(
        dispatch_bwd_fn,
        (dx_recv, *tmi_flat, *router_operands),
        _dispatch_bwd_lifted_args,
    )
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
        dispatch_fn: Callable[..., Any],
        combine_fn: Callable[..., Any],
        combine_bwd_fn: Callable[..., Any],
        dispatch_bwd_fn: Callable[..., Any],
        router_operands: tuple[Any, ...],
        topk: int,
        dispatch_lifted_args: tuple[Any, ...],
        combine_lifted_args: tuple[Any, ...],
        combine_bwd_lifted_args: tuple[Any, ...],
        dispatch_bwd_lifted_args: tuple[Any, ...],
    ) -> torch.Tensor:
        with torch.no_grad():
            y, recv_x, y1, y2, tmi_flat, offs = _flex_ep_forward(
                x,
                topk_idx,
                w13,
                w2,
                dispatch_fn,
                combine_fn,
                router_operands,
                topk=topk,
                dispatch_lifted_args=dispatch_lifted_args,
                combine_lifted_args=combine_lifted_args,
            )
        ctx.dispatch_bwd_fn = dispatch_bwd_fn
        ctx.combine_bwd_fn = combine_bwd_fn
        ctx.router_operands = router_operands
        ctx.tmi_flat = tmi_flat
        ctx.combine_bwd_lifted_args = combine_bwd_lifted_args
        ctx.dispatch_bwd_lifted_args = dispatch_bwd_lifted_args
        token_end = _token_end_from_tmi(tmi_flat)
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
        tmi_flat = ctx.tmi_flat
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
            *tmi_flat,
            *router_operands,
            _combine_bwd_lifted_args=ctx.combine_bwd_lifted_args,
            _dispatch_bwd_lifted_args=ctx.dispatch_bwd_lifted_args,
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
            None,
            None,
            None,
        )


def _flex_ep_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    dispatch_fn: Callable[..., Any],
    combine_fn: Callable[..., Any],
    combine_bwd_fn: Callable[..., Any],
    dispatch_bwd_fn: Callable[..., Any],
    *router_operands: Any,
    num_experts: int,
    ep_rank: int,
    ep_size: int,
    max_tokens: int,
    topk: int,
    num_ctas: int = 152,
    _dispatch_lifted_args: tuple[Any, ...] = (),
    _combine_lifted_args: tuple[Any, ...] = (),
    _combine_bwd_lifted_args: tuple[Any, ...] = (),
    _dispatch_bwd_lifted_args: tuple[Any, ...] = (),
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
        dispatch_fn,
        combine_fn,
        combine_bwd_fn,
        dispatch_bwd_fn,
        router_operands,
        topk,
        _dispatch_lifted_args,
        _combine_lifted_args,
        _combine_bwd_lifted_args,
        _dispatch_bwd_lifted_args,
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
        dispatch_fn: Router subgraph called as
            ``dispatch_fn(x_expanded, topk_idx, *router_operands,
            *_dispatch_lifted_args)``. It must return ``(recv_x, *tmi_flat)``,
            where ``tmi_flat`` has ``TMI_FLAT_LEN`` TokenMappingInfo entries.
        combine_fn: Router subgraph called as
            ``combine_fn(y3, *tmi_flat, *router_operands,
            *_combine_lifted_args)``. It must return the HOP output tensor.
        combine_bwd_fn: Backward router subgraph called as
            ``combine_bwd_fn(dy, *tmi_flat, *router_operands,
            *_combine_bwd_lifted_args)``. It must return the gradient for the
            local expert output ``y3``.
        dispatch_bwd_fn: Backward router subgraph called as
            ``dispatch_bwd_fn(dx_recv, *tmi_flat, *router_operands,
            *_dispatch_bwd_lifted_args)``. It must return expanded input
            gradients with shape ``[num_tokens, topk, hidden]``.
        router_operands: Flat tensor/int operands captured by the router
            subgraphs, such as workspace tensors, offsets, and EP rank metadata.
        num_experts: Total number of global experts.
        ep_rank: This rank within the expert-parallel group.
        ep_size: Number of ranks in the expert-parallel group.
        max_tokens: Maximum local token count used for router buffer sizing.
        topk: Number of selected experts per token.
        num_ctas: CTA count used by backend router kernels.
        _dispatch_lifted_args: Extra operands appended to ``dispatch_fn`` after
            ``router_operands`` when graph transforms lift closed-over values.
        _combine_lifted_args: Extra operands appended to ``combine_fn``.
        _combine_bwd_lifted_args: Extra operands appended to ``combine_bwd_fn``.
        _dispatch_bwd_lifted_args: Extra operands appended to
            ``dispatch_bwd_fn``.
    """

    def __init__(self) -> None:
        super().__init__("flex_ep", cacheable=True)
        self.subgraph_indexes = [4, 5, 6, 7]

    def __call__(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        dispatch_fn: Callable[..., Any],
        combine_fn: Callable[..., Any],
        combine_bwd_fn: Callable[..., Any],
        dispatch_bwd_fn: Callable[..., Any],
        *router_operands: Any,
        num_experts: int,
        ep_rank: int,
        ep_size: int,
        max_tokens: int,
        topk: int,
        num_ctas: int = 152,
        _dispatch_lifted_args: tuple[Any, ...] = (),
        _combine_lifted_args: tuple[Any, ...] = (),
        _combine_bwd_lifted_args: tuple[Any, ...] = (),
        _dispatch_bwd_lifted_args: tuple[Any, ...] = (),
    ) -> torch.Tensor:
        return super().__call__(
            x,
            topk_idx,
            w13,
            w2,
            dispatch_fn,
            combine_fn,
            combine_bwd_fn,
            dispatch_bwd_fn,
            *router_operands,
            num_experts=num_experts,
            ep_rank=ep_rank,
            ep_size=ep_size,
            max_tokens=max_tokens,
            topk=topk,
            num_ctas=num_ctas,
            _dispatch_lifted_args=_dispatch_lifted_args,
            _combine_lifted_args=_combine_lifted_args,
            _combine_bwd_lifted_args=_combine_bwd_lifted_args,
            _dispatch_bwd_lifted_args=_dispatch_bwd_lifted_args,
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
        *tmi_and_router_operands: Any,
        _combine_bwd_lifted_args: tuple[Any, ...] = (),
        _dispatch_bwd_lifted_args: tuple[Any, ...] = (),
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
            *tmi_and_router_operands,
            _combine_bwd_lifted_args=_combine_bwd_lifted_args,
            _dispatch_bwd_lifted_args=_dispatch_bwd_lifted_args,
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
    *tmi_and_router_operands: Any,
    _combine_bwd_lifted_args: tuple[Any, ...] = (),
    _dispatch_bwd_lifted_args: tuple[Any, ...] = (),
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
        *tmi_and_router_operands,
        _combine_bwd_lifted_args=_combine_bwd_lifted_args,
        _dispatch_bwd_lifted_args=_dispatch_bwd_lifted_args,
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
    tmi_and_router_operands: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tmi_flat, router_operands = _split_tmi_and_router_operands(tmi_and_router_operands)
    if not isinstance(combine_bwd_fn, torch.fx.GraphModule):
        combine_bwd_fn = _maybe_reenter_make_fx(combine_bwd_fn)(
            dy,
            *tmi_flat,
            *router_operands,
            *kwargs["_combine_bwd_lifted_args"],
        )
    if not isinstance(dispatch_bwd_fn, torch.fx.GraphModule):
        dispatch_bwd_fn = _maybe_reenter_make_fx(dispatch_bwd_fn)(
            recv_x,
            *tmi_flat,
            *router_operands,
            *kwargs["_dispatch_bwd_lifted_args"],
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
        *tmi_and_router_operands,
        **kwargs,
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
        *tmi_and_router_operands,
    )
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, node_args)  # type: ignore[attr-defined]
    proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)  # type: ignore[attr-defined]
    with torch.fx.experimental.proxy_tensor.set_original_aten_op(flex_ep_backward):
        out_proxy = mode.tracer.create_proxy(
            "call_function",
            flex_ep_backward,
            proxy_args,
            proxy_kwargs,
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
    *tmi_and_router_operands: Any,
    _combine_bwd_lifted_args: tuple[Any, ...] = (),
    _dispatch_bwd_lifted_args: tuple[Any, ...] = (),
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
        tmi_and_router_operands,
        {
            "_combine_bwd_lifted_args": _combine_bwd_lifted_args,
            "_dispatch_bwd_lifted_args": _dispatch_bwd_lifted_args,
        },
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
    *tmi_and_router_operands: Any,
    _combine_bwd_lifted_args: tuple[Any, ...] = (),
    _dispatch_bwd_lifted_args: tuple[Any, ...] = (),
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
            *tmi_and_router_operands,
            _combine_bwd_lifted_args=_combine_bwd_lifted_args,
            _dispatch_bwd_lifted_args=_dispatch_bwd_lifted_args,
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
    *tmi_and_router_operands: Any,
    _combine_bwd_lifted_args: tuple[Any, ...] = (),
    _dispatch_bwd_lifted_args: tuple[Any, ...] = (),
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
            tmi_and_router_operands,
            _combine_bwd_lifted_args,
            _dispatch_bwd_lifted_args,
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
        tmi_and_router_operands,
        combine_bwd_lifted_args,
        dispatch_bwd_lifted_args,
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
            *tmi_and_router_operands,
            _combine_bwd_lifted_args=combine_bwd_lifted_args,
            _dispatch_bwd_lifted_args=dispatch_bwd_lifted_args,
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
    dispatch_fn: Callable[..., Any],
    combine_fn: Callable[..., Any],
    combine_bwd_fn: Callable[..., Any],
    dispatch_bwd_fn: Callable[..., Any],
    *router_operands: Any,
    num_experts: int,
    ep_rank: int,
    ep_size: int,
    max_tokens: int,
    topk: int,
    num_ctas: int = 152,
    _dispatch_lifted_args: tuple[Any, ...] = (),
    _combine_lifted_args: tuple[Any, ...] = (),
    _combine_bwd_lifted_args: tuple[Any, ...] = (),
    _dispatch_bwd_lifted_args: tuple[Any, ...] = (),
) -> torch.Tensor:
    return _flex_ep_impl(
        x,
        topk_idx,
        w13,
        w2,
        dispatch_fn,
        combine_fn,
        combine_bwd_fn,
        dispatch_bwd_fn,
        *router_operands,
        num_experts=num_experts,
        ep_rank=ep_rank,
        ep_size=ep_size,
        max_tokens=max_tokens,
        topk=topk,
        num_ctas=num_ctas,
        _dispatch_lifted_args=_dispatch_lifted_args,
        _combine_lifted_args=_combine_lifted_args,
        _combine_bwd_lifted_args=_combine_bwd_lifted_args,
        _dispatch_bwd_lifted_args=_dispatch_bwd_lifted_args,
    )


@flex_ep.py_impl(DispatchKey.Autograd)
def flex_ep_autograd(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    dispatch_fn: Callable[..., Any],
    combine_fn: Callable[..., Any],
    combine_bwd_fn: Callable[..., Any],
    dispatch_bwd_fn: Callable[..., Any],
    *router_operands: Any,
    num_experts: int,
    ep_rank: int,
    ep_size: int,
    max_tokens: int,
    topk: int,
    num_ctas: int = 152,
    _dispatch_lifted_args: tuple[Any, ...] = (),
    _combine_lifted_args: tuple[Any, ...] = (),
    _combine_bwd_lifted_args: tuple[Any, ...] = (),
    _dispatch_bwd_lifted_args: tuple[Any, ...] = (),
) -> torch.Tensor:
    return _flex_ep_impl(
        x,
        topk_idx,
        w13,
        w2,
        dispatch_fn,
        combine_fn,
        combine_bwd_fn,
        dispatch_bwd_fn,
        *router_operands,
        num_experts=num_experts,
        ep_rank=ep_rank,
        ep_size=ep_size,
        max_tokens=max_tokens,
        topk=topk,
        num_ctas=num_ctas,
        _dispatch_lifted_args=_dispatch_lifted_args,
        _combine_lifted_args=_combine_lifted_args,
        _combine_bwd_lifted_args=_combine_bwd_lifted_args,
        _dispatch_bwd_lifted_args=_dispatch_bwd_lifted_args,
    )


def _trace_flex_ep_proxy(
    mode: ProxyTorchDispatchMode,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    dispatch_fn: Callable[..., Any],
    combine_fn: Callable[..., Any],
    combine_bwd_fn: Callable[..., Any],
    dispatch_bwd_fn: Callable[..., Any],
    router_operands: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    example_out = flex_ep(
        x,
        topk_idx,
        w13,
        w2,
        dispatch_fn,
        combine_fn,
        combine_bwd_fn,
        dispatch_bwd_fn,
        *router_operands,
        **kwargs,
    )
    if not isinstance(mode.tracer, torch.fx.Tracer):
        raise AssertionError(
            f"expected proxy_mode.tracer to be torch.fx.Tracer, got {type(mode.tracer)}"
        )
    for name, fn in (
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
        dispatch_fn,
        combine_fn,
        combine_bwd_fn,
        dispatch_bwd_fn,
        *router_operands,
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
    dispatch_fn: Callable[..., Any],
    combine_fn: Callable[..., Any],
    combine_bwd_fn: Callable[..., Any],
    dispatch_bwd_fn: Callable[..., Any],
    *router_operands: Any,
    num_experts: int,
    ep_rank: int,
    ep_size: int,
    max_tokens: int,
    topk: int,
    num_ctas: int = 152,
    _dispatch_lifted_args: tuple[Any, ...] = (),
    _combine_lifted_args: tuple[Any, ...] = (),
    _combine_bwd_lifted_args: tuple[Any, ...] = (),
    _dispatch_bwd_lifted_args: tuple[Any, ...] = (),
) -> torch.Tensor:
    return _trace_flex_ep_proxy(
        mode,
        x,
        topk_idx,
        w13,
        w2,
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
            "_dispatch_lifted_args": _dispatch_lifted_args,
            "_combine_lifted_args": _combine_lifted_args,
            "_combine_bwd_lifted_args": _combine_bwd_lifted_args,
            "_dispatch_bwd_lifted_args": _dispatch_bwd_lifted_args,
        },
    )


@flex_ep.py_impl(FakeTensorMode)
def flex_ep_fake_tensor_mode(
    mode: FakeTensorMode,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    dispatch_fn: Callable[..., Any],
    combine_fn: Callable[..., Any],
    combine_bwd_fn: Callable[..., Any],
    dispatch_bwd_fn: Callable[..., Any],
    *router_operands: Any,
    num_experts: int,
    ep_rank: int,
    ep_size: int,
    max_tokens: int,
    topk: int,
    num_ctas: int = 152,
    _dispatch_lifted_args: tuple[Any, ...] = (),
    _combine_lifted_args: tuple[Any, ...] = (),
    _combine_bwd_lifted_args: tuple[Any, ...] = (),
    _dispatch_bwd_lifted_args: tuple[Any, ...] = (),
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
            dispatch_fn,
            combine_fn,
            router_operands,
            topk=topk,
            dispatch_lifted_args=_dispatch_lifted_args,
            combine_lifted_args=_combine_lifted_args,
        )
        return y


@flex_ep.py_functionalize_impl
def flex_ep_functionalize(
    ctx,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    dispatch_fn: Callable[..., Any],
    combine_fn: Callable[..., Any],
    combine_bwd_fn: Callable[..., Any],
    dispatch_bwd_fn: Callable[..., Any],
    *router_operands: Any,
    num_experts: int,
    ep_rank: int,
    ep_size: int,
    max_tokens: int,
    topk: int,
    num_ctas: int = 152,
    _dispatch_lifted_args: tuple[Any, ...] = (),
    _combine_lifted_args: tuple[Any, ...] = (),
    _combine_bwd_lifted_args: tuple[Any, ...] = (),
    _dispatch_bwd_lifted_args: tuple[Any, ...] = (),
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
            _dispatch_lifted_args,
            _combine_lifted_args,
            _combine_bwd_lifted_args,
            _dispatch_bwd_lifted_args,
        ),
    )
    (
        x,
        topk_idx,
        w13,
        w2,
        router_operands,
        dispatch_lifted_args,
        combine_lifted_args,
        combine_bwd_lifted_args,
        dispatch_bwd_lifted_args,
    ) = unwrapped
    with ctx.redispatch_to_next():
        with suspend_functionalization(), disable_proxy_modes_tracing():
            functional_dispatch_fn = ctx.functionalize(dispatch_fn)
            functional_combine_fn = ctx.functionalize(combine_fn)
            functional_combine_bwd_fn = ctx.functionalize(combine_bwd_fn)
            functional_dispatch_bwd_fn = ctx.functionalize(dispatch_bwd_fn)
        out = flex_ep(
            x,
            topk_idx,
            w13,
            w2,
            functional_dispatch_fn,
            functional_combine_fn,
            functional_combine_bwd_fn,
            functional_dispatch_bwd_fn,
            *router_operands,
            num_experts=num_experts,
            ep_rank=ep_rank,
            ep_size=ep_size,
            max_tokens=max_tokens,
            topk=topk,
            num_ctas=num_ctas,
            _dispatch_lifted_args=dispatch_lifted_args,
            _combine_lifted_args=combine_lifted_args,
            _combine_bwd_lifted_args=combine_bwd_lifted_args,
            _dispatch_bwd_lifted_args=dispatch_bwd_lifted_args,
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
_flex_ep_lib.define("swiglu_forward(Tensor y1) -> Tensor")
_flex_ep_lib.define("swiglu_backward(Tensor dy2, Tensor y1) -> Tensor")
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

_FLEX_EP_SIDE_EFFECT_OPS = [
    has_side_effect(getattr(torch.ops._flex_ep, name).default)
    for name in (
        "ep_allgather",
        "router_dispatch",
        "router_combine",
        "barrier_arrive",
        "barrier_wait",
        "barrier_wait_no_clone",
        "zfill_ranges_inplace",
        "fill_i64_inplace",
    )
]


TOKEN_ALIGNMENT = 128


def _align_up_tensor(x: torch.Tensor, alignment: int) -> torch.Tensor:
    return ((x + alignment - 1) // alignment) * alignment


_INDUCTOR_ROUTER_OP_NAMES = {
    "ep_allgather": "flex_ep_allgather",
    "router_compute_all_expert_offsets": "flex_ep_router_compute_all_expert_offsets",
    "router_compute_dest_offsets": "flex_ep_router_compute_dest_offsets",
    "router_dispatch": "flex_ep_router_dispatch",
    "router_combine": "flex_ep_router_combine",
    "barrier_arrive": "flex_ep_barrier_arrive",
    "barrier_wait": "flex_ep_barrier_wait",
    "swiglu_forward": "flex_ep_swiglu_forward",
    "swiglu_backward": "flex_ep_swiglu_backward",
    "swiglu_forward_with_offsets": "flex_ep_swiglu_forward_with_offsets",
    "swiglu_backward_with_offsets": "flex_ep_swiglu_backward_with_offsets",
    "clone_valid_prefix": "flex_ep_clone_valid_prefix",
    "zfill_ranges_inplace": "flex_ep_zfill_ranges_inplace",
}


def _inductor_router_op(name: str) -> Any | None:
    try:
        return getattr(torch.ops.inductor, _INDUCTOR_ROUTER_OP_NAMES[name])
    except AttributeError:
        return None


def _require_inductor_router_op(name: str) -> Any:
    op = _inductor_router_op(name)
    if op is None:
        op_name = _INDUCTOR_ROUTER_OP_NAMES[name]
        raise RuntimeError(
            "flex_ep NVL RouterEP backend requires torch.ops.inductor "
            f"{op_name}. Register the RouterEP backend kernels before using "
            "EP>1 NVL execution."
        )
    return op


def _decode_global_token_id(
    origin_ids: torch.Tensor,
    *,
    max_B: int,
    topk: int,
) -> torch.Tensor:
    return torch.remainder(origin_ids, max_B * topk)


@torch.library.impl(_flex_ep_lib, "ep_allgather", "CompositeExplicitAutograd")
def _ep_allgather(
    output: torch.Tensor,
    input: torch.Tensor,
    buffers_cuda_ptrs: torch.Tensor,
    offs_output: int,
    ep_rank: int,
) -> torch.Tensor:
    op = _inductor_router_op("ep_allgather")
    if op is not None:
        op(output, input, buffers_cuda_ptrs, offs_output, ep_rank)
        return output
    if buffers_cuda_ptrs.numel() != 1:
        import torch.distributed as dist

        del offs_output, ep_rank
        dist.all_gather_into_tensor(output, input.contiguous())
        return output
    if ep_rank != 0:
        _require_inductor_router_op("ep_allgather")
    output[0].copy_(input)
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
    op = _inductor_router_op("router_compute_all_expert_offsets")
    if op is not None:
        return op(all_expert_counts, ep_rank, local_experts, token_alignment)

    ep_size = all_expert_counts.shape[0]
    offsets = all_expert_counts.new_empty(
        (ep_size, local_experts, ep_size + 1),
        dtype=torch.int32,
    )
    expert_start = all_expert_counts.new_empty(
        (ep_size, local_experts + 1),
        dtype=torch.int32,
    )
    grand_total = all_expert_counts.sum(0).new_empty((ep_size,), dtype=torch.int64)
    for dest in range(ep_size):
        counts = all_expert_counts[
            :,
            dest * local_experts : (dest + 1) * local_experts,
        ]
        total_per_expert = counts.sum(0)
        grand_total[dest] = total_per_expert.sum()
        aligned = _align_up_tensor(total_per_expert, token_alignment).to(torch.int32)
        starts = torch.cat(
            (
                torch.zeros(1, device=all_expert_counts.device, dtype=torch.int32),
                aligned.cumsum(0),
            )
        )
        expert_start[dest].copy_(starts)
        offsets[dest, :, 0].copy_(starts[:-1])
        offsets[dest, :, 1:].copy_(
            starts[:-1].unsqueeze(1) + counts.cumsum(0, dtype=torch.int32).T
        )
    return offsets, grand_total[ep_rank].clone(), expert_start[ep_rank].clone()


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
    op = _inductor_router_op("router_compute_dest_offsets")
    if op is not None:
        return op(topk_idx, recv_ofs, ep_size)

    B, topk = topk_idx.shape
    num_experts = recv_ofs.shape[0]
    local_experts = num_experts // ep_size
    flat_idx = topk_idx.reshape(-1)
    dest_ranks = torch.div(flat_idx, local_experts, rounding_mode="floor").to(
        torch.int32
    )
    dest_offsets = torch.empty(
        flat_idx.shape, device=topk_idx.device, dtype=torch.int64
    )
    for expert in range(num_experts):
        is_mine = flat_idx == expert
        rank_in_expert = is_mine.to(torch.int64).cumsum(0) - 1
        dest_offsets = torch.where(
            is_mine,
            recv_ofs[expert].to(torch.int64) + rank_in_expert,
            dest_offsets,
        )
    return dest_ranks.view(B, topk), dest_offsets.view(B, topk)


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
    op = _inductor_router_op("router_dispatch")
    if op is not None:
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
    if buffers_cuda_ptrs.numel() != 1 or ep_rank != 0:
        _require_inductor_router_op("router_dispatch")

    B, topk, D = my_tokens.shape
    del offs_recv_tokens, offs_recv_scaling_factors, offs_recv_weights, num_ctas
    flat_tokens = my_tokens.reshape(B * topk, D)
    dispatch_recv_buffer.view(my_tokens.dtype).view(-1, D)[dest_offsets.reshape(-1)] = (
        flat_tokens
    )
    if offs_recv_origin_global_token_id >= 0:
        token_ids = torch.arange(B * topk, device=my_tokens.device, dtype=torch.int64)
        dispatch_recv_origin_global_token_id[dest_offsets.reshape(-1)] = (
            ep_rank * max_B * topk + token_ids
        )
    if my_topk_weights is not None:
        dispatch_recv_weights[dest_offsets.reshape(-1)] = my_topk_weights.reshape(-1)
    if my_scaling_factors is not None:
        sf_dim = my_scaling_factors.shape[-1]
        dispatch_recv_buffer_scaling_factors.view(my_scaling_factors.dtype).view(
            -1,
            sf_dim,
        )[dest_offsets.reshape(-1)] = my_scaling_factors.reshape(-1, sf_dim)
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
    op = _inductor_router_op("router_combine")
    if op is not None:
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
    if buffers_cuda_ptrs.numel() != 1 or ep_rank != 0:
        _require_inductor_router_op("router_combine")

    del expert_begin_offset_per_ep, token_send_end, num_ctas
    del offs_combine_recv_tokens, offs_combine_recv_scale_factors
    del offs_combine_recv_weights
    D = send_tokens.shape[-1]
    dest_idx = _decode_global_token_id(
        send_origin_global_token_id,
        max_B=max_B,
        topk=topk,
    )
    combine_recv_buffer.view(send_tokens.dtype).view(max_B * topk, D)[dest_idx] = (
        send_tokens
    )
    if send_weights is not None:
        combine_recv_weights.view(-1)[dest_idx] = send_weights
    if send_scale_factors is not None:
        sf_dim = send_scale_factors.shape[-1]
        combine_recv_scale_factors.view(send_scale_factors.dtype).view(
            max_B * topk,
            sf_dim,
        )[dest_idx] = send_scale_factors
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
    op = _inductor_router_op("barrier_arrive")
    if op is not None:
        return op(flag)
    flag.add_(1)
    return flag.clone()


@torch.library.impl(_flex_ep_lib, "barrier_wait", "CompositeExplicitAutograd")
def _barrier_wait(
    input: torch.Tensor,
    cuda_ptrs: torch.Tensor,
    offs_flag: int,
    expected: torch.Tensor,
    timeout_s: float = 5.0,
) -> torch.Tensor:
    op = _inductor_router_op("barrier_wait")
    if op is not None:
        op(cuda_ptrs, offs_flag, expected, timeout_s)
        return input.clone()
    if cuda_ptrs.numel() != 1:
        _require_inductor_router_op("barrier_wait")
    del offs_flag, expected, timeout_s
    return input.clone()


@torch.library.impl(_flex_ep_lib, "barrier_wait_no_clone", "CompositeExplicitAutograd")
def _barrier_wait_no_clone(
    input: torch.Tensor,
    cuda_ptrs: torch.Tensor,
    offs_flag: int,
    expected: torch.Tensor,
    timeout_s: float = 5.0,
) -> torch.Tensor:
    op = _inductor_router_op("barrier_wait")
    if op is not None:
        op(cuda_ptrs, offs_flag, expected, timeout_s)
        return input
    if cuda_ptrs.numel() != 1:
        _require_inductor_router_op("barrier_wait")
    del offs_flag, expected, timeout_s
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


@torch.library.impl(_flex_ep_lib, "swiglu_forward", "CompositeExplicitAutograd")
def _swiglu_forward_impl(y1: torch.Tensor) -> torch.Tensor:
    op = _inductor_router_op("swiglu_forward")
    if (
        op is not None
        and y1.is_cuda
        and y1.dtype == torch.bfloat16
        and y1.is_contiguous()
    ):
        return op(y1)
    return _swiglu_reference(y1)


@torch.library.register_fake("_flex_ep::swiglu_forward")
def _swiglu_forward_fake(y1: torch.Tensor) -> torch.Tensor:
    return _swiglu_reference(y1)


@torch.library.impl(_flex_ep_lib, "swiglu_backward", "CompositeExplicitAutograd")
def _swiglu_backward_impl(dy2: torch.Tensor, y1: torch.Tensor) -> torch.Tensor:
    op = _inductor_router_op("swiglu_backward")
    if (
        op is not None
        and dy2.is_cuda
        and y1.is_cuda
        and dy2.dtype == torch.bfloat16
        and y1.dtype == torch.bfloat16
        and dy2.is_contiguous()
        and y1.is_contiguous()
    ):
        return op(dy2, y1)
    return _swiglu_backward_reference(dy2, y1)


@torch.library.register_fake("_flex_ep::swiglu_backward")
def _swiglu_backward_fake(dy2: torch.Tensor, y1: torch.Tensor) -> torch.Tensor:
    return _swiglu_backward_reference(dy2, y1)


@torch.library.impl(
    _flex_ep_lib, "swiglu_forward_with_offsets", "CompositeExplicitAutograd"
)
def _swiglu_forward_with_offsets_impl(
    y1: torch.Tensor,
    token_end: torch.Tensor,
) -> torch.Tensor:
    op = _inductor_router_op("swiglu_forward_with_offsets")
    if (
        op is not None
        and y1.is_cuda
        and token_end.is_cuda
        and y1.dtype == torch.bfloat16
        and token_end.dtype == torch.int64
        and token_end.numel() == 1
        and y1.is_contiguous()
    ):
        return op(y1, token_end)
    return _swiglu_reference(y1)


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
    op = _inductor_router_op("swiglu_backward_with_offsets")
    if (
        op is not None
        and dy2.is_cuda
        and y1.is_cuda
        and token_end.is_cuda
        and dy2.dtype == torch.bfloat16
        and y1.dtype == torch.bfloat16
        and token_end.dtype == torch.int64
        and token_end.numel() == 1
        and dy2.is_contiguous()
        and y1.is_contiguous()
    ):
        return op(dy2, y1, token_end)
    return _swiglu_backward_reference(dy2, y1)


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
    op = _inductor_router_op("clone_valid_prefix")
    if (
        op is not None
        and input.is_cuda
        and token_end.is_cuda
        and token_end.dtype == torch.int64
        and token_end.numel() == 1
        and input.is_contiguous()
    ):
        return op(input, token_end)
    return input.clone()


@torch.library.register_fake("_flex_ep::clone_valid_prefix")
def _clone_valid_prefix_fake(
    input: torch.Tensor,
    token_end: torch.Tensor,
) -> torch.Tensor:
    del token_end
    return torch.empty_like(input)


@torch.library.impl(_flex_ep_lib, "zfill_ranges_inplace", "CompositeExplicitAutograd")
def _zfill_ranges_inplace(
    input: torch.Tensor,
    begin_ofs: torch.Tensor,
    end_ofs: torch.Tensor,
    max_values_per_batch: int,
) -> torch.Tensor:
    op = _inductor_router_op("zfill_ranges_inplace")
    if op is not None:
        op(input, begin_ofs, end_ofs, max_values_per_batch)
        return input
    for start, end in zip(begin_ofs.cpu().tolist(), end_ofs.cpu().tolist()):
        input[start:end].zero_()
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
def _fill_i64_inplace(input: torch.Tensor, value: int) -> torch.Tensor:
    input.fill_(value)
    return input


@torch.library.register_fake("_flex_ep::fill_i64_inplace")
def _fill_i64_inplace_fake(input: torch.Tensor, value: int) -> torch.Tensor:
    del value
    return torch.empty_like(input)
