# mypy: allow-untyped-defs
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed.distributed_c10d import ProcessGroup


@dataclass(frozen=True)
class Routing:
    handle: object
    topk_idx: torch.Tensor


class _DispatchAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        ts: "TokenSwitch",
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        max_recv_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _N, H = tokens.shape
        K = topk_weights.shape[1]
        out_tokens = tokens.new_zeros(max_recv_tokens, H)
        out_topk_weights = topk_weights.new_zeros(max_recv_tokens, K)
        out_topk_idx = routing.topk_idx.new_zeros(max_recv_tokens, K)
        ts._dispatch(routing, tokens, topk_weights, out_tokens, out_topk_weights, out_topk_idx)
        ctx.ts = ts
        ctx.routing = routing
        ctx.tokens_shape = tokens.shape
        return out_tokens, out_topk_weights, out_topk_idx

    @staticmethod
    def backward(
        ctx: Any,
        grad_out_tokens: torch.Tensor,
        grad_out_topk_weights: torch.Tensor,
        grad_out_topk_idx: torch.Tensor,
    ) -> tuple:
        grad_tokens = grad_out_tokens.new_zeros(ctx.tokens_shape)
        ctx.ts._combine(ctx.routing, grad_out_tokens.contiguous(), grad_tokens)
        return None, None, grad_tokens, None, None


class _CombineAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        ts: "TokenSwitch",
        routing: Routing,
        expert_tokens: torch.Tensor,
    ) -> torch.Tensor:
        N = routing.topk_idx.shape[0]
        H = expert_tokens.shape[1]
        out_tokens = expert_tokens.new_zeros(N, H)
        ts._combine(routing, expert_tokens, out_tokens)
        ctx.ts = ts
        ctx.routing = routing
        ctx.expert_shape = expert_tokens.shape
        ctx.expert_dtype = expert_tokens.dtype
        ctx.top_k = routing.topk_idx.shape[1]
        return out_tokens

    @staticmethod
    def backward(ctx: Any, grad_out_tokens: torch.Tensor) -> tuple:
        M, H = ctx.expert_shape
        N = grad_out_tokens.shape[0]
        K = ctx.top_k
        dtype = ctx.expert_dtype
        # ncclEpDispatch requires the output buffer sized to the group's
        # max_recv_tokens_per_rank, regardless of what shape expert_tokens had
        # in forward (often a slice like out_tokens[:M]). Allocate full-size,
        # run dispatch, then slice to ctx.expert_shape so the returned grad
        # matches the input that produced it.
        max_recv = ctx.ts._max_recv_tokens_per_rank
        grad_expert_full = grad_out_tokens.new_zeros(max_recv, H).to(dtype)
        dummy_weights = grad_out_tokens.new_zeros(N, K, dtype=torch.float32)
        dummy_out_weights = grad_out_tokens.new_zeros(max_recv, K, dtype=torch.float32)
        dummy_out_idx = ctx.routing.topk_idx.new_zeros(max_recv, K)
        ctx.ts._dispatch(
            ctx.routing,
            grad_out_tokens.to(dtype).contiguous(),
            dummy_weights,
            grad_expert_full,
            dummy_out_weights,
            dummy_out_idx,
        )
        return None, None, grad_expert_full[:M].contiguous()


class TokenSwitch(abc.ABC):
    """Abstract token routing switch (e.g. expert-parallel dispatch / combine).

    Typical usage: :meth:`create_routing`, then :meth:`dispatch` / :meth:`combine`.
    """

    @abc.abstractmethod
    def create_routing(
        self,
        topk_idx: torch.Tensor,
        per_expert_token_counts: torch.Tensor | None = None,
    ) -> Routing:
        """Create expert routing for the current phase (e.g. top-k indices).

        ``per_expert_token_counts`` is optional 1D int32, length >= local experts:
        output buffer for per-expert receive counts (NCCL EP ``RECV_EXPERT_COUNTER``).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _dispatch(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        out_tokens: torch.Tensor,
        out_topk_weights: torch.Tensor,
        out_topk_idx: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _combine(
        self,
        routing: Routing,
        expert_tokens: torch.Tensor,
        out_tokens: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    def dispatch(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        max_recv_tokens: int | None = None,
        *,
        out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        With ``out=(out_tokens, out_topk_weights, out_topk_idx)``: writes to the provided
        buffers and returns them; no autograd support.
        Without ``out``: allocates output buffers and returns
        ``(out_tokens, out_topk_weights, out_topk_idx)`` with autograd support.
        ``max_recv_tokens`` is required when ``out`` is not provided.
        ``topk_weights`` receives no gradient (routing metadata).
        """
        if out is not None:
            self._dispatch(routing, tokens, topk_weights, *out)
            return out
        if max_recv_tokens is None:
            raise ValueError("max_recv_tokens is required when out= is not provided")
        return _DispatchAutograd.apply(self, routing, tokens, topk_weights, max_recv_tokens)  # type: ignore[return-value]

    def combine(
        self,
        routing: Routing,
        expert_tokens: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gather expert outputs back to token order.

        With ``out=out_tokens``: writes to the provided buffer and returns it;
        no autograd support.
        Without ``out``: allocates an output buffer and returns it with autograd support.
        """
        if out is not None:
            self._combine(routing, expert_tokens, out)
            return out
        return _CombineAutograd.apply(self, routing, expert_tokens)  # type: ignore[return-value]


class TokenSwitchNCCL(TokenSwitch):
    """Token switch backed by NCCL EP (:func:`ncclEpCreateGroup` / dispatch / combine)."""

    def __init__(
        self,
        process_group: ProcessGroup,
        num_experts: int,
        max_dispatch_tokens_per_rank: int,
        max_recv_tokens_per_rank: int,
        max_token_bytes: int,
    ) -> None:
        c10d = torch._C._distributed_c10d
        if not hasattr(c10d, "_NcclEpGroup"):
            raise RuntimeError(
                "TokenSwitchNCCL requires a build with NCCL EP (USE_NCCL_EP)."
            )
        self._max_recv_tokens_per_rank = max_recv_tokens_per_rank
        # NCCL_EP_AUTO (0) for qp count and channel count
        self._group = c10d._NcclEpGroup.create(
            process_group,
            num_experts,
            max_dispatch_tokens_per_rank,
            max_recv_tokens_per_rank,
            max_token_bytes,
            0,
            0,
        )

    def create_routing(
        self,
        topk_idx: torch.Tensor,
        per_expert_token_counts: torch.Tensor | None = None,
    ) -> Routing:
        """Create expert routing for this phase; pass to :meth:`dispatch` / :meth:`combine`."""
        c10d = torch._C._distributed_c10d
        handle = c10d._NcclEpHandle.create(
            self._group,
            topk_idx,
            per_expert_token_counts,
        )
        return Routing(handle=handle, topk_idx=topk_idx)

    def _dispatch(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        out_tokens: torch.Tensor,
        out_topk_weights: torch.Tensor,
        out_topk_idx: torch.Tensor,
    ) -> None:
        torch._C._distributed_c10d._nccl_ep_dispatch(
            routing.handle,
            tokens,
            topk_weights,
            out_tokens,
            out_topk_weights,
            out_topk_idx,
        )

    def _combine(
        self,
        routing: Routing,
        expert_tokens: torch.Tensor,
        out_tokens: torch.Tensor,
    ) -> None:
        torch._C._distributed_c10d._nccl_ep_combine(
            routing.handle,
            expert_tokens,
            out_tokens,
        )
