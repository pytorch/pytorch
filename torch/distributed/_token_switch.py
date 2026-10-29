# mypy: allow-untyped-defs
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup


def _import_nccl_ep() -> Any:
    try:
        # pyrefly: ignore [missing-import]  # built only with USE_NCCL_EP
        import torch._nccl_ep as _ep
    except ModuleNotFoundError as e:
        if e.name != "torch._nccl_ep":
            raise
        raise ImportError(
            "TokenSwitchNCCL requires PyTorch built with USE_NCCL_EP=1."
        ) from e

    return _ep


@dataclass(frozen=True, slots=True)
class Routing:
    handle: object
    topk_idx: torch.Tensor
    layout: str = "flat"  # "flat" | "expert_major"


class _DispatchAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        ts: TokenSwitch,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        max_recv_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        _N, H = tokens.shape
        K = topk_weights.shape[1]
        out_tokens, out_topk_weights, out_topk_idx = ts._alloc_dispatch_outputs(
            routing, tokens, topk_weights, max_recv_tokens, H, K
        )
        ts._dispatch(
            routing, tokens, topk_weights, out_tokens, out_topk_weights, out_topk_idx
        )
        ctx.ts = ts
        ctx.routing = routing
        return out_tokens, out_topk_weights, out_topk_idx

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx: Any,
        grad_out_tokens: torch.Tensor,
        grad_out_topk_weights: torch.Tensor | None,
        grad_out_topk_idx: torch.Tensor | None,
    ) -> tuple[None, None, torch.Tensor, torch.Tensor, None]:
        if grad_out_topk_weights is None:
            raise NotImplementedError(
                "TokenSwitch dispatch backward requires received topk_weights"
            )
        grad_tokens = grad_out_tokens.new_zeros(
            ctx.routing.topk_idx.shape[0], grad_out_tokens.shape[-1]
        )
        grad_weights = grad_out_topk_weights.new_zeros(ctx.routing.topk_idx.shape)
        ctx.ts._dispatch_backward(
            ctx.routing,
            grad_out_tokens.contiguous(),
            grad_out_topk_weights.contiguous(),
            grad_tokens,
            grad_weights,
        )
        return None, None, grad_tokens, grad_weights, None


class _CombineAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        ts: TokenSwitch,
        routing: Routing,
        expert_tokens: torch.Tensor,
    ) -> torch.Tensor:
        N = routing.topk_idx.shape[0]
        H = expert_tokens.shape[-1]
        out_tokens = expert_tokens.new_zeros(N, H)
        ts._combine(routing, expert_tokens, out_tokens)
        ctx.ts = ts
        ctx.routing = routing
        ctx.expert_shape = expert_tokens.shape
        ctx.expert_dtype = expert_tokens.dtype
        return out_tokens

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx: Any, grad_out_tokens: torch.Tensor
    ) -> tuple[None, None, torch.Tensor]:
        ts = ctx.ts
        H = ctx.expert_shape[-1]
        grad_expert_full = grad_out_tokens.new_zeros(
            ts.max_recv_tokens_per_rank, H, dtype=ctx.expert_dtype
        )
        ts._combine_backward(
            ctx.routing,
            grad_out_tokens.to(ctx.expert_dtype).contiguous(),
            grad_expert_full,
        )
        return None, None, grad_expert_full[: ctx.expert_shape[0]].contiguous()


class TokenSwitch(abc.ABC):
    """Abstract token routing switch (e.g. expert-parallel dispatch / combine).

    Typical usage: :meth:`create_routing`, then :meth:`dispatch` / :meth:`combine`.

    Contract every backend implements:

    - A :class:`Routing` fixes the receive-slot layout. Every :meth:`dispatch` and
      :meth:`combine` on the same Routing uses that layout, so dispatching again
      (e.g. the backward of :meth:`combine`) writes the new tokens into the same
      slots. A backend may compute the layout in :meth:`create_routing` or in the
      first :meth:`dispatch` on the Routing.
    - Calls on different Routings may be interleaved in any order, e.g. two
      dispatches before either combine.
    - Flat layout: a token is received once per destination rank, however many of
      its top-k experts live there. Each ``out_topk_idx`` row holds the local ids of
      that rank's experts among the token's top-k, and -1 in its other entries;
      ``out_topk_weights`` holds the matching weights, and 0 where the id is -1.
      The order of the entries within a row is up to the backend.
    - Expert-major layout: one receive slot per (token, local expert).
    - :meth:`combine` sums, without weights, one row per receive slot that came
      from the token. Callers apply ``topk_weights`` and reduce over their local
      experts before combining. With an identity expert and the flat layout, a
      token comes back multiplied by its number of destination ranks.
    - Rows of the dispatch outputs past the number of received slots are
      unspecified.
    """

    @property
    @abc.abstractmethod
    def max_recv_tokens_per_rank(self) -> int:
        """Upper bound on receive slots per rank; sizes full dispatch outputs."""
        raise NotImplementedError

    @abc.abstractmethod
    def create_routing(
        self,
        topk_idx: torch.Tensor,
        per_expert_token_counts: torch.Tensor | None = None,
        *,
        layout: str,
    ) -> Routing:
        """Create expert routing for the current phase (e.g. top-k indices).

        ``per_expert_token_counts`` is optional 1D int32, length >= local experts:
        output buffer for per-expert receive counts (NCCL EP ``RECV_EXPERT_COUNTER``),
        i.e. the number of received (token, local expert) pairs per local expert.
        ``layout`` selects the dispatch output memory layout.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _alloc_dispatch_outputs(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        max_recv_tokens: int,
        H: int,
        K: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Allocate dispatch output buffers with shapes matching routing.layout."""
        raise NotImplementedError

    @abc.abstractmethod
    def _dispatch(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        out_tokens: torch.Tensor,
        out_topk_weights: torch.Tensor | None,
        out_topk_idx: torch.Tensor | None,
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

    @abc.abstractmethod
    def _combine_backward(
        self,
        routing: Routing,
        grad_tokens: torch.Tensor,
        out_expert_grads: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _dispatch_backward(
        self,
        routing: Routing,
        grad_recv_tokens: torch.Tensor,
        grad_recv_topk_weights: torch.Tensor,
        out_token_grads: torch.Tensor,
        out_topk_weight_grads: torch.Tensor,
    ) -> None:
        r"""Reverse dispatch into both source-shaped token and routing-weight buffers."""
        raise NotImplementedError

    def dispatch(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        max_recv_tokens: int | None = None,
        *,
        out: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
        | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        r"""Route tokens to experts.

        Returns ``(out_tokens, out_topk_weights, out_topk_idx)``.  For
        expert-major layouts ``out_topk_idx`` is ``None``; for LL+expert-major
        ``out_topk_weights`` is also ``None``.

        With ``out=(out_tokens, out_topk_weights, out_topk_idx)``: writes to the
        provided buffers and returns them; no autograd support.
        Without ``out``: allocates output buffers and returns the tuple with
        autograd support.  ``max_recv_tokens`` is required when ``out`` is not
        provided. Backends must implement gradients for both tokens and routing
        weights. NCCL supports first-order gradients for FP32 ``topk_weights``.
        """
        if out is not None:
            self._dispatch(routing, tokens, topk_weights, *out)
            return out
        if max_recv_tokens is None:
            raise ValueError("max_recv_tokens is required when out= is not provided")
        return _DispatchAutograd.apply(
            self, routing, tokens, topk_weights, max_recv_tokens
        )  # type: ignore[return-value]

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
    """Token switch backed by NCCL EP (:func:`ncclEpCreateGroup` / dispatch / combine).

    The dispatch output layout is chosen per routing via
    :meth:`create_routing`'s ``layout`` argument.
    """

    def __init__(
        self,
        process_group: ProcessGroup,
        num_experts: int,
        max_dispatch_tokens_per_rank: int,
        max_recv_tokens_per_rank: int,
        max_token_bytes: int,
    ) -> None:
        self._ep = _import_nccl_ep()
        ep = self._ep

        self._layout_map = {
            "flat": ep.Layout.FLAT,
            "expert_major": ep.Layout.EXPERT_MAJOR,
        }

        self._max_recv_tokens_per_rank = max_recv_tokens_per_rank
        self._group = ep._NcclEpGroup.create(
            process_group,
            num_experts,
            max_dispatch_tokens_per_rank,
            max_recv_tokens_per_rank,
            max_token_bytes,
        )

    @property
    def max_recv_tokens_per_rank(self) -> int:
        return self._max_recv_tokens_per_rank

    def _alloc_dispatch_outputs(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        max_recv_tokens: int,
        H: int,
        K: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if routing.layout == "expert_major":
            # topk_weights is 1D (one scalar weight per recv slot);
            # topk_idx is not populated (nullptr in NCCL EP).
            return (
                tokens.new_zeros(max_recv_tokens, H),
                topk_weights.new_zeros(max_recv_tokens),
                None,
            )
        # flat: standard 2D outputs
        return (
            tokens.new_zeros(max_recv_tokens, H),
            topk_weights.new_zeros(max_recv_tokens, K),
            tokens.new_zeros(max_recv_tokens, K, dtype=torch.int64),
        )

    def create_routing(
        self,
        topk_idx: torch.Tensor,
        per_expert_token_counts: torch.Tensor | None = None,
        *,
        layout: str,
    ) -> Routing:
        r"""Create expert routing for this phase; pass to :meth:`dispatch` / :meth:`combine`.

        ``topk_idx`` must have dtype ``torch.int32`` or ``torch.int64``.
        The NCCL EP handle retains an int64 copy of int32 indices.
        ``layout`` (required) controls the dispatch output memory layout:
        ``"flat"`` or ``"expert_major"``.
        """
        if layout not in self._layout_map:
            raise ValueError(
                f"layout must be one of {list(self._layout_map)}; got {layout!r}"
            )
        handle = self._ep._NcclEpHandle.create(
            self._group,
            topk_idx,
            per_expert_token_counts,
            self._layout_map[layout],
        )
        return Routing(handle=handle, topk_idx=topk_idx, layout=layout)

    def _dispatch(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        out_tokens: torch.Tensor,
        out_topk_weights: torch.Tensor | None,
        out_topk_idx: torch.Tensor | None,
    ) -> None:
        self._ep._nccl_ep_dispatch(
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
        self._ep._nccl_ep_combine(
            routing.handle,
            expert_tokens,
            out_tokens,
        )

    def _combine_backward(
        self,
        routing: Routing,
        grad_tokens: torch.Tensor,
        out_expert_grads: torch.Tensor,
    ) -> None:
        self._ep._nccl_ep_dispatch_backward(
            routing.handle, grad_tokens, out_expert_grads
        )

    def _dispatch_backward(
        self,
        routing: Routing,
        grad_recv_tokens: torch.Tensor,
        grad_recv_topk_weights: torch.Tensor,
        out_token_grads: torch.Tensor,
        out_topk_weight_grads: torch.Tensor,
    ) -> None:
        self._ep._nccl_ep_combine_backward(
            routing.handle,
            grad_recv_tokens,
            grad_recv_topk_weights,
            out_token_grads,
            out_topk_weight_grads,
        )
