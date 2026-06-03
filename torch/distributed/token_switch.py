# mypy: allow-untyped-defs
from __future__ import annotations

import abc
from dataclasses import dataclass

import torch
from torch.distributed.distributed_c10d import ProcessGroup


@dataclass(frozen=True)
class Routing:
    handle: object
    topk_idx: torch.Tensor


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
    def dispatch(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        out_tokens: torch.Tensor,
        out_topk_weights: torch.Tensor,
        out_topk_idx: torch.Tensor,
    ) -> None:
        """Route tokens to experts; writes ``out_*`` tensors.

        Uses ``topk_idx`` from the provided :class:`Routing`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def combine(
        self,
        routing: Routing,
        expert_tokens: torch.Tensor,
        out_tokens: torch.Tensor,
    ) -> None:
        """Gather expert outputs back to token order; writes ``out_tokens``."""
        raise NotImplementedError


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

    def dispatch(
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

    def combine(
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
