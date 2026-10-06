# mypy: allow-untyped-defs
from __future__ import annotations

import abc
import importlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup


def _prepare_nccl_ep(ep: Any) -> None:
    # Editable installs can place native artifacts outside the Python source tree.
    home = Path(ep.__file__).resolve().parent / "share" / "nccl_ep"
    ep_home = Path(os.environ.get("NCCL_EP_HOME") or home)
    nccl_home = Path(os.environ.get("NCCL_HOME") or home)
    source = Path(
        os.environ.get("NCCL_EP_JIT_SOURCE_DIR") or ep_home / "include" / "nccl_ep"
    )
    include = Path(
        os.environ.get("NCCL_EP_JIT_BUILD_INCLUDE_DIR") or nccl_home / "include"
    )
    for header in (
        source.parent / "nccl_ep.h",
        source / "common.hpp",
        source / "ep_enums.h",
        source / "device" / "ht_ep.cuh",
        include / "nccl.h",
        include / "nccl_device.h",
        include / "nccl_device" / "impl" / "core__types.h",
    ):
        if not header.is_file():
            raise ImportError(
                f"NCCL EP runtime JIT header is missing: {header}. "
                "Rebuild PyTorch with USE_NCCL_EP=1 or check NCCL_EP_HOME, "
                "NCCL_HOME, NCCL_EP_JIT_SOURCE_DIR, and "
                "NCCL_EP_JIT_BUILD_INCLUDE_DIR."
            )

    if not os.environ.get("NCCL_EP_HOME"):
        os.environ["NCCL_EP_HOME"] = str(home)
    if not os.environ.get("NCCL_HOME"):
        os.environ["NCCL_HOME"] = str(home)

    # Use the target machine's toolkit when nvcc is selected through PATH.
    if not any(
        os.environ.get(name)
        for name in ("NCCL_EP_JIT_CUDA_INCLUDE_DIR", "CUDA_HOME", "CUDA_PATH")
    ):
        nvcc = shutil.which(
            os.environ.get("NCCL_EP_JIT_NVCC") or os.environ.get("NVCC") or "nvcc"
        )
        if nvcc:
            cuda_include = Path(nvcc).resolve().parent.parent / "include"
            if cuda_include.is_dir():
                os.environ["NCCL_EP_JIT_CUDA_INCLUDE_DIR"] = str(cuda_include)


def _import_nccl_ep() -> Any:
    try:
        ep = importlib.import_module("torch._nccl_ep")
    except ImportError as e:
        if isinstance(e, ModuleNotFoundError) and e.name == "torch._nccl_ep":
            raise ImportError(
                "TokenSwitchNCCL requires PyTorch built with USE_NCCL_EP=1 "
                "and NCCL_EP_SOURCE_DIR pointing to nccl-extensions/nccl_ep."
            ) from e
        raise ImportError(
            f"Failed to load torch._nccl_ep or its native dependencies: {e}"
        ) from e

    _prepare_nccl_ep(ep)
    return ep


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
        ctx.tokens_shape = tokens.shape
        return out_tokens, out_topk_weights, out_topk_idx

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx: Any,
        grad_out_tokens: torch.Tensor,
        grad_out_topk_weights: torch.Tensor | None,
        grad_out_topk_idx: torch.Tensor | None,
    ) -> tuple[None, None, torch.Tensor, None, None]:
        grad_tokens = grad_out_tokens.new_zeros(ctx.tokens_shape)
        ctx.ts._combine(ctx.routing, grad_out_tokens.contiguous(), grad_tokens)
        return None, None, grad_tokens, None, None


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
        ctx.top_k = routing.topk_idx.shape[1]
        return out_tokens

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx: Any, grad_out_tokens: torch.Tensor
    ) -> tuple[None, None, torch.Tensor]:
        ts = ctx.ts
        H = ctx.expert_shape[-1]
        N = grad_out_tokens.shape[0]
        K = ctx.top_k
        dtype = ctx.expert_dtype
        # Allocate the full-size dispatch output buffer matching the layout's
        # expected shape, dispatch the gradient into it, then slice back to
        # ctx.expert_shape so the returned grad matches the combine input.
        max_recv = ts._max_recv_tokens_per_rank
        grad_expert_full, dummy_out_weights, dummy_out_idx = ts._alloc_dispatch_outputs(
            ctx.routing,
            grad_out_tokens,
            grad_out_tokens.new_zeros(N, K, dtype=torch.float32),
            max_recv,
            H,
            K,
        )
        grad_expert_full = grad_expert_full.to(dtype)
        dummy_weights = grad_out_tokens.new_zeros(N, K, dtype=torch.float32)
        ctx.ts._dispatch(
            ctx.routing,
            grad_out_tokens.to(dtype).contiguous(),
            dummy_weights,
            grad_expert_full,
            dummy_out_weights,
            dummy_out_idx,
        )
        # Slice the leading dim back to expert_shape[0] (no-op for LL+EM where
        # the full buffer was already the right size).
        M = ctx.expert_shape[0]
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
        *,
        layout: str,
    ) -> Routing:
        """Create expert routing for the current phase (e.g. top-k indices).

        ``per_expert_token_counts`` is optional 1D int32, length >= local experts:
        output buffer for per-expert receive counts (NCCL EP ``RECV_EXPERT_COUNTER``).
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
        """Route tokens to experts.

        Returns ``(out_tokens, out_topk_weights, out_topk_idx)``.  For
        expert-major layouts ``out_topk_idx`` is ``None``; for LL+expert-major
        ``out_topk_weights`` is also ``None``.

        With ``out=(out_tokens, out_topk_weights, out_topk_idx)``: writes to the
        provided buffers and returns them; no autograd support.
        Without ``out``: allocates output buffers and returns the tuple with
        autograd support.  ``max_recv_tokens`` is required when ``out`` is not
        provided.  ``topk_weights`` receives no gradient (routing metadata).
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

    Routing handles retain their EP group. Complete outstanding CUDA work and
    release routing handles and token switches before explicitly destroying the
    process group.
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
        """Create expert routing for this phase; pass to :meth:`dispatch` / :meth:`combine`.

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
