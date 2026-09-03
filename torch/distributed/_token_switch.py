# mypy: allow-untyped-defs
from __future__ import annotations

import abc
import ctypes
import importlib.util
import os
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup


# Keeps preloaded shared libraries resident for the process lifetime.
_NCCL_EP_KEEPALIVE: list[ctypes.CDLL] = []


def _find_pkg_dir(name: str) -> str | None:
    # Locate an installed package's directory without importing it. find_spec
    # raises (rather than returning None) when a dotted name's parent namespace
    # is absent, so guard that.
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return spec.submodule_search_locations[0]


def _prepare_nccl4py() -> None:
    # Dynamic (USE_SYSTEM_NCCL=ON / wheel) build only: the extension NEEDED-links
    # libnccl_ep.so, which the nccl4py wheel provides at runtime. Point nccl-ep's
    # JIT at nccl4py's EP headers (NCCL_EP_HOME) and the nvidia.nccl wheel's NCCL
    # headers (NCCL_HOME), and make libnccl_ep resolvable, before importing the
    # extension (its libnccl dependency is already loaded by torch). The static
    # build bakes all of this in and never reaches here.
    nccl_pkg = _find_pkg_dir("nccl")
    if nccl_pkg is None:
        raise ImportError(
            "TokenSwitchNCCL needs the 'nccl4py' package for this build's "
            "libnccl_ep.so and runtime JIT headers. Install it with "
            "`pip install nccl4py==0.3.1`."
        )
    ep_dir = os.path.join(nccl_pkg, "ep")
    if "NCCL_EP_HOME" not in os.environ:
        ep_headers = os.path.join(ep_dir, "include", "nccl_ep")
        if not os.path.isdir(ep_headers):
            raise ImportError(
                f"nccl4py at {nccl_pkg} is missing the EP JIT headers expected "
                f"at {ep_headers}; reinstall nccl4py or set NCCL_EP_HOME."
            )
        os.environ["NCCL_EP_HOME"] = ep_dir

    # Point the JIT at NCCL headers. libnccl.so itself needs no preload here: a
    # USE_SYSTEM_NCCL=ON torch (the only build that reaches this path) already
    # dynamically loaded the system libnccl, so libnccl_ep.so's NEEDED
    # libnccl.so.2 resolves to that already-loaded copy.
    nvidia_nccl = _find_pkg_dir("nvidia.nccl")
    if nvidia_nccl is not None and os.path.isdir(os.path.join(nvidia_nccl, "include")):
        os.environ.setdefault("NCCL_HOME", nvidia_nccl)

    # nccl4py ships libnccl_ep unversioned (libnccl_ep.so) while the extension's
    # NEEDED entry is its SONAME libnccl_ep.so.0; load it by path to register
    # that SONAME for the import below.
    lib = os.path.join(ep_dir, "lib", "libnccl_ep.so")
    if os.path.isfile(lib):
        _NCCL_EP_KEEPALIVE.append(ctypes.CDLL(lib, mode=ctypes.RTLD_LOCAL))


def _import_nccl_ep() -> Any:
    # The EP bindings live in the optional torch._nccl_ep extension (USE_NCCL_EP).
    # A USE_SYSTEM_NCCL=OFF build links libnccl_ep statically and bakes its JIT
    # header paths, so the extension imports directly -- self-contained, no
    # nccl4py. A USE_SYSTEM_NCCL=ON build NEEDED-links libnccl_ep from the nccl4py
    # wheel, so the first import fails until nccl4py is set up; fall back to that
    # and retry.
    try:
        # pyrefly: ignore [missing-import]  # built only with USE_NCCL_EP
        import torch._nccl_ep as _ep

        return _ep
    except ImportError:
        pass

    _prepare_nccl4py()
    try:
        # pyrefly: ignore [missing-import]  # built only with USE_NCCL_EP
        import torch._nccl_ep as _ep
    except ImportError as e:
        raise ImportError(
            "torch._nccl_ep is unavailable; this PyTorch was not built with "
            "USE_NCCL_EP (or, for a USE_SYSTEM_NCCL=ON build, the nccl4py wheel "
            "is missing)."
        ) from e

    return _ep


@dataclass(frozen=True, slots=True)
class Routing:
    handle: object
    topk_idx: torch.Tensor


class _DispatchAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        ts: TokenSwitch,
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
        grad_out_topk_weights: torch.Tensor,
        grad_out_topk_idx: torch.Tensor,
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
    # pyrefly: ignore [bad-override]
    def backward(
        ctx: Any, grad_out_tokens: torch.Tensor
    ) -> tuple[None, None, torch.Tensor]:
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
    """Token switch backed by NCCL EP (:func:`ncclEpCreateGroup` / dispatch / combine)."""

    def __init__(
        self,
        process_group: ProcessGroup,
        num_experts: int,
        max_dispatch_tokens_per_rank: int,
        max_recv_tokens_per_rank: int,
        max_token_bytes: int,
    ) -> None:
        self._ep = _import_nccl_ep()
        self._max_recv_tokens_per_rank = max_recv_tokens_per_rank
        self._group = self._ep._NcclEpGroup.create(
            process_group,
            num_experts,
            max_dispatch_tokens_per_rank,
            max_recv_tokens_per_rank,
            max_token_bytes,
        )

    def create_routing(
        self,
        topk_idx: torch.Tensor,
        per_expert_token_counts: torch.Tensor | None = None,
    ) -> Routing:
        """Create expert routing for this phase; pass to :meth:`dispatch` / :meth:`combine`."""
        handle = self._ep._NcclEpHandle.create(
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
