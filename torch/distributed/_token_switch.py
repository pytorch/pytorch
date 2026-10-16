# mypy: allow-untyped-defs
from __future__ import annotations

import abc
import ctypes
import importlib.util
import os
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist


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


def _import_mori() -> tuple[Any, Any]:
    # MoRI EP v2 imports its flydsl kernel backend only when an op selects it, so
    # check for flydsl here to fail at construction rather than at first dispatch.
    try:
        import mori.cco as cco
        import mori.ops.dispatch_combine_v2 as ep
    except ImportError as e:
        raise ImportError(
            "TokenSwitchMoRI needs the 'mori' package (MoRI EP v2) from "
            "https://github.com/ROCm/mori."
        ) from e
    if importlib.util.find_spec("flydsl") is None:
        raise ImportError(
            "TokenSwitchMoRI needs MoRI's flydsl kernel backend; install it with "
            "`pip install amd_mori[flydsl]`."
        )
    return cco, ep


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
        max_recv = ts.max_recv_tokens_per_rank
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


@dataclass
class _MoRIRoutingState:
    # Routing is frozen; the MoRI handle (and, for expert_major, the slot map) is
    # filled in by the first dispatch.
    topk_idx: torch.Tensor
    mori_handle: Any = None
    # expert_major: (recv row * top_k + k) feeding each slot, and each slot's recv
    # row (recv_cap for unused slots, which combine drops).
    slot_src: torch.Tensor | None = None
    slot_dest_row: torch.Tensor | None = None


class TokenSwitchMoRI(TokenSwitch):
    """Intranode token switch backed by MoRI EP v2 with the flydsl kernel backend.

    The first :meth:`dispatch` on a Routing computes its receive-slot layout; later
    dispatches on it replay that layout. ``hidden_dim``, ``top_k`` and ``dtype`` are
    fixed at construction.

    The expert_major layout is built on MoRI's flat dispatch: slots are ordered by
    local expert, then by receive row, and can number up to
    :attr:`max_expert_major_slots_per_rank` (more than ``max_recv_tokens_per_rank``
    when a token hits several experts on one rank).
    """

    def __init__(
        self,
        process_group: ProcessGroup,
        num_experts: int,
        max_dispatch_tokens_per_rank: int,
        hidden_dim: int,
        top_k: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        cco, ep = _import_mori()
        self._ep = ep
        self._pg = process_group
        self._rank = dist.get_rank(process_group)
        world_size = dist.get_world_size(process_group)
        if num_experts % world_size:
            raise ValueError(
                f"num_experts={num_experts} is not divisible by world_size={world_size}"
            )
        self._num_experts = num_experts
        self._experts_per_rank = num_experts // world_size
        self._top_k = top_k

        uid = [cco.Communicator.get_unique_id() if self._rank == 0 else None]
        src = dist.get_global_rank(process_group, 0)
        dist.broadcast_object_list(uid, src=src, group=process_group)
        m = max_dispatch_tokens_per_rank
        window = world_size * m * hidden_dim * dtype.itemsize * 2 + (1 << 24)
        self._comm = cco.Communicator.init(
            world_size, self._rank, uid[0], 2 * window + (1 << 28)
        )
        cfg = ep.EpDispatchCombineConfig(
            rank=self._rank,
            world_size=world_size,
            hidden_dim=hidden_dim,
            max_num_inp_token_per_rank=m,
            num_experts_per_rank=self._experts_per_rank,
            num_experts_per_token=top_k,
            data_type=dtype,
            kernel_backend="flydsl",
        )
        self._op = ep.EpDispatchCombineOp(cfg, self._comm)
        self._comm.barrier()
        self._max_recv_tokens_per_rank = cfg.max_recv
        self._recv_cap = cfg.effective_max_recv
        self._em_slots = self._recv_cap * min(top_k, self._experts_per_rank)
        self._last_op: str | None = None
        self._fence = torch.zeros(1, device=torch.cuda.current_device())

    @property
    def max_recv_tokens_per_rank(self) -> int:
        return self._max_recv_tokens_per_rank

    @property
    def max_expert_major_slots_per_rank(self) -> int:
        """Upper bound on expert_major receive slots per rank."""
        return self._em_slots

    def close(self) -> None:
        self._op.close()
        self._comm.destroy()

    def _fence_if_repeat(self, op: str) -> None:
        # MoRI's kernels synchronize ranks on entry only, and a dispatch writes into
        # peers' receive buffers before that. Alternating dispatch and combine is
        # ordered by the other op's entry barrier; two of the same op in a row are
        # not, so a fast peer could overwrite our buffers before we copy the previous
        # results out. The all_reduce waits for every rank's earlier stream work.
        if self._last_op == op:
            dist.all_reduce(self._fence, group=self._pg)
        self._last_op = op

    def create_routing(
        self,
        topk_idx: torch.Tensor,
        per_expert_token_counts: torch.Tensor | None = None,
        *,
        layout: str,
    ) -> Routing:
        """Create expert routing for this phase; pass to :meth:`dispatch` / :meth:`combine`.

        ``layout`` is ``"flat"`` or ``"expert_major"``. Filling
        ``per_expert_token_counts`` costs an all_reduce on the process group, since
        MoRI computes no counts before dispatch.
        """
        if layout not in ("flat", "expert_major"):
            raise ValueError(f"layout must be 'flat' or 'expert_major'; got {layout!r}")
        if topk_idx.shape[1] != self._top_k:
            raise ValueError(
                f"top_k is fixed at {self._top_k}, got {topk_idx.shape[1]}"
            )
        if per_expert_token_counts is not None:
            ids = topk_idx.flatten().long()
            counts = torch.bincount(ids, minlength=self._num_experts)
            dist.all_reduce(counts, group=self._pg)
            epr = self._experts_per_rank
            lo = self._rank * epr
            per_expert_token_counts[:epr].copy_(counts[lo : lo + epr])
        state = _MoRIRoutingState(topk_idx.to(torch.int32).contiguous())
        return Routing(handle=state, topk_idx=topk_idx, layout=layout)

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
            rows = max(max_recv_tokens, self._em_slots)
            return tokens.new_zeros(rows, H), topk_weights.new_zeros(rows), None
        return (
            tokens.new_zeros(max_recv_tokens, H),
            topk_weights.new_zeros(max_recv_tokens, K),
            tokens.new_zeros(max_recv_tokens, K, dtype=torch.int64),
        )

    def _set_expert_major_slots(
        self, state: _MoRIRoutingState, out_idx: torch.Tensor, total_recv: torch.Tensor
    ) -> None:
        # Order every (recv row, k) pair that targets a local expert by (local
        # expert, recv row); the first pairs in that order fill the slots.
        epr, cap = self._experts_per_rank, self._recv_cap
        local = out_idx.long() - self._rank * epr
        row = torch.arange(cap, device=out_idx.device)[:, None]
        valid = (local >= 0) & (local < epr) & (row < total_recv.long())
        key = torch.where(valid, local * cap + row, epr * cap).flatten()
        src = key.argsort(stable=True)[: self._em_slots]
        used = torch.arange(self._em_slots, device=src.device) < valid.sum()
        state.slot_src = torch.where(used, src, 0)
        state.slot_dest_row = torch.where(used, src // self._top_k, cap)

    def _dispatch(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        out_tokens: torch.Tensor,
        out_topk_weights: torch.Tensor | None,
        out_topk_idx: torch.Tensor | None,
    ) -> None:
        flat = routing.layout == "flat"
        if out_topk_weights is None or (flat and out_topk_idx is None):
            raise ValueError(
                "TokenSwitchMoRI needs out_topk_weights (and flat: out_topk_idx)"
            )
        self._fence_if_repeat("dispatch")
        state = routing.handle
        tokens = tokens.contiguous()
        weights = topk_weights.to(torch.float32).contiguous()
        if state.mori_handle is None:
            out, out_w, _, out_idx, _, h = self._op.dispatch(
                tokens, weights, None, state.topk_idx, return_routing=True
            )
            # The handle's dest map and recv count alias op-owned buffers that the
            # next routing dispatch overwrites (ROCm/mori#715), so keep a copy.
            # TODO: once a MoRI release includes ROCm/mori#733, pass
            # snapshot=True to the dispatch above and drop the copy.
            state.mori_handle = self._ep.EpDispatchRoutingHandle(
                h.disp_dest_tok_id_map.clone(),
                h.inter_node_disp_dest_tok_id_map,
                h.inter_node_disp_send_map,
                h.total_recv_token_num.clone(),
                cur_rank_num_token=h.cur_rank_num_token,
            )
            if not flat:
                total = state.mori_handle.total_recv_token_num
                self._set_expert_major_slots(state, out_idx, total)
        else:
            out, out_w, _, out_idx, _ = self._op.dispatch(
                tokens, weights, None, state.topk_idx, routing=state.mori_handle
            )
        # MoRI's outputs are views of op-owned buffers that the next dispatch
        # overwrites, so copy them out. Rows past the receive count (or slot count)
        # are garbage, which the contract allows; trimming them needs a host sync.
        if not flat:
            n = min(out_tokens.shape[0], self._em_slots)
            src = state.slot_src[:n]
            out_tokens[:n].copy_(out[src // self._top_k])
            out_topk_weights[:n].copy_(out_w.reshape(-1)[src])
            return
        # MoRI forwards each token's global top-k; keep this rank's experts as local
        # ids in place and mark the others -1.
        n = min(out_tokens.shape[0], out.shape[0])
        out_tokens[:n].copy_(out[:n])
        epr = self._experts_per_rank
        local = out_idx[:n].long() - self._rank * epr
        other = (local < 0) | (local >= epr)
        out_topk_idx[:n].copy_(local.masked_fill_(other, -1))
        out_topk_weights[:n].copy_(out_w[:n]).masked_fill_(other, 0.0)

    def _combine(
        self,
        routing: Routing,
        expert_tokens: torch.Tensor,
        out_tokens: torch.Tensor,
    ) -> None:
        mori_handle = routing.handle.mori_handle
        if mori_handle is None:
            raise RuntimeError(
                "TokenSwitchMoRI: combine() before the first dispatch() on this Routing"
            )
        self._fence_if_repeat("combine")
        if routing.layout == "flat":
            out, _ = self._op.combine(expert_tokens.contiguous(), routing=mori_handle)
        else:
            # Sum each recv row's slots into MoRI's combine input buffer; MoRI skips
            # its own staging copy when handed that buffer.
            staged, cap = self._op.combine_in_view(), self._recv_cap
            m = min(expert_tokens.shape[0], self._em_slots)
            dest = routing.handle.slot_dest_row[:m]
            acc = staged.new_zeros(cap + 1, staged.shape[1], dtype=torch.float32)
            acc.index_add_(0, dest, expert_tokens[:m].float())
            staged.copy_(acc[:cap])
            out, _ = self._op.combine(staged, routing=mori_handle)
        out_tokens.copy_(out)
