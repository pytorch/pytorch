# mypy: allow-untyped-defs
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed._token_switch import Routing, TokenSwitch


if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup


def _import_mori() -> tuple[Any, Any]:
    # MoRI EP v2 imports its flydsl kernel backend only when an op selects it, so
    # check for flydsl here to fail at construction rather than at first dispatch.
    try:
        # pyrefly: ignore [missing-import]  # optional, from ROCm/mori
        import mori.cco as cco

        # pyrefly: ignore [missing-import]  # optional, from ROCm/mori
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


@dataclass
class _MoRIRoutingState:
    # Routing is frozen; the MoRI handle (and, for expert_major, the slot map) is
    # filled in by the first dispatch.
    topk_idx: torch.Tensor
    mori_handle: Any = None
    # expert_major: (recv row * top_k + k) feeding each slot, and each slot's recv
    # row (recv_cap for unused slots, which combine drops).
    slot_src: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    slot_dest_row: torch.Tensor = field(default_factory=lambda: torch.empty(0))


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
        self._world_size = world_size
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
        self._closed = False

    @property
    def max_recv_tokens_per_rank(self) -> int:
        return self._max_recv_tokens_per_rank

    @property
    def max_expert_major_slots_per_rank(self) -> int:
        """Upper bound on expert_major receive slots per rank."""
        return self._em_slots

    def close(self) -> None:
        """Release the MoRI op and communicator; the switch is unusable afterwards."""
        if self._closed:
            return
        self._closed = True
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
        if topk_idx.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                f"topk_idx must have dtype int32 or int64, got {topk_idx.dtype}"
            )
        if topk_idx.dim() != 2 or topk_idx.shape[1] != self._top_k:
            raise ValueError(
                f"topk_idx must have shape (num_tokens, {self._top_k}), "
                f"got {tuple(topk_idx.shape)}"
            )
        if per_expert_token_counts is not None:
            c = per_expert_token_counts
            if (
                c.dim() != 1
                or c.dtype != torch.int32
                or c.device != topk_idx.device
                or c.numel() != self._experts_per_rank
            ):
                raise ValueError(
                    f"per_expert_token_counts must be a 1D int32 tensor on {topk_idx.device} "
                    f"with {self._experts_per_rank} elements; got {c.dtype} on {c.device} "
                    f"with shape {tuple(c.shape)}"
                )
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

    @staticmethod
    def _state(routing: Routing) -> _MoRIRoutingState:
        state = routing.handle
        if not isinstance(state, _MoRIRoutingState):
            raise TypeError(
                f"TokenSwitchMoRI got a Routing from another backend ({type(state)})"
            )
        return state

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
        need = self._recv_cap if flat else self._em_slots
        for t in (out_tokens, out_topk_weights, out_topk_idx):
            if t is not None and t.shape[0] < need:
                raise ValueError(
                    f"TokenSwitchMoRI {routing.layout} dispatch outputs require at "
                    f"least {need} rows, got {t.shape[0]}"
                )
        self._fence_if_repeat("dispatch")
        state = self._state(routing)
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
            n, src = self._em_slots, state.slot_src
            out_tokens[:n].copy_(out[src // self._top_k])
            out_topk_weights[:n].copy_(out_w.reshape(-1)[src])
            return
        # MoRI forwards each token's global top-k; keep this rank's experts as local
        # ids in place and mark the others -1.
        n = out.shape[0]
        out_tokens[:n].copy_(out[:n])
        epr = self._experts_per_rank
        local = out_idx[:n].long() - self._rank * epr
        other = (local < 0) | (local >= epr)
        # pyrefly: ignore [unsupported-operation]  # flat: checked on entry
        out_topk_idx[:n].copy_(local.masked_fill_(other, -1))
        out_topk_weights[:n].copy_(out_w[:n]).masked_fill_(other, 0.0)

    def _combine(
        self,
        routing: Routing,
        expert_tokens: torch.Tensor,
        out_tokens: torch.Tensor,
    ) -> None:
        state = self._state(routing)
        mori_handle = state.mori_handle
        if mori_handle is None:
            raise RuntimeError(
                "TokenSwitchMoRI: combine() before the first dispatch() on this Routing"
            )
        self._fence_if_repeat("combine")
        if routing.layout == "flat":
            rows = expert_tokens[: self._recv_cap].contiguous()
            out, _ = self._op.combine(rows, routing=mori_handle)
        else:
            # Sum each recv row's slots into MoRI's combine input buffer; MoRI skips
            # its own staging copy when handed that buffer.
            staged, cap = self._op.combine_in_view(), self._recv_cap
            m = min(expert_tokens.shape[0], self._em_slots)
            dest = state.slot_dest_row[:m]
            acc = staged.new_zeros(cap + 1, staged.shape[1], dtype=torch.float32)
            acc.index_add_(0, dest, expert_tokens[:m].float())
            staged.copy_(acc[:cap])
            out, _ = self._op.combine(staged, routing=mori_handle)
        out_tokens.copy_(out)

    def _combine_backward(
        self,
        routing: Routing,
        grad_tokens: torch.Tensor,
        out_expert_grads: torch.Tensor,
    ) -> None:
        flat, K = routing.layout == "flat", self._top_k
        need = self._recv_cap if flat else self._em_slots
        # A combine input with fewer rows than the layout's bound gets a gradient
        # buffer that short; dispatch into a full one and copy its leading rows out.
        out = out_expert_grads
        if out.shape[0] < need:
            out = out.new_zeros(need, out.shape[1])
        rows = out.shape[0]
        weights = grad_tokens.new_zeros(grad_tokens.shape[0], K, dtype=torch.float32)
        if flat:
            out_w = weights.new_zeros(rows, K)
            out_idx = weights.new_zeros(rows, K, dtype=torch.int64)
        else:
            out_w, out_idx = weights.new_zeros(rows), None
        self._dispatch(routing, grad_tokens, weights, out, out_w, out_idx)
        # The forward combine ignores rows past the receive count (flat) and unused
        # slots (expert_major), so their gradient is zero, not what dispatch copied.
        state = self._state(routing)
        if flat:
            total = state.mori_handle.total_recv_token_num.long()
            dead = torch.arange(need, device=out.device) >= total
        else:
            dead = state.slot_dest_row >= self._recv_cap
        out[:need].masked_fill_(dead[:, None], 0)
        if out is not out_expert_grads:
            out_expert_grads.copy_(out[: out_expert_grads.shape[0]])

    def _dispatch_backward(
        self,
        routing: Routing,
        grad_recv_tokens: torch.Tensor,
        grad_recv_topk_weights: torch.Tensor,
        out_token_grads: torch.Tensor,
        out_topk_weight_grads: torch.Tensor,
    ) -> None:
        self._combine(routing, grad_recv_tokens, out_token_grads)
        # Received weights are copies, so source weight (t, k) gets the gradient at
        # its receive slot on expert k's rank. Gather every rank's slot gradients in
        # fp32 (a MoRI combine would round them to the token dtype) and index them.
        state = self._state(routing)
        K, cap = self._top_k, self._recv_cap
        g = grad_recv_topk_weights
        grads = g.new_zeros(cap * K, dtype=torch.float32)
        if routing.layout == "flat":
            n = min(cap, g.shape[0])
            grads[: n * K].copy_(g[:n].reshape(-1))
        else:
            n = min(g.shape[0], self._em_slots)
            used = state.slot_dest_row[:n] < cap
            grads.index_add_(0, state.slot_src[:n], torch.where(used, g[:n], 0).float())
        all_grads = grads.new_empty(self._world_size * cap * K)
        dist.all_gather_into_tensor(all_grads, grads, group=self._pg)
        # MoRI's dest map holds dest_rank * cap + row per (t, k). A token goes to a
        # rank once, so its other k on that rank hold a dropped marker instead; take
        # the slot from the k that was sent.
        m = out_topk_weight_grads.shape[0]
        enc = state.mori_handle.disp_dest_tok_id_map[: m * K].view(m, K).long()
        epr = self._experts_per_rank
        dest = state.topk_idx[:m].long().div(epr, rounding_mode="floor")
        sent = enc < self._world_size * cap
        same = (dest[:, :, None] == dest[:, None, :]) & sent[:, None, :]
        slot = torch.where(same, enc[:, None, :], -1).amax(dim=2)
        k = torch.arange(K, device=slot.device)
        picked = all_grads[slot.clamp(min=0) * K + k]
        out_topk_weight_grads.copy_(torch.where(slot >= 0, picked, 0))
