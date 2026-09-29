# Owner(s): ["oncall: distributed"]


import logging
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed._token_switch import (
    _import_nccl_ep,
    Routing,
    TokenSwitch,
    TokenSwitchNCCL,
)
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
)


log = logging.getLogger(__name__)


def _nccl_ep_available() -> bool:
    # The torch._nccl_ep extension is built only with USE_NCCL_EP, and a
    # USE_SYSTEM_NCCL=ON build additionally needs the nccl4py wheel at runtime.
    # Actually importing it is the real check: find_spec would only locate the
    # extension without dlopening it, so it would miss a missing libnccl_ep.so.
    if not torch.cuda.is_available():
        return False
    try:
        _import_nccl_ep()
    except Exception:
        log.debug("torch._nccl_ep unavailable; skipping EP tests", exc_info=True)
        return False
    return True


def requires_nccl_ep():
    return skip_but_pass_in_sandcastle_if(
        not _nccl_ep_available(),
        "Test requires a USE_NCCL_EP build (plus nccl4py for USE_SYSTEM_NCCL=ON)",
    )


NUM_TOKENS = 16
TOP_K = 1
HIDDEN = 64
TOKEN_SIZE_BYTES = HIDDEN * 2
NUM_MULTI_ROUND_DISPATCH_COMBINE = 3


def _generate_topk(rank, world_size, num_tokens, top_k, device):
    remote_expert = (rank + 1) % world_size
    topk_idx = torch.full(
        (num_tokens, top_k), remote_expert, dtype=torch.int64, device=device
    )
    topk_weights = torch.full(
        (num_tokens, top_k), 1.0 / top_k, dtype=torch.float32, device=device
    )
    return topk_idx, topk_weights


# Contract tests: top-k > 1 with several experts per rank, so a token can hit a
# destination rank more than once and several ranks at once.
CONTRACT_TOP_K = 2
CONTRACT_EXPERTS_PER_RANK = 2


def _to_local_packed(idx, weights, first_expert, experts_per_rank):
    """Flat-layout convention: local expert ids ascending, packed, -1 / 0 padding."""
    local = idx - first_expert
    in_range = (local >= 0) & (local < experts_per_rank)
    key = torch.where(in_range, local, experts_per_rank)
    key, perm = key.sort(dim=1)
    routed = key < experts_per_rank
    return torch.where(routed, key, -1), torch.where(routed, weights.gather(1, perm), 0)


@dataclass
class _ReferenceHandle:
    topk_idx_all: torch.Tensor  # [world_size, N, K], every rank's routing
    recv_src: torch.Tensor  # [num_recv, 2] (src rank, src token) per receive slot
    send_token: torch.Tensor  # [num_send] own token per (token, dest rank) pair
    send_flat_slot: torch.Tensor  # [num_send] dest_rank * max_recv + slot on dest


class _ReferenceTokenSwitch(TokenSwitch):
    """Test-only TokenSwitch built on all_gather, flat layout only.

    It implements the contract on any process group (gloo on CPU here), so the
    contract tests run without an EP library. Every rank must pass the same
    number of tokens. Receive slots on a rank are ordered by (src rank, src token).
    """

    def __init__(self, pg, num_experts, max_dispatch_tokens_per_rank):
        self._pg = pg
        self._rank = dist.get_rank(pg)
        self._world_size = dist.get_world_size(pg)
        self._experts_per_rank = num_experts // self._world_size
        self._max_recv = self._world_size * max_dispatch_tokens_per_rank

    @property
    def max_recv_tokens_per_rank(self):
        return self._max_recv

    def _all_gather(self, t):
        parts = [torch.empty_like(t) for _ in range(self._world_size)]
        dist.all_gather(parts, t.contiguous(), group=self._pg)
        return torch.stack(parts)

    def create_routing(self, topk_idx, per_expert_token_counts=None, *, layout):
        if layout != "flat":
            raise ValueError(f"reference TokenSwitch is flat-only; got {layout!r}")
        idx_all = self._all_gather(topk_idx)
        epr = self._experts_per_rank
        dest = [[set((row // epr).tolist()) for row in r] for r in idx_all]
        recv_src, send_token, send_slot = [], [], []
        for d in range(self._world_size):
            srcs = [
                (r, t)
                for r in range(self._world_size)
                for t in range(len(dest[r]))
                if d in dest[r][t]
            ]
            if d == self._rank:
                recv_src = srcs
            for slot, (r, t) in enumerate(srcs):
                if r == self._rank:
                    send_token.append(t)
                    send_slot.append(d * self._max_recv + slot)
        if per_expert_token_counts is not None:
            local = idx_all.flatten() - self._rank * epr
            counts = torch.bincount(local[(local >= 0) & (local < epr)], minlength=epr)
            per_expert_token_counts[:epr].copy_(counts)
        handle = _ReferenceHandle(
            idx_all,
            torch.tensor(recv_src, dtype=torch.int64).reshape(-1, 2),
            torch.tensor(send_token, dtype=torch.int64),
            torch.tensor(send_slot, dtype=torch.int64),
        )
        return Routing(handle=handle, topk_idx=topk_idx, layout=layout)

    def _alloc_dispatch_outputs(
        self, routing, tokens, topk_weights, max_recv_tokens, H, K
    ):
        return (
            tokens.new_zeros(max_recv_tokens, H),
            topk_weights.new_zeros(max_recv_tokens, K),
            tokens.new_zeros(max_recv_tokens, K, dtype=torch.int64),
        )

    def _dispatch(
        self, routing, tokens, topk_weights, out_tokens, out_topk_weights, out_topk_idx
    ):
        h = routing.handle
        r, t = h.recv_src[:, 0], h.recv_src[:, 1]
        n = len(r)
        out_tokens[:n].copy_(self._all_gather(tokens)[r, t])
        epr = self._experts_per_rank
        w_all = self._all_gather(topk_weights)
        first = self._rank * epr
        idx, w = _to_local_packed(h.topk_idx_all[r, t], w_all[r, t], first, epr)
        out_topk_idx[:n].copy_(idx)
        out_topk_weights[:n].copy_(w)

    def _combine(self, routing, expert_tokens, out_tokens):
        h = routing.handle
        padded = expert_tokens.new_zeros(self._max_recv, expert_tokens.shape[-1])
        padded[: expert_tokens.shape[0]].copy_(expert_tokens)
        rows = self._all_gather(padded).flatten(0, 1)[h.send_flat_slot]
        out_tokens.zero_().index_add_(0, h.send_token, rows)


class _TokenSwitchContractTests:
    """Contract tests (see TokenSwitch's docstring) shared by every backend.

    The host class provides ``device``, ``_init()``, ``contract_dtype`` and
    ``get_contract_token_switch()`` (world_size * CONTRACT_EXPERTS_PER_RANK experts,
    top-k CONTRACT_TOP_K, flat layout).
    """

    def _contract_inputs(self, seed):
        # Every rank builds every rank's inputs, so it knows exactly what it receives.
        K, epr = CONTRACT_TOP_K, CONTRACT_EXPERTS_PER_RANK
        E = self.world_size * epr
        idx, xs = [], []
        for r in range(self.world_size):
            g = torch.Generator().manual_seed(seed * 100 + r)
            rows = [torch.randperm(E, generator=g)[:K] for _ in range(NUM_TOKENS)]
            idx.append(torch.stack(rows))
            x = torch.randn(NUM_TOKENS, HIDDEN, generator=g)
            x[:, 0] = r  # col0/col1 encode (src rank, src token)
            x[:, 1] = torch.arange(NUM_TOKENS)
            xs.append(x.to(self.contract_dtype))
        dest = [[{int(e) // epr for e in row} for row in i.tolist()] for i in idx]
        ndest = torch.tensor([float(len(d)) for d in dest[self.rank]]).unsqueeze(1)
        expected = sorted(
            (r, t)
            for r in range(self.world_size)
            for t in range(NUM_TOKENS)
            if self.rank in dest[r][t]
        )
        # distinct weight per top-k slot, so the receiver's weight alignment is checkable
        weights = (0.1 * torch.arange(1, K + 1)).expand(NUM_TOKENS, K).contiguous()
        return (
            idx[self.rank].to(self.device),
            weights.to(self.device),
            xs[self.rank].to(self.device),
            ndest.to(self.device),
            expected,
            idx,
        )

    def _contract_out_buffers(self, ts):
        max_recv, K, dev = ts.max_recv_tokens_per_rank, CONTRACT_TOP_K, self.device
        return (
            torch.zeros(max_recv, HIDDEN, dtype=self.contract_dtype, device=dev),
            torch.zeros(max_recv, K, dtype=torch.float32, device=dev),
            torch.zeros(max_recv, K, dtype=torch.int64, device=dev),
        )

    def _sources(self, rows):
        return [(int(a), int(b)) for a, b in rows[:, :2].float().tolist()]

    def test_contract_per_expert_counts(self):
        self._init()
        ts = self.get_contract_token_switch()
        topk_idx, _w, _x, _nd, _expected, all_idx = self._contract_inputs(seed=1)
        epr = CONTRACT_EXPERTS_PER_RANK
        counts = torch.zeros(epr, dtype=torch.int32, device=self.device)
        ts.create_routing(topk_idx, counts, layout="flat")
        # counts are received (token, local expert) pairs, not received rows
        all_ids = torch.cat(all_idx).flatten()
        per_expert = torch.bincount(all_ids, minlength=self.world_size * epr)
        lo = self.rank * epr
        self.assertEqual(counts.cpu().long(), per_expert[lo : lo + epr])

    def test_contract_flat_dispatch_rows_and_topk(self):
        self._init()
        ts = self.get_contract_token_switch()
        topk_idx, w, x, _nd, expected, all_idx = self._contract_inputs(seed=2)
        routing = ts.create_routing(topk_idx, layout="flat")
        out_tokens, out_w, out_idx = self._contract_out_buffers(ts)
        ts.dispatch(routing, x, w, out=(out_tokens, out_w, out_idx))
        n = len(expected)
        srcs = self._sources(out_tokens[:n].cpu())
        # one row per (source token, destination rank), carrying that token's payload
        self.assertEqual(sorted(srcs), expected)
        lo = self.rank * CONTRACT_EXPERTS_PER_RANK
        want_idx = torch.full((n, CONTRACT_TOP_K), -1, dtype=torch.int64)
        want_w = torch.zeros(n, CONTRACT_TOP_K)
        for row, (r, t) in enumerate(srcs):
            mine = sorted(
                (int(e) - lo, k)
                for k, e in enumerate(all_idx[r][t])
                if 0 <= int(e) - lo < CONTRACT_EXPERTS_PER_RANK
            )
            for j, (e, k) in enumerate(mine):
                want_idx[row, j] = e
                want_w[row, j] = 0.1 * (k + 1)
        self.assertEqual(out_idx[:n].cpu(), want_idx)
        self.assertEqual(out_w[:n].cpu(), want_w)

    def test_contract_flat_combine_sums_per_destination_rank(self):
        self._init()
        ts = self.get_contract_token_switch()
        topk_idx, w, x, ndest, expected, _ = self._contract_inputs(seed=3)
        routing = ts.create_routing(topk_idx, layout="flat")
        out_tokens, out_w, out_idx = self._contract_out_buffers(ts)
        ts.dispatch(routing, x, w, out=(out_tokens, out_w, out_idx))
        combined = torch.zeros_like(x)
        ts.combine(routing, out_tokens[: len(expected)].contiguous(), out=combined)
        self.assertEqual(combined.float(), ndest * x.float())

    def test_contract_redispatch_keeps_layout(self):
        # A second dispatch on the same Routing (what combine's backward does) must put
        # the new payload into the first dispatch's slots.
        self._init()
        ts = self.get_contract_token_switch()
        topk_idx, w, x, _nd, expected, _ = self._contract_inputs(seed=4)
        routing = ts.create_routing(topk_idx, layout="flat")
        n = len(expected)
        out_tokens, out_w, out_idx = self._contract_out_buffers(ts)
        ts.dispatch(routing, x, w, out=(out_tokens, out_w, out_idx))
        first = out_tokens[:n].clone()
        x2 = x.clone()
        x2[:, 2:] = -x2[:, 2:]
        ts.dispatch(routing, x2, w, out=(out_tokens, out_w, out_idx))
        self.assertEqual(out_tokens[:n, :2], first[:, :2])
        self.assertEqual(out_tokens[:n, 2:], -first[:, 2:])

    def test_contract_autograd_roundtrip(self):
        self._init()
        ts = self.get_contract_token_switch()
        topk_idx, w, x, ndest, expected, _ = self._contract_inputs(seed=5)
        routing = ts.create_routing(topk_idx, layout="flat")
        tokens = x.clone().requires_grad_(True)
        dispatched, _, _ = ts.dispatch(routing, tokens, w, ts.max_recv_tokens_per_rank)
        combined = ts.combine(routing, dispatched[: len(expected)].contiguous())
        combined.sum().backward()
        self.assertEqual(combined.float(), ndest * x.float())
        self.assertEqual(tokens.grad.float(), ndest.expand(NUM_TOKENS, HIDDEN))

    def test_contract_combine_backward_uses_forward_slots(self):
        self._init()
        ts = self.get_contract_token_switch()
        topk_idx, w, x, _nd, expected, _ = self._contract_inputs(seed=6)
        routing = ts.create_routing(topk_idx, layout="flat")
        out_tokens, out_w, out_idx = self._contract_out_buffers(ts)
        ts.dispatch(routing, x, w, out=(out_tokens, out_w, out_idx))
        n = len(expected)
        expert_tokens = out_tokens[:n].contiguous().requires_grad_(True)
        # scale each combined row by its token id so every gradient row is distinct
        scale = torch.arange(1.0, NUM_TOKENS + 1, device=self.device).unsqueeze(1)
        (ts.combine(routing, expert_tokens).float() * scale).sum().backward()
        want = (out_tokens[:n, 1].float() + 1)[:, None].expand(n, HIDDEN)
        self.assertEqual(expert_tokens.grad.float(), want)

    def test_contract_two_routings_autograd(self):
        # Two live Routings: layer 1's backward runs after layer 2's forward dispatch.
        self._init()
        ts = self.get_contract_token_switch()
        idx1, w1, x, nd1, exp1, _ = self._contract_inputs(seed=7)
        idx2, w2, _x2, nd2, exp2, _ = self._contract_inputs(seed=8)
        r1 = ts.create_routing(idx1, layout="flat")
        r2 = ts.create_routing(idx2, layout="flat")
        max_recv = ts.max_recv_tokens_per_rank
        tokens = x.clone().requires_grad_(True)
        d1, _, _ = ts.dispatch(r1, tokens, w1, max_recv)
        h = ts.combine(r1, d1[: len(exp1)].contiguous())
        d2, _, _ = ts.dispatch(r2, h, w2, max_recv)
        y = ts.combine(r2, d2[: len(exp2)].contiguous())
        y.sum().backward()
        self.assertEqual(y.float(), nd1 * nd2 * x.float())
        self.assertEqual(tokens.grad.float(), (nd1 * nd2).expand(NUM_TOKENS, HIDDEN))

    def test_contract_interleaved_routings(self):
        # Two dispatches before either combine (e.g. running one layer or micro-batch
        # ahead); their backwards then run two combines and two dispatches back to back.
        self._init()
        ts = self.get_contract_token_switch()
        max_recv = ts.max_recv_tokens_per_rank
        for seed in range(9, 15, 2):
            idx1, w1, x1, nd1, exp1, _ = self._contract_inputs(seed=seed)
            idx2, w2, x2, nd2, exp2, _ = self._contract_inputs(seed=seed + 1)
            r1 = ts.create_routing(idx1, layout="flat")
            r2 = ts.create_routing(idx2, layout="flat")
            t1 = x1.clone().requires_grad_(True)
            t2 = x2.clone().requires_grad_(True)
            d1, _, _ = ts.dispatch(r1, t1, w1, max_recv)
            d2, _, _ = ts.dispatch(r2, t2, w2, max_recv)
            y1 = ts.combine(r1, d1[: len(exp1)].contiguous())
            y2 = ts.combine(r2, d2[: len(exp2)].contiguous())
            (y1.sum() + 2 * y2.sum()).backward()
            self.assertEqual(y1.float(), nd1 * x1.float())
            self.assertEqual(y2.float(), nd2 * x2.float())
            self.assertEqual(t1.grad.float(), nd1.expand(NUM_TOKENS, HIDDEN))
            self.assertEqual(t2.grad.float(), (2 * nd2).expand(NUM_TOKENS, HIDDEN))


class TokenSwitchReferenceTest(_TokenSwitchContractTests, MultiProcContinuousTest):
    """Runs the contract tests on the gloo reference backend (no GPU, no EP library)."""

    world_size = 2
    contract_dtype = torch.float32
    _cached_contract_token_switch: _ReferenceTokenSwitch | None = None

    @classmethod
    def backend_str(cls):
        return "gloo"

    @classmethod
    def device_type(cls):
        return "cpu"

    @property
    def device(self):
        return torch.device("cpu")

    @classmethod
    def get_contract_token_switch(cls):
        if cls._cached_contract_token_switch is None:
            pg = dist.distributed_c10d._get_default_group()
            cls._cached_contract_token_switch = _ReferenceTokenSwitch(
                pg, dist.get_world_size(pg) * CONTRACT_EXPERTS_PER_RANK, NUM_TOKENS
            )
        return cls._cached_contract_token_switch

    def _init(self):
        dist.barrier()


@requires_nccl_ep()
class TokenSwitchNCCLTest(_TokenSwitchContractTests, MultiProcContinuousTest):
    _cached_token_switch: TokenSwitchNCCL | None = None
    _cached_contract_token_switch: TokenSwitchNCCL | None = None
    contract_dtype = torch.bfloat16

    @classmethod
    def backend_str(cls):
        return "nccl"

    @property
    def device(self):
        return torch.device("cuda", self.rank)

    @classmethod
    def get_token_switch(cls) -> TokenSwitchNCCL:
        if cls._cached_token_switch is None:
            pg = dist.distributed_c10d._get_default_group()
            rank = dist.get_rank(pg)
            world_size = dist.get_world_size(pg)
            print(f"rank {rank} creating token switch")
            dist.barrier(pg)
            cls._cached_token_switch = TokenSwitchNCCL(
                pg, world_size, NUM_TOKENS, world_size * NUM_TOKENS, TOKEN_SIZE_BYTES
            )
        return cls._cached_token_switch

    @classmethod
    def get_contract_token_switch(cls) -> TokenSwitchNCCL:
        if cls._cached_contract_token_switch is None:
            pg = dist.distributed_c10d._get_default_group()
            world_size = dist.get_world_size(pg)
            dist.barrier(pg)
            cls._cached_contract_token_switch = TokenSwitchNCCL(
                pg,
                world_size * CONTRACT_EXPERTS_PER_RANK,
                NUM_TOKENS,
                world_size * NUM_TOKENS,
                TOKEN_SIZE_BYTES,
            )
        return cls._cached_contract_token_switch

    def _init(self):
        torch.cuda.set_device(self.device)
        dist.barrier()

    @skip_if_lt_x_gpu(2)
    def test_create_routing(self):
        self._init()
        ts = self.get_token_switch()
        num_experts = self.world_size
        num_local_experts = num_experts // self.world_size
        topk_idx, _topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        per_expert_counts = torch.zeros(
            num_local_experts, dtype=torch.int32, device=self.device
        )
        ts.create_routing(topk_idx, per_expert_counts, layout="flat")
        self.assertEqual(per_expert_counts.dtype, torch.int32)
        self.assertEqual(per_expert_counts.numel(), num_local_experts)
        self.assertEqual(per_expert_counts.item(), NUM_TOKENS)
        torch.cuda.synchronize()

    @skip_if_lt_x_gpu(2)
    def test_dispatch(self):
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        per_expert_counts = torch.zeros(
            self.world_size, dtype=torch.int32, device=self.device
        )
        routing = ts.create_routing(topk_idx, per_expert_counts, layout="flat")

        token_val = float(self.rank + 1)
        tokens = torch.full(
            (NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16, device=self.device
        )
        out_tokens = torch.zeros(
            (num_recv_tokens, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        out_topk_weights = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.float32, device=self.device
        )
        out_topk_idx = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.int64, device=self.device
        )

        ts.dispatch(
            routing,
            tokens,
            topk_weights,
            out=(out_tokens, out_topk_weights, out_topk_idx),
        )
        torch.cuda.synchronize()

        src_rank = (self.rank - 1) % self.world_size
        expected_val = float(src_rank + 1)
        received = out_tokens[:NUM_TOKENS].float()
        self.assertTrue(
            received.eq(expected_val).all(),
            lambda msg: (
                f"{msg}\nrank {self.rank}: expected {expected_val}, got {received[0, 0].item()}"
            ),
        )

    @skip_if_lt_x_gpu(2)
    def test_dispatch_combine_roundtrip(self):
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        per_expert_counts = torch.zeros(
            self.world_size, dtype=torch.int32, device=self.device
        )
        routing = ts.create_routing(topk_idx, per_expert_counts, layout="flat")

        token_val = float(self.rank + 1)
        tokens = torch.full(
            (NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16, device=self.device
        )
        out_tokens = torch.zeros(
            (num_recv_tokens, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        out_topk_weights = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.float32, device=self.device
        )
        out_topk_idx = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.int64, device=self.device
        )

        ts.dispatch(
            routing,
            tokens,
            topk_weights,
            out=(out_tokens, out_topk_weights, out_topk_idx),
        )
        torch.cuda.synchronize()

        expert_tokens = out_tokens[:NUM_TOKENS].contiguous()
        combined = torch.zeros(
            (NUM_TOKENS, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        ts.combine(routing, expert_tokens, out=combined)
        torch.cuda.synchronize()

        expected = torch.full((NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16)
        self.assertEqual(combined.cpu(), expected)

    @skip_if_lt_x_gpu(2)
    def test_dispatch_autograd_backward(self):
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        per_expert_counts = torch.zeros(
            self.world_size, dtype=torch.int32, device=self.device
        )
        routing = ts.create_routing(topk_idx, per_expert_counts, layout="flat")

        tokens = torch.full(
            (NUM_TOKENS, HIDDEN),
            float(self.rank + 1),
            dtype=torch.bfloat16,
            device=self.device,
            requires_grad=True,
        )
        out_tokens, _out_weights, _out_idx = ts.dispatch(
            routing, tokens, topk_weights, num_recv_tokens
        )
        out_tokens.sum().backward()
        torch.cuda.synchronize()
        self.assertEqual(_out_idx.dtype, torch.int64)

        # grad_out_tokens is all-ones; combine routes them back: each token gets 1.0 per top-k slot
        self.assertIsNotNone(tokens.grad)
        self.assertEqual(tokens.grad, torch.ones_like(tokens))

    @skip_if_lt_x_gpu(2)
    def test_combine_autograd_backward(self):
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        per_expert_counts = torch.zeros(
            self.world_size, dtype=torch.int32, device=self.device
        )
        routing = ts.create_routing(topk_idx, per_expert_counts, layout="flat")

        # Pre-dispatch so the routing handle is primed, then test combine autograd
        tokens = torch.full(
            (NUM_TOKENS, HIDDEN),
            float(self.rank + 1),
            dtype=torch.bfloat16,
            device=self.device,
        )
        out_tokens_buf = torch.zeros(
            num_recv_tokens, HIDDEN, dtype=torch.bfloat16, device=self.device
        )
        out_weights_buf = torch.zeros(
            num_recv_tokens, TOP_K, dtype=torch.float32, device=self.device
        )
        out_idx_buf = torch.zeros(
            num_recv_tokens, TOP_K, dtype=torch.int64, device=self.device
        )
        ts.dispatch(
            routing,
            tokens,
            topk_weights,
            out=(out_tokens_buf, out_weights_buf, out_idx_buf),
        )
        torch.cuda.synchronize()

        expert_tokens = (
            out_tokens_buf[:NUM_TOKENS].contiguous().detach().requires_grad_(True)
        )
        combined = ts.combine(routing, expert_tokens)
        combined.sum().backward()
        torch.cuda.synchronize()

        # grad_out_tokens is all-ones; dispatch routes them back to expert ranks
        self.assertIsNotNone(expert_tokens.grad)
        self.assertEqual(expert_tokens.grad, torch.ones_like(expert_tokens))

    @skip_if_lt_x_gpu(2)
    def test_dispatch_combine_autograd_roundtrip(self):
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        per_expert_counts = torch.zeros(
            self.world_size, dtype=torch.int32, device=self.device
        )
        routing = ts.create_routing(topk_idx, per_expert_counts, layout="flat")

        token_val = float(self.rank + 1)
        tokens = torch.full(
            (NUM_TOKENS, HIDDEN),
            token_val,
            dtype=torch.bfloat16,
            device=self.device,
            requires_grad=True,
        )
        dispatched, _weights, _idx = ts.dispatch(
            routing, tokens, topk_weights, num_recv_tokens
        )
        expert_out = dispatched[:NUM_TOKENS].contiguous()
        combined = ts.combine(routing, expert_out)
        combined.sum().backward()
        torch.cuda.synchronize()

        # With identity expert (no-op), gradient should round-trip back as all-ones
        self.assertIsNotNone(tokens.grad)
        self.assertEqual(tokens.grad, torch.ones_like(tokens))

    @skip_if_lt_x_gpu(2)
    def test_dispatch_combine_multiple_rounds(self):
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        per_expert_counts = torch.zeros(
            self.world_size, dtype=torch.int32, device=self.device
        )
        routing = ts.create_routing(topk_idx, per_expert_counts, layout="flat")

        out_tokens = torch.zeros(
            (num_recv_tokens, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        out_topk_weights = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.float32, device=self.device
        )
        out_topk_idx = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.int64, device=self.device
        )
        combined = torch.zeros(
            (NUM_TOKENS, HIDDEN), dtype=torch.bfloat16, device=self.device
        )

        for r in range(NUM_MULTI_ROUND_DISPATCH_COMBINE):
            token_val = float(self.rank + 1 + r)
            tokens = torch.full(
                (NUM_TOKENS, HIDDEN),
                token_val,
                dtype=torch.bfloat16,
                device=self.device,
            )
            ts.dispatch(
                routing,
                tokens,
                topk_weights,
                out=(out_tokens, out_topk_weights, out_topk_idx),
            )
            torch.cuda.synchronize()

            expert_tokens = out_tokens[:NUM_TOKENS].contiguous()
            ts.combine(routing, expert_tokens, out=combined)
            torch.cuda.synchronize()

        expected = torch.full(
            (NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16, device=self.device
        )
        self.assertEqual(combined, expected)

    @skip_if_lt_x_gpu(2)
    def test_dispatch_expert_major(self):
        """Expert-major layout: out_topk_weights is 1-D, out_topk_idx is absent."""
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        routing = ts.create_routing(topk_idx, layout="expert_major")

        token_val = float(self.rank + 1)
        tokens = torch.full(
            (NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16, device=self.device
        )
        out_tokens = torch.zeros(
            (num_recv_tokens, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        # 1-D weights: one scalar per received slot (no topk dim).
        out_topk_weights = torch.zeros(
            num_recv_tokens, dtype=torch.float32, device=self.device
        )

        ts.dispatch(
            routing,
            tokens,
            topk_weights,
            out=(out_tokens, out_topk_weights, None),
        )
        torch.cuda.synchronize()

        src_rank = (self.rank - 1) % self.world_size
        expected_val = float(src_rank + 1)
        received = out_tokens[:NUM_TOKENS].float()
        self.assertTrue(
            received.eq(expected_val).all(),
            f"rank {self.rank}: expected {expected_val}, got {received[0, 0].item()}",
        )

    @skip_if_lt_x_gpu(2)
    def test_dispatch_combine_roundtrip_expert_major(self):
        """Expert-major HT roundtrip: user applies 1-D weights before combine."""
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        routing = ts.create_routing(topk_idx, layout="expert_major")

        token_val = float(self.rank + 1)
        tokens = torch.full(
            (NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16, device=self.device
        )
        out_tokens = torch.zeros(
            (num_recv_tokens, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        out_topk_weights = torch.zeros(
            num_recv_tokens, dtype=torch.float32, device=self.device
        )

        ts.dispatch(
            routing,
            tokens,
            topk_weights,
            out=(out_tokens, out_topk_weights, None),
        )
        torch.cuda.synchronize()

        # HT+expert_major: multiply by weights before combine.
        # out_topk_weights[:NUM_TOKENS] covers this rank's expert zone.
        expert_tokens = out_tokens[:NUM_TOKENS].contiguous()
        expert_weights = out_topk_weights[:NUM_TOKENS]
        expert_tokens_weighted = (
            (expert_tokens.float().mul_(expert_weights[:, None]))
            .to(torch.bfloat16)
            .contiguous()
        )

        combined = torch.zeros(
            (NUM_TOKENS, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        ts.combine(routing, expert_tokens_weighted, out=combined)
        torch.cuda.synchronize()

        # weights are 1/TOP_K = 1.0, so the roundtrip is lossless.
        expected = torch.full((NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16)
        self.assertEqual(combined.cpu(), expected)

    @skip_if_lt_x_gpu(2)
    @parametrize("explicit_rendezvous", [True, False])
    def test_dispatch_symm_mem_tokens(self, explicit_rendezvous):
        """Dispatch with symm_mem-backed input and output (zero-copy window path).

        ``explicit_rendezvous`` toggles whether the caller rendezvous the symm_mem
        tensors up front; when False, dispatch's ``make_ep_tensor`` establishes the
        rendezvous implicitly on first use.
        """
        self._init()
        ts = self.get_token_switch()
        pg = dist.distributed_c10d._get_default_group()
        num_recv_tokens = self.world_size * NUM_TOKENS

        # Allocate the token buffer in symmetric memory so NCCL EP can address
        # it via the registered ncclWindow without an extra device-side copy.
        symm_tokens = symm_mem.empty(
            NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device=self.device
        )
        if explicit_rendezvous:
            symm_mem.rendezvous(symm_tokens, group=pg)

        token_val = float(self.rank + 1)
        symm_tokens.fill_(token_val)

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        routing = ts.create_routing(topk_idx, layout="flat")

        # Output is also symm_mem-backed: dispatch writes into it via the window.
        out_tokens = symm_mem.empty(
            num_recv_tokens, HIDDEN, dtype=torch.bfloat16, device=self.device
        )
        if explicit_rendezvous:
            symm_mem.rendezvous(out_tokens, group=pg)

        out_topk_weights = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.float32, device=self.device
        )
        out_topk_idx = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.int64, device=self.device
        )

        ts.dispatch(
            routing,
            symm_tokens,
            topk_weights,
            out=(out_tokens, out_topk_weights, out_topk_idx),
        )
        torch.cuda.synchronize()

        src_rank = (self.rank - 1) % self.world_size
        expected_val = float(src_rank + 1)
        received = out_tokens[:NUM_TOKENS].float()
        self.assertTrue(
            received.eq(expected_val).all(),
            f"rank {self.rank}: expected {expected_val}, got {received[0, 0].item()}",
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("explicit_rendezvous", [True, False])
    def test_combine_symm_mem_tokens(self, explicit_rendezvous):
        """Combine with symm_mem-backed expert tokens (zero-copy window path).

        ``explicit_rendezvous`` toggles whether the caller rendezvous the symm_mem
        tensor up front; when False, combine's ``make_ep_tensor`` establishes the
        rendezvous implicitly on first use.
        """
        self._init()
        ts = self.get_token_switch()
        pg = dist.distributed_c10d._get_default_group()
        num_recv_tokens = self.world_size * NUM_TOKENS

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        routing = ts.create_routing(topk_idx, layout="flat")

        token_val = float(self.rank + 1)
        tokens = torch.full(
            (NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16, device=self.device
        )
        out_tokens = torch.zeros(
            (num_recv_tokens, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        out_topk_weights = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.float32, device=self.device
        )
        out_topk_idx = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.int64, device=self.device
        )

        ts.dispatch(
            routing,
            tokens,
            topk_weights,
            out=(out_tokens, out_topk_weights, out_topk_idx),
        )
        torch.cuda.synchronize()

        # Allocate expert_tokens in symmetric memory for zero-copy combine path.
        symm_expert_tokens = symm_mem.empty(
            NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device=self.device
        )
        if explicit_rendezvous:
            symm_mem.rendezvous(symm_expert_tokens, group=pg)
        symm_expert_tokens.copy_(out_tokens[:NUM_TOKENS])

        combined = torch.zeros(
            (NUM_TOKENS, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        ts.combine(routing, symm_expert_tokens, out=combined)
        torch.cuda.synchronize()

        expected = torch.full((NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16)
        self.assertEqual(combined.cpu(), expected)

    @skip_if_lt_x_gpu(2)
    def test_dispatch_group_gemm_combine_symm_mem(self):
        """Full pipeline: dispatch (symm_mem) -> group_gemm -> weight mul -> combine (symm_mem)."""
        self._init()
        ts = self.get_token_switch()
        pg = dist.distributed_c10d._get_default_group()
        num_recv_tokens = self.world_size * NUM_TOKENS

        token_val = float(self.rank + 1)
        tokens = torch.full(
            (NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16, device=self.device
        )

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        routing = ts.create_routing(topk_idx, layout="expert_major")

        out_tokens = symm_mem.empty(
            num_recv_tokens, HIDDEN, dtype=torch.bfloat16, device=self.device
        )
        symm_mem.rendezvous(out_tokens, group=pg)
        out_topk_weights = torch.zeros(
            num_recv_tokens, dtype=torch.float32, device=self.device
        )

        ts.dispatch(
            routing,
            tokens,
            topk_weights,
            out=(out_tokens, out_topk_weights, None),
        )
        torch.cuda.synchronize()

        # group_gemm: simulate with a scaled identity weight matrix.
        GEMM_SCALE = 2.0
        expert_tokens = out_tokens[:NUM_TOKENS].contiguous()
        weight = torch.eye(HIDDEN, dtype=torch.float32, device=self.device) * GEMM_SCALE
        gemm_out = expert_tokens.float().mm(weight).to(torch.bfloat16)

        # weight mul + allocate directly into symm_mem pool to avoid a copy.
        expert_weights = out_topk_weights[:NUM_TOKENS]
        with torch.cuda.use_mem_pool(symm_mem.get_mem_pool(self.device)):
            gemm_weighted = (
                (gemm_out.float().mul_(expert_weights[:, None]))
                .to(torch.bfloat16)
                .contiguous()
            )
        symm_mem.rendezvous(gemm_weighted, group=pg)

        combined = torch.zeros(
            (NUM_TOKENS, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        ts.combine(routing, gemm_weighted, out=combined)
        torch.cuda.synchronize()

        # weights are 1/TOP_K = 1.0 and GEMM_SCALE applied, so expected = token_val * GEMM_SCALE.
        expected_val = token_val * GEMM_SCALE
        expected = torch.full(
            (NUM_TOKENS, HIDDEN), expected_val, dtype=torch.bfloat16, device=self.device
        )
        self.assertEqual(combined, expected)


instantiate_parametrized_tests(TokenSwitchNCCLTest)


class TokenSwitchNCCL2Test(TokenSwitchNCCLTest):
    _cached_token_switch: TokenSwitchNCCL | None = None
    _cached_contract_token_switch: TokenSwitchNCCL | None = None

    @classmethod
    def backend_str(cls):
        return "nccl2"


if __name__ == "__main__":
    run_tests()
