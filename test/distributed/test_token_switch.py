# Owner(s): ["oncall: distributed"]


import gc
import importlib.util
import logging
import os
import subprocess
import sys
import unittest
from dataclasses import dataclass
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed._functional_collectives import wait_tensor
from torch.distributed._token_switch import (
    _import_mori,
    _import_nccl_ep,
    Routing,
    TokenSwitch,
    TokenSwitchMoRI,
    TokenSwitchNCCL,
)
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TestCase,
)


log = logging.getLogger(__name__)


def _nccl_ep_available() -> bool:
    # The torch._nccl_ep extension is built only with USE_NCCL_EP, and a
    # USE_SYSTEM_NCCL=ON build additionally needs the nccl-extensions wheel at runtime.
    # Actually importing it is the real check: find_spec would only locate the
    # extension without dlopening it, so it would miss a missing libnccl_ep.so.
    if not torch.cuda.is_available():
        return False
    if any(
        torch.cuda.get_device_capability(i) < (9, 0)
        for i in range(torch.cuda.device_count())
    ):
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
        "Test requires sm_90+ and a USE_NCCL_EP build (plus nccl-extensions for USE_SYSTEM_NCCL=ON)",
    )


def _mori_available() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    try:
        _import_mori()
    except Exception:
        log.debug("MoRI unavailable; skipping MoRI tests", exc_info=True)
        return False
    return True


def requires_mori():
    return skip_but_pass_in_sandcastle_if(
        not _mori_available(), "Test requires MoRI EP v2 with flydsl on ROCm"
    )


@unittest.skipIf(
    not torch.cuda.is_available() or importlib.util.find_spec("torch._nccl_ep") is None,
    "Test requires CUDA and a USE_NCCL_EP build",
)
class TokenSwitchNCCLLoadingTest(TestCase):
    hw_classification = HardwareClassification.CUDA

    def _run_script(self, script):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_import_without_python_bindings(self):
        self._run_script("""
import sys
from unittest.mock import patch

from torch.distributed._token_switch import _import_nccl_ep

with patch.dict(sys.modules, {"nccl.ep": None, "nccl._extensions": None}):
    _import_nccl_ep()
""")

    def test_without_runtime_package(self):
        self._run_script("""
import sys
from unittest.mock import patch

from torch.distributed._token_switch import _import_nccl_ep
from torch.testing._internal.common_utils import TestCase

checks = TestCase()
with patch.dict(sys.modules, {"nccl": None, "nccl.ep": None}):
    try:
        import torch._nccl_ep
    except ImportError:
        with checks.assertRaisesRegex(ImportError, "nccl-extensions"):
            _import_nccl_ep()
    else:
        native = _import_nccl_ep()
        checks.assertTrue(hasattr(native, "_NcclEpGroup"))
""")


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


def _to_local(idx, weights, first_expert, experts_per_rank):
    """Local expert ids in the token's top-k positions, -1 / weight 0 elsewhere."""
    local = idx - first_expert
    mine = (local >= 0) & (local < experts_per_rank)
    return torch.where(mine, local, -1), torch.where(mine, weights, 0)


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
        idx, w = _to_local(h.topk_idx_all[r, t], w_all[r, t], first, epr)
        out_topk_idx[:n].copy_(idx)
        out_topk_weights[:n].copy_(w)

    def _combine(self, routing, expert_tokens, out_tokens):
        h = routing.handle
        padded = expert_tokens.new_zeros(self._max_recv, expert_tokens.shape[-1])
        padded[: expert_tokens.shape[0]].copy_(expert_tokens)
        rows = self._all_gather(padded).flatten(0, 1)[h.send_flat_slot]
        out_tokens.zero_().index_add_(0, h.send_token, rows)

    def _combine_backward(self, routing, grad_tokens, out_expert_grads):
        h = routing.handle
        r, t = h.recv_src[:, 0], h.recv_src[:, 1]
        out_expert_grads[: len(r)].copy_(self._all_gather(grad_tokens)[r, t])

    def _dispatch_backward(
        self,
        routing,
        grad_recv_tokens,
        grad_recv_topk_weights,
        out_token_grads,
        out_topk_weight_grads,
    ):
        self._combine(routing, grad_recv_tokens, out_token_grads)
        h = routing.handle
        r, t = h.recv_src[:, 0], h.recv_src[:, 1]
        epr = self._experts_per_rank
        _, weights = _to_local(
            h.topk_idx_all[r, t],
            grad_recv_topk_weights[: len(r)],
            self._rank * epr,
            epr,
        )
        self._combine(routing, weights, out_topk_weight_grads)


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
        # each row: this rank's (local expert, weight) pairs from the source token's
        # top-k, in any order, and -1 with weight 0 in the remaining entries
        lo = self.rank * CONTRACT_EXPERTS_PER_RANK
        got_idx, got_w = out_idx[:n].tolist(), out_w[:n].tolist()
        for row, (r, t) in enumerate(srcs):
            want = sorted(
                (int(e) - lo, round(0.1 * (k + 1), 4))
                for k, e in enumerate(all_idx[r][t])
                if 0 <= int(e) - lo < CONTRACT_EXPERTS_PER_RANK
            )
            pairs = list(zip(got_idx[row], got_w[row]))
            got = sorted((i, round(w, 4)) for i, w in pairs if i != -1)
            self.assertEqual(got, want)
            pad = [w for i, w in pairs if i == -1]
            self.assertEqual(pad, [0.0] * (CONTRACT_TOP_K - len(want)))

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
    def backend_str(cls, device):
        return "gloo"

    @property
    def device_type(self):
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

    def test_backend_requires_dispatch_backward(self):
        class IncompleteTokenSwitch(_ReferenceTokenSwitch):
            _dispatch_backward = TokenSwitch._dispatch_backward

        with self.assertRaisesRegex(TypeError, "_dispatch_backward"):
            IncompleteTokenSwitch(
                None, self.world_size * CONTRACT_EXPERTS_PER_RANK, NUM_TOKENS
            )

    @parametrize("loss_target", ["tokens", "weights", "both", "tokens_frozen"])
    def test_dispatch_weight_gradients(self, loss_target):
        self._init()
        ts = self.get_contract_token_switch()
        ids, weights, tokens, copies, _, _ = self._contract_inputs(seed=15)
        tokens.requires_grad_(loss_target != "tokens_frozen")
        weights.requires_grad_()
        routing = ts.create_routing(ids, layout="flat")
        received, scores, _ = ts.dispatch(
            routing, tokens, weights, ts.max_recv_tokens_per_rank
        )
        score_grads = scores.new_empty(scores.numel() * 2)[::2].view_as(scores)
        score_grads.copy_(
            2 * scores.detach()
            + received[:, :1].detach() / 8
            + received[:, 1:2].detach() / 16
            + self.rank
            + 1
        )
        self.assertFalse(score_grads.is_contiguous())
        loss = received.new_zeros(())
        if loss_target in ("tokens", "both"):
            loss = loss + received.sum()
        if loss_target != "tokens":
            loss = loss + (scores * score_grads).sum()
        loss.backward()
        expected = (
            torch.zeros_like(tokens)
            if loss_target == "weights"
            else copies.expand_as(tokens)
        )
        if loss_target == "tokens_frozen":
            self.assertIsNone(tokens.grad)
        else:
            self.assertEqual(tokens.grad, expected)
        expected_weights = (
            2 * weights.detach()
            + tokens[:, :1] / 8
            + tokens[:, 1:2] / 16
            + ids // CONTRACT_EXPERTS_PER_RANK
            + 1
        )
        if loss_target == "tokens":
            expected_weights.zero_()
        self.assertEqual(weights.grad, expected_weights)


instantiate_parametrized_tests(TokenSwitchReferenceTest)


@instantiate_parametrized_tests
class _TokenSwitchFlatTests:
    """Flat-layout tests with TOP_K=1 and one expert per rank.

    The host class provides ``device``, ``_init()``, ``get_token_switch()``
    (world_size experts, top-k TOP_K) and ``get_weight_grad_token_switch(capacity)``
    (two experts per rank, top-k 3).
    """

    def _weight_grad_inputs(self):
        token = torch.arange(NUM_TOKENS, device=self.device)
        destination = (token + self.rank) % self.world_size
        ids = torch.stack(
            (
                destination * 2,
                destination * 2 + 1,
                (destination + 1) % self.world_size * 2,
            ),
            dim=1,
        )
        ids = torch.where(token[:, None] % 2 == 0, ids, ids.flip(1))
        tokens = (
            torch.arange(HIDDEN, device=self.device).expand(NUM_TOKENS, -1).float() / 8
        )
        tokens = tokens.contiguous()
        tokens[:, 0] = self.rank + 1
        tokens[:, 1] = token + 1
        weights = (
            torch.arange(NUM_TOKENS * 3, device=self.device)
            .reshape(NUM_TOKENS, 3)
            .float()
        )
        weights = weights / 32 + 0.125 + self.rank / 16
        return ids, tokens.to(torch.bfloat16), weights

    @skip_if_lt_x_gpu(2)
    @parametrize("layout", ["flat", "expert_major"])
    @parametrize("loss_target", ["tokens", "weights", "both", "tokens_frozen"])
    def test_dispatch_weight_gradients(self, layout, loss_target):
        self._init()
        ids, tokens, weights = self._weight_grad_inputs()
        capacity = NUM_TOKENS * self.world_size * 3
        ts = self.get_weight_grad_token_switch(capacity)
        routing = ts.create_routing(ids, layout=layout)
        tokens.requires_grad_(loss_target != "tokens_frozen")
        weights.requires_grad_()
        received, scores, _ = ts.dispatch(routing, tokens, weights, capacity)
        token_grads = received.new_empty(received.numel() * 2)[::2].view_as(received)
        token_grads.copy_(received.detach() / 16)
        score_grads = scores.new_empty(scores.numel() * 2)[::2].view_as(scores)
        coefficient = received[:, 0].detach() / 8 + received[:, 1].detach() / 16 + 1
        coefficient = coefficient[:, None] if layout == "flat" else coefficient
        # Nonzero gradients in invalid flat slots must not reach source weights.
        score_grads.copy_(2 * scores.detach() + coefficient)
        self.assertFalse(token_grads.is_contiguous())
        self.assertFalse(score_grads.is_contiguous())
        outputs, gradients = [], []
        if loss_target in ("tokens", "both"):
            outputs.append(received)
            gradients.append(token_grads)
        if loss_target != "tokens":
            outputs.append(scores)
            gradients.append(score_grads)
        torch.autograd.backward(outputs, gradients)
        if loss_target == "tokens_frozen":
            self.assertIsNone(tokens.grad)
        else:
            copies = 2 if layout == "flat" else 3
            expected_tokens = tokens.detach() * (copies / 16)
            if loss_target == "weights":
                expected_tokens.zero_()
            self.assertEqual(tokens.grad, expected_tokens, rtol=0, atol=0)
        expected_weights = (
            2 * weights.detach() + tokens[:, :1] / 8 + tokens[:, 1:2] / 16 + 1
        )
        if loss_target == "tokens":
            expected_weights.zero_()
        self.assertEqual(weights.grad, expected_weights, rtol=0, atol=0)
        torch.cuda.synchronize()
        dist.barrier()
        del received, scores, outputs, gradients
        routing = None
        ts = None

    @skip_if_lt_x_gpu(2)
    @parametrize("layout", ["flat", "expert_major"])
    def test_dispatch_data_dependent_weights(self, layout):
        self._init()
        ids, tokens, _ = self._weight_grad_inputs()
        tokens.requires_grad_()
        weights = tokens[:, :3].float().softmax(dim=1)
        weights.retain_grad()
        capacity = NUM_TOKENS * self.world_size * 3
        ts = self.get_weight_grad_token_switch(capacity)
        routing = ts.create_routing(ids, layout=layout)
        received, scores, _ = ts.dispatch(routing, tokens, weights, capacity)
        routed_weights = scores.sum(dim=1) if layout == "flat" else scores
        expert_tokens = received.float() * (self.rank + 1)
        expert_tokens = (expert_tokens * routed_weights[:, None]).to(received.dtype)
        result = ts.combine(routing, expert_tokens)
        loss = result.float().sum()
        loss.backward()
        reference = tokens.detach().clone().requires_grad_()
        reference_weights = reference[:, :3].float().softmax(dim=1)
        reference_weights.retain_grad()
        rank_scale = ids // 2 + 1
        coefficient = (reference_weights * rank_scale).sum(dim=1, keepdim=True)
        expected = reference.float() * coefficient
        expected.sum().backward()
        self.assertEqual(result.float(), expected, rtol=0.02, atol=0.004)
        self.assertEqual(weights.grad, reference_weights.grad, rtol=0, atol=0)
        self.assertEqual(tokens.grad, reference.grad, rtol=0.02, atol=0.004)
        torch.cuda.synchronize()
        dist.barrier()
        del received, scores, expert_tokens, result, loss
        routing = None
        ts = None

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
        routing = ts.create_routing(topk_idx, per_expert_counts, layout="flat")
        if isinstance(ts, TokenSwitchNCCL):
            self.assertEqual(routing.handle.get_num_recv_tokens(), NUM_TOKENS)
        self.assertEqual(per_expert_counts.dtype, torch.int32)
        self.assertEqual(per_expert_counts.numel(), num_local_experts)
        self.assertEqual(per_expert_counts.item(), NUM_TOKENS)
        torch.cuda.synchronize()

    @skip_if_lt_x_gpu(2)
    @parametrize("dtype", [torch.float32, torch.int16, torch.bool])
    def test_create_routing_invalid_dtype(self, dtype):
        self._init()
        ts = self.get_token_switch()
        topk_idx = torch.zeros(NUM_TOKENS, TOP_K, dtype=dtype, device=self.device)
        with self.assertRaisesRegex(
            TypeError, "topk_idx must have dtype int32 or int64"
        ):
            ts.create_routing(topk_idx, layout="flat")

    @skip_if_lt_x_gpu(2)
    def test_create_routing_with_then_without_counts(self):
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS
        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        per_expert_counts = torch.zeros(1, dtype=torch.int32, device=self.device)
        ts.create_routing(topk_idx, per_expert_counts, layout="flat")
        routing = ts.create_routing(topk_idx, layout="flat")

        tokens = torch.full(
            (NUM_TOKENS, HIDDEN),
            float(self.rank + 1),
            dtype=torch.bfloat16,
            device=self.device,
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
        expected = torch.full(
            (NUM_TOKENS, HIDDEN),
            float(src_rank + 1),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.assertEqual(out_tokens[:NUM_TOKENS], expected)

    @skip_if_lt_x_gpu(2)
    def test_create_routing_invalid_counts(self):
        self._init()
        ts = self.get_token_switch()
        topk_idx, _topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        for bad_counts in (
            torch.zeros(1, dtype=torch.int64, device=self.device),
            torch.zeros(0, dtype=torch.int32, device=self.device),
            torch.zeros(2, dtype=torch.int32, device=self.device),
            torch.zeros((1, 1), dtype=torch.int32, device=self.device),
            torch.zeros(1, dtype=torch.int32),
        ):
            with self.assertRaisesRegex(
                (RuntimeError, ValueError), "must be a 1D int32 tensor"
            ):
                ts.create_routing(topk_idx, bad_counts, layout="flat")

    @skip_if_lt_x_gpu(2)
    def test_dispatch(self):
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        per_expert_counts = torch.zeros(1, dtype=torch.int32, device=self.device)
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
        per_expert_counts = torch.zeros(1, dtype=torch.int32, device=self.device)
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
        per_expert_counts = torch.zeros(1, dtype=torch.int32, device=self.device)
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
        per_expert_counts = torch.zeros(1, dtype=torch.int32, device=self.device)
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
        per_expert_counts = torch.zeros(1, dtype=torch.int32, device=self.device)
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
        per_expert_counts = torch.zeros(1, dtype=torch.int32, device=self.device)
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


class _TokenSwitchExpertMajorTests:
    """Expert-major layout tests, shared by the backends that support it."""

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


@instantiate_parametrized_tests
class _TokenSwitchNCCLSymmMemTests:
    """NCCL-only symm_mem window tests; the host class is TokenSwitchNCCLTest."""

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
    def test_dispatch_symm_mem_tokens_storage_offset(self):
        """Dispatch with symm_mem views that start past the beginning of their storage."""
        self._init()
        ts = self.get_token_switch()
        pg = dist.distributed_c10d._get_default_group()
        num_recv_tokens = self.world_size * NUM_TOKENS

        symm_buf = symm_mem.empty(
            2 * NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device=self.device
        )
        symm_mem.rendezvous(symm_buf, group=pg)
        symm_buf.fill_(-1)
        symm_tokens = symm_buf[NUM_TOKENS:]
        symm_tokens.fill_(float(self.rank + 1))

        out_buf = symm_mem.empty(
            2 * num_recv_tokens, HIDDEN, dtype=torch.bfloat16, device=self.device
        )
        symm_mem.rendezvous(out_buf, group=pg)
        out_buf.zero_()
        out_tokens = out_buf[num_recv_tokens:]

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        routing = ts.create_routing(topk_idx, layout="flat")
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
        expected = torch.full(
            (NUM_TOKENS, HIDDEN),
            float(src_rank + 1),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.assertEqual(out_tokens[:NUM_TOKENS], expected)
        self.assertEqual(out_buf[:num_recv_tokens], torch.zeros_like(out_tokens))

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


@requires_nccl_ep()
class TokenSwitchNCCLTest(
    _TokenSwitchNCCLSymmMemTests,
    _TokenSwitchFlatTests,
    _TokenSwitchExpertMajorTests,
    _TokenSwitchContractTests,
    MultiProcContinuousTest,
):
    _cached_token_switch: TokenSwitchNCCL | None = None
    _cached_contract_token_switch: TokenSwitchNCCL | None = None
    contract_dtype = torch.bfloat16

    @classmethod
    def backend_str(cls, device):
        return "nccl"

    @classmethod
    def _cleanup_worker(cls):
        cls._cached_token_switch = None
        cls._cached_contract_token_switch = None

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
    def get_weight_grad_token_switch(cls, capacity: int) -> TokenSwitchNCCL:
        pg = dist.distributed_c10d._get_default_group()
        return TokenSwitchNCCL(
            pg, dist.get_world_size(pg) * 2, NUM_TOKENS, capacity, TOKEN_SIZE_BYTES
        )

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
        # Zero-copy dispatch/combine needs NCCL window handles.
        # The default CUDA backend would silently fall back to a device-pointer copy.
        if symm_mem.get_backend(self.device) != "NCCL":
            symm_mem.set_backend("NCCL")
        self.assertEqual(symm_mem.get_backend(self.device), "NCCL")
        dist.barrier()

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


class TokenSwitchNCCL2Test(TokenSwitchNCCLTest):
    _cached_token_switch: TokenSwitchNCCL | None = None
    _cached_contract_token_switch: TokenSwitchNCCL | None = None

    @classmethod
    def backend_str(cls, device):
        return "nccl2"


@requires_nccl_ep()
class TokenSwitchRoutingDtypeTest(MultiProcContinuousTest):
    _cached_token_switch: TokenSwitchNCCL | None = None

    @classmethod
    def backend_str(cls, device):
        return "nccl"

    def _get_token_switch(self):
        cls = type(self)
        if cls._cached_token_switch is None:
            pg = dist.distributed_c10d._get_default_group()
            dist.barrier(pg)
            cls._cached_token_switch = TokenSwitchNCCL(
                pg,
                self.world_size,
                NUM_TOKENS,
                self.world_size * NUM_TOKENS,
                TOKEN_SIZE_BYTES,
            )
        return cls._cached_token_switch

    @classmethod
    def _cleanup_worker(cls):
        cls._cached_token_switch = None

    @skip_if_lt_x_gpu(2)
    @dtypes(torch.int32, torch.int64)
    @parametrize("layout", ["flat", "expert_major"])
    def test_dispatch_combine_roundtrip(self, device, dtype, layout):
        device = torch.device(torch.device(device).type, self.rank)
        torch.cuda.set_device(device)
        ts = self._get_token_switch()
        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, device
        )
        topk_idx = topk_idx.to(dtype)
        routing = ts.create_routing(topk_idx, layout=layout)
        tokens = torch.full(
            (NUM_TOKENS, HIDDEN), self.rank + 1, dtype=torch.bfloat16, device=device
        )
        dispatched, weights, indices = ts.dispatch(
            routing, tokens, topk_weights, ts.max_recv_tokens_per_rank
        )
        src_rank = (self.rank - 1) % self.world_size
        self.assertEqual(dispatched[:NUM_TOKENS], torch.full_like(tokens, src_rank + 1))
        self.assertEqual(weights[:NUM_TOKENS], torch.ones_like(weights[:NUM_TOKENS]))
        if layout == "flat":
            self.assertEqual(indices.dtype, torch.int64)
        else:
            self.assertIsNone(indices)
        combined = ts.combine(routing, dispatched[:NUM_TOKENS].contiguous())
        self.assertEqual(combined, tokens)
        self.assertEqual(topk_idx.dtype, dtype)

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "layout,output_index",
        [
            ("flat", None),
            ("flat", 0),
            ("flat", 1),
            ("flat", 2),
            ("expert_major", None),
            ("expert_major", 0),
            ("expert_major", 1),
        ],
    )
    def test_dispatch_requires_full_capacity(self, device, layout, output_index):
        device = torch.device(torch.device(device).type, self.rank)
        torch.cuda.set_device(device)
        ts = self._get_token_switch()
        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, device
        )
        routing = ts.create_routing(topk_idx, layout=layout)
        num_recv = routing.handle.get_num_recv_tokens()
        self.assertEqual(num_recv, NUM_TOKENS)
        self.assertLess(num_recv, ts.max_recv_tokens_per_rank)
        tokens = torch.zeros(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)
        error = f"require at least {ts.max_recv_tokens_per_rank} rows"
        if output_index is None:
            with self.assertRaisesRegex(ValueError, error):
                ts.dispatch(routing, tokens, topk_weights, num_recv)
        else:
            out = list(
                ts.dispatch(routing, tokens, topk_weights, ts.max_recv_tokens_per_rank)
            )
            out[output_index] = out[output_index][:num_recv]
            with self.assertRaisesRegex(ValueError, error):
                ts.dispatch(routing, tokens, topk_weights, out=tuple(out))


instantiate_device_type_tests(TokenSwitchRoutingDtypeTest, globals(), only_for="cuda")


@instantiate_parametrized_tests
class _TokenSwitchMoRIOnlyTests:
    """MoRI-only tests; the host class is TokenSwitchMoRITest."""

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "layout,output_index",
        [
            ("flat", 0),
            ("flat", 1),
            ("flat", 2),
            ("expert_major", 0),
            ("expert_major", 1),
        ],
    )
    def test_dispatch_short_out_raises(self, layout, output_index):
        self._init()
        ts = self.get_contract_token_switch()
        topk_idx, w, x, _nd, _exp, _ = self._contract_inputs(seed=31)
        routing = ts.create_routing(topk_idx, layout=layout)
        need = ts.max_expert_major_slots_per_rank
        if layout == "flat":
            need = ts.max_recv_tokens_per_rank
        out = list(
            ts._alloc_dispatch_outputs(routing, x, w, need, HIDDEN, CONTRACT_TOP_K)
        )
        out[output_index] = out[output_index][: need - 1]
        with self.assertRaisesRegex(ValueError, f"require at least {need} rows"):
            ts.dispatch(routing, x, w, out=tuple(out))

    @skip_if_lt_x_gpu(2)
    @parametrize("layout", ["flat", "expert_major"])
    def test_combine_backward_zeroes_unused_rows(self, layout):
        # combine ignores rows past the received slots, so their gradient is zero.
        self._init()
        ts = self.get_contract_token_switch()
        topk_idx, w, x, _nd, expected, _ = self._contract_inputs(seed=32)
        epr = CONTRACT_EXPERTS_PER_RANK
        counts = torch.zeros(epr, dtype=torch.int32, device=self.device)
        routing = ts.create_routing(topk_idx, counts, layout=layout)
        max_recv = ts.max_recv_tokens_per_rank
        if layout == "flat":
            # Fill the receive buffers of ranks 0 and 1 first, so their rows past
            # this routing's receive count hold stale tokens.
            ids = torch.arange(CONTRACT_TOP_K, device=self.device) * epr
            fill = ts.create_routing(ids.repeat(NUM_TOKENS, 1), layout="flat")
            ts.dispatch(fill, x, w, max_recv)
        d, _, _ = ts.dispatch(routing, x, w, max_recv)
        n = len(expected) if layout == "flat" else int(counts.sum())
        d = d.detach().requires_grad_(True)
        ts.combine(routing, d).float().sum().backward()
        grad = d.grad.float()
        self.assertEqual(grad[:n], torch.ones_like(grad[:n]))
        self.assertEqual(grad[n:], torch.zeros_like(grad[n:]))


@requires_mori()
class TokenSwitchMoRITest(
    _TokenSwitchMoRIOnlyTests,
    _TokenSwitchFlatTests,
    _TokenSwitchExpertMajorTests,
    _TokenSwitchContractTests,
    MultiProcContinuousTest,
):
    _cached_token_switch: TokenSwitchMoRI | None = None
    _cached_contract_token_switch: TokenSwitchMoRI | None = None
    _cached_weight_grad_token_switch: TokenSwitchMoRI | None = None
    contract_dtype = torch.bfloat16

    @classmethod
    def backend_str(cls, device):
        return "nccl"

    @classmethod
    def _cleanup_worker(cls):
        cls._cached_token_switch = None
        cls._cached_contract_token_switch = None
        cls._cached_weight_grad_token_switch = None

    @property
    def device(self):
        return torch.device("cuda", self.rank)

    @classmethod
    def get_token_switch(cls) -> TokenSwitchMoRI:
        if cls._cached_token_switch is None:
            pg = dist.distributed_c10d._get_default_group()
            cls._cached_token_switch = TokenSwitchMoRI(
                pg, dist.get_world_size(pg), NUM_TOKENS, HIDDEN, TOP_K
            )
        return cls._cached_token_switch

    @classmethod
    def get_weight_grad_token_switch(cls, capacity: int) -> TokenSwitchMoRI:
        if cls._cached_weight_grad_token_switch is None:
            pg = dist.distributed_c10d._get_default_group()
            cls._cached_weight_grad_token_switch = TokenSwitchMoRI(
                pg, dist.get_world_size(pg) * 2, NUM_TOKENS, HIDDEN, 3
            )
        return cls._cached_weight_grad_token_switch

    @classmethod
    def get_contract_token_switch(cls) -> TokenSwitchMoRI:
        if cls._cached_contract_token_switch is None:
            pg = dist.distributed_c10d._get_default_group()
            E = dist.get_world_size(pg) * CONTRACT_EXPERTS_PER_RANK
            cls._cached_contract_token_switch = TokenSwitchMoRI(
                pg, E, NUM_TOKENS, HIDDEN, CONTRACT_TOP_K
            )
        return cls._cached_contract_token_switch

    def _init(self):
        torch.cuda.set_device(self.device)
        dist.barrier()

    @skip_if_lt_x_gpu(2)
    def test_expert_major_slots_and_combine(self):
        # One slot per (token, local expert), grouped by local expert; a second
        # dispatch on the Routing keeps the slots; unweighted combine of an identity
        # expert returns K * x (one row per (token, expert) pair).
        self._init()
        ts = self.get_contract_token_switch()
        epr = CONTRACT_EXPERTS_PER_RANK
        topk_idx, w, x, _nd, _exp, all_idx = self._contract_inputs(seed=21)
        counts = torch.zeros(epr, dtype=torch.int32, device=self.device)
        routing = ts.create_routing(topk_idx, counts, layout="expert_major")
        out_tokens, out_w, out_idx = ts._alloc_dispatch_outputs(
            routing, x, w, ts.max_recv_tokens_per_rank, HIDDEN, CONTRACT_TOP_K
        )
        self.assertIsNone(out_idx)
        ts.dispatch(routing, x, w, out=(out_tokens, out_w, None))
        lo = self.rank * epr
        zones = [
            sorted(
                (r, t, round(0.1 * (k + 1), 4))
                for r in range(self.world_size)
                for t in range(NUM_TOKENS)
                for k, e in enumerate(all_idx[r][t].tolist())
                if e == lo + j
            )
            for j in range(epr)
        ]
        self.assertEqual(counts.cpu().tolist(), [len(z) for z in zones])
        n = sum(len(z) for z in zones)
        rows = out_tokens[:n].float().cpu()
        wts = out_w[:n].cpu().tolist()
        start = 0
        for zone in zones:
            got = [
                (int(rows[i, 0]), int(rows[i, 1]), round(wts[i], 4))
                for i in range(start, start + len(zone))
            ]
            self.assertEqual(sorted(got), zone)
            start += len(zone)
        first = out_tokens[:n].clone()
        x2 = x.clone()
        x2[:, 2:] = -x2[:, 2:]
        ts.dispatch(routing, x2, w, out=(out_tokens, out_w, None))
        self.assertEqual(out_tokens[:n, :2], first[:, :2])
        self.assertEqual(out_tokens[:n, 2:], -first[:, 2:])
        combined = torch.zeros_like(x)
        ts.combine(routing, first, out=combined)
        self.assertEqual(combined.float(), CONTRACT_TOP_K * x.float())

    @skip_if_lt_x_gpu(2)
    def test_expert_major_interleaved_autograd(self):
        # Two expert_major Routings, two dispatches before either combine; each
        # token's gradient is its number of (token, expert) pairs, K.
        self._init()
        ts = self.get_contract_token_switch()
        max_recv = ts.max_recv_tokens_per_rank
        idx1, w1, x1, _nd1, _e1, _ = self._contract_inputs(seed=23)
        idx2, w2, x2, _nd2, _e2, _ = self._contract_inputs(seed=24)
        r1 = ts.create_routing(idx1, layout="expert_major")
        r2 = ts.create_routing(idx2, layout="expert_major")
        t1 = x1.clone().requires_grad_(True)
        t2 = x2.clone().requires_grad_(True)
        d1, _, _ = ts.dispatch(r1, t1, w1, max_recv)
        d2, _, _ = ts.dispatch(r2, t2, w2, max_recv)
        y1 = ts.combine(r1, d1)
        y2 = ts.combine(r2, d2)
        (y1.sum() + 2 * y2.sum()).backward()
        k = CONTRACT_TOP_K
        self.assertEqual(y1.float(), k * x1.float())
        self.assertEqual(y2.float(), k * x2.float())
        self.assertEqual(t1.grad.float(), torch.full_like(x1, k, dtype=torch.float32))
        self.assertEqual(
            t2.grad.float(), torch.full_like(x2, 2 * k, dtype=torch.float32)
        )


def _work_count():
    return torch._C._distributed_c10d._get_work_registry_size()


@requires_nccl_ep()
class TokenSwitchLifecycleTest(MultiProcContinuousTest):
    hw_classification = HardwareClassification.CUDA

    @classmethod
    def backend_str(cls):
        return "nccl"

    def _switch(self, close_on_cleanup=True):
        torch.cuda.set_device(self.rank)
        dist.barrier()
        switch = TokenSwitchNCCL(
            dist.distributed_c10d._get_default_group(),
            self.world_size,
            16,
            self.world_size * 16,
            128,
        )
        if close_on_cleanup:
            self.addCleanup(switch.close)
        return switch

    def _ids(self, shift=1, dtype=torch.int64):
        dst = (self.rank + shift) % self.world_size
        return torch.full((16, 1), dst, device="cuda", dtype=dtype)

    def _inputs(self, close_on_cleanup=True):
        switch = self._switch(close_on_cleanup)
        return switch, switch.create_routing(self._ids(), layout="flat")

    def _buffers(self, rank, phase=0):
        # Row-distinct payloads and weights so misrouted or permuted rows fail.
        capacity = self.world_size * 16
        rows = torch.arange(16, device="cuda") + 16 * (rank + self.world_size * phase)
        tokens = (rows[:, None] + 1).expand(16, 64).to(torch.bfloat16)
        weights = (rows[:, None] + 1).float() / 128
        received = tokens.new_zeros(capacity, 64)
        scores = weights.new_zeros(capacity, 1)
        ids = torch.full((capacity, 1), -1, device="cuda", dtype=torch.int64)
        return tokens, weights, received, scores, ids

    def _source(self, shift=1, phase=0):
        return self._buffers((self.rank - shift) % self.world_size, phase)[:2]

    @skip_if_lt_x_gpu(2)
    def test_close_with_retained_routing(self):
        switch, routing = self._inputs()
        tokens, weights, received, scores, ids = self._buffers(self.rank)
        other_device = (self.rank + 1) % self.world_size
        with torch.cuda.device(other_device):
            switch.close()
            switch.close()
            self.assertEqual(torch.cuda.current_device(), other_device)
        with self.assertRaisesRegex(ValueError, "group is closed"):
            switch.dispatch(routing, tokens, weights, out=(received, scores, ids))
        with self.assertRaisesRegex(ValueError, "group is closed"):
            routing.handle.get_num_recv_tokens()
        with self.assertRaisesRegex(ValueError, "group is closed"):
            switch.create_routing(routing.topk_idx, layout="flat")
        del routing, switch
        gc.collect()

    @skip_if_lt_x_gpu(2)
    def test_handle_retains_group(self):
        # The switch must be unreferenced here, so it is not closed on cleanup.
        switch, routing = self._inputs(close_on_cleanup=False)
        ep = switch._ep
        del switch
        gc.collect()
        self.assertEqual(routing.handle.get_num_recv_tokens(), 16)
        tokens, weights, received, scores, ids = self._buffers(self.rank)
        ep._nccl_ep_dispatch(routing.handle, tokens, weights, received, scores, ids)
        source_tokens, source_weights = self._source()
        self.assertEqual(received[:16], source_tokens)
        self.assertEqual(scores[:16], source_weights)
        del routing
        gc.collect()

    @skip_if_lt_x_gpu(2)
    def test_rejects_mismatched_shapes(self):
        # Every rank fails the same check before communicating, so none hangs.
        switch, routing = self._inputs()
        ep = switch._ep
        handle = routing.handle
        tokens, weights, received, scores, ids = self._buffers(self.rank)
        wide = tokens.new_zeros(16, 65)
        wide_received = received.new_zeros(received.shape[0], 65)
        dispatch_cases = [
            ("tokens must have one row", tokens[:8], weights[:8], received, ids),
            ("above max_token_bytes", wide, weights, wide_received, ids),
            ("topk_weights must be float32", tokens, weights.double(), received, ids),
            ("out_tokens must have at least", tokens, weights, received[:8], ids),
            ("out_topk_idx must be", tokens, weights, received, ids[:8]),
        ]
        for error, src, src_weights, dst, dst_ids in dispatch_cases:
            with self.assertRaisesRegex(ValueError, error):
                ep._nccl_ep_dispatch(handle, src, src_weights, dst, scores, dst_ids)
        with self.assertRaisesRegex(ValueError, "combine output must be"):
            ep._nccl_ep_combine(handle, received, tokens[:8].clone())
        with self.assertRaisesRegex(ValueError, "topk_idx must be"):
            switch.create_routing(self._ids()[:, 0], layout="flat")

    @skip_if_lt_x_gpu(2)
    def test_async_roundtrip_and_backward(self):
        switch, routing = self._inputs()
        ep = switch._ep
        producer = torch.cuda.Stream()
        consumer = torch.cuda.Stream()
        initial_work_count = _work_count()
        with torch.cuda.stream(producer):
            tokens, weights, received, scores, ids = self._buffers(self.rank)
            ep._nccl_ep_dispatch(
                routing.handle, tokens, weights, received, scores, ids, async_op=True
            )
        self.assertGreater(_work_count(), initial_work_count)
        with torch.cuda.stream(consumer):
            wait_tensor(received)
            wait_tensor(scores)
            wait_tensor(ids)
            combined = torch.zeros_like(tokens)
            ep._nccl_ep_combine(routing.handle, received, combined, async_op=True)
        with torch.cuda.stream(producer):
            wait_tensor(combined)
            expert_grads = torch.zeros_like(received)
            ep._nccl_ep_dispatch_backward(
                routing.handle, tokens, expert_grads, async_op=True
            )
        with torch.cuda.stream(consumer):
            wait_tensor(expert_grads)
            source_grads = torch.zeros_like(tokens)
            score_grads = torch.zeros_like(weights)
            ep._nccl_ep_combine_backward(
                routing.handle,
                received,
                scores,
                source_grads,
                score_grads,
                async_op=True,
            )
        with torch.cuda.stream(producer):
            wait_tensor(source_grads)
            wait_tensor(score_grads)
            source_tokens, source_weights = self._source()
            self.assertEqual(received[:16], source_tokens)
            self.assertEqual(scores[:16], source_weights)
            self.assertEqual(combined, tokens)
            self.assertEqual(source_grads, tokens)
            self.assertEqual(expert_grads, received)
            self.assertEqual(score_grads, weights)
        producer.synchronize()
        self.assertEqual(_work_count(), initial_work_count)
        del routing, switch
        gc.collect()

    @skip_if_lt_x_gpu(2)
    def test_int32_routing_on_producer_stream(self):
        switch = self._switch()
        producer = torch.cuda.Stream()
        consumer = torch.cuda.Stream()
        for shift in range(1, 4):
            with torch.cuda.stream(producer):
                ids = self._ids(shift, dtype=torch.int32)
                routing = switch.create_routing(ids, layout="flat")
                tokens, weights, received, scores, received_ids = self._buffers(
                    self.rank
                )
                switch._ep._nccl_ep_dispatch(
                    routing.handle,
                    tokens,
                    weights,
                    received,
                    scores,
                    received_ids,
                    async_op=True,
                )
            with torch.cuda.stream(consumer):
                wait_tensor(received)
                wait_tensor(scores)
                wait_tensor(received_ids)
                self.assertEqual(received[:16], self._source(shift)[0])
            consumer.synchronize()
            del routing

    @skip_if_lt_x_gpu(2)
    def test_async_wait_on_other_stream(self):
        switch, routing = self._inputs()
        producer = torch.cuda.Stream()
        consumer = torch.cuda.Stream()
        initial_work_count = _work_count()
        for phase in range(3):
            with torch.cuda.stream(producer):
                tokens, weights, received, scores, ids = self._buffers(self.rank, phase)
                switch._ep._nccl_ep_dispatch(
                    routing.handle,
                    tokens,
                    weights,
                    received,
                    scores,
                    ids,
                    async_op=True,
                )
            del tokens, weights
            self.assertGreater(_work_count(), initial_work_count)
            with torch.cuda.stream(consumer):
                ready = wait_tensor(received)
                wait_tensor(scores)
                wait_tensor(ids)
                result = ready[:16].clone()
            consumer.synchronize()
            self.assertEqual(result, self._source(phase=phase)[0])
            self.assertEqual(_work_count(), initial_work_count)
        del routing, switch
        gc.collect()


class TestRoutingUpdate(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_update_reuses_handle(self):
        switch = TokenSwitchNCCL.__new__(TokenSwitchNCCL)
        handle = Mock()
        old_ids = torch.zeros(4, 1, dtype=torch.int64)
        new_ids = torch.ones_like(old_ids)
        counts = torch.zeros(1, dtype=torch.int32)
        routing = Routing(handle, old_ids, "expert_major")
        updated = switch.update_routing(routing, new_ids, counts)
        handle.update.assert_called_once()
        self.assertIs(handle.update.call_args.args[0], new_ids)
        self.assertEqual(handle.update.call_args.args[1], counts)
        self.assertIsNot(handle.update.call_args.args[1], counts)
        self.assertIs(updated.handle, handle)
        self.assertIs(updated.topk_idx, new_ids)
        self.assertEqual(updated.layout, routing.layout)
        self.assertIs(routing.topk_idx, old_ids)


def _source_rows(rank):
    return torch.arange(16, device="cuda") + 16 * rank + 1


@requires_nccl_ep()
class TokenSwitchRoutingTest(MultiProcContinuousTest):
    hw_classification = HardwareClassification.CUDA

    @classmethod
    def backend_str(cls):
        return "nccl"

    def _check_received(self, received, sources):
        # Receive-slot order is layout-specific; compare the received row set.
        values = received[:, 0].float()
        values = values[values != 0].sort().values
        empty = torch.empty(0, dtype=torch.int64, device="cuda")
        expected = torch.cat([empty, *map(_source_rows, sources)]).float()
        self.assertEqual(values, expected.sort().values, rtol=0, atol=0)

    @skip_if_lt_x_gpu(2)
    def test_changing_routes_eager_and_graph(self):
        torch.cuda.set_device(self.rank)
        for layout in ("flat", "expert_major"):
            for dtype in (torch.int32, torch.int64):
                switch = TokenSwitchNCCL(
                    dist.distributed_c10d._get_default_group(),
                    self.world_size,
                    16,
                    self.world_size * 16,
                    128,
                )
                self.addCleanup(switch.close)
                ids = torch.full((16, 1), self.rank, device="cuda", dtype=dtype)
                counts = torch.zeros(1, device="cuda", dtype=torch.int32)
                routing = switch.create_routing(ids, counts, layout=layout)
                with self.assertRaisesRegex(TypeError, "int32 or int64"):
                    routing.handle.update(ids.float(), counts)
                with self.assertRaisesRegex(ValueError, "original top-k shape"):
                    routing.handle.update(ids[:8], counts)
                with self.assertRaisesRegex(ValueError, "original device"):
                    routing.handle.update(torch.zeros(16, 1, dtype=dtype), counts)
                if layout == "expert_major":
                    with self.assertRaisesRegex(ValueError, "per-expert token counts"):
                        routing.handle.update(ids)
                rows = _source_rows(self.rank)[:, None]
                tokens = rows.expand(16, 64).to(torch.bfloat16)
                weights = rows.float() / 128
                capacity = self.world_size * 16
                outputs = switch._alloc_dispatch_outputs(
                    routing, tokens, weights, capacity, 64, 1
                )
                combined = torch.zeros_like(tokens)
                token_grads = torch.zeros_like(tokens)
                weight_grads = torch.zeros_like(weights)
                expert_grads = torch.zeros_like(outputs[0])

                def step():
                    updated = switch.update_routing(routing, ids, counts)
                    for output in outputs:
                        if output is not None:
                            output.zero_()
                    switch.dispatch(updated, tokens, weights, out=outputs)
                    switch.combine(updated, outputs[0], out=combined)
                    switch._dispatch_backward(
                        updated, outputs[0], outputs[1], token_grads, weight_grads
                    )
                    expert_grads.zero_()
                    switch._combine_backward(updated, tokens, expert_grads)

                # Warm every native variant before capture, on a side stream.
                warmup = torch.cuda.Stream()
                warmup.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup):
                    for _ in range(3):
                        step()
                torch.cuda.current_stream().wait_stream(warmup)
                for phase in range(3):
                    ids.fill_((self.rank + phase + 1) % self.world_size)
                    step()
                    self.assertEqual(combined, tokens, rtol=0, atol=0)
                    self.assertEqual(token_grads, tokens, rtol=0, atol=0)
                    self.assertEqual(weight_grads, weights, rtol=0, atol=0)
                    self.assertEqual(expert_grads, outputs[0], rtol=0, atol=0)
                    source = (self.rank - phase - 1) % self.world_size
                    self._check_received(outputs[0], [source])
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    step()
                for phase in range(self.world_size + 2):
                    # Alternate balanced routes with all ranks targeting one rank.
                    destination = (
                        phase % self.world_size
                        if phase % 2
                        else (self.rank + phase) % self.world_size
                    )
                    ids.fill_(destination)
                    graph.replay()
                    self.assertEqual(combined, tokens, rtol=0, atol=0)
                    self.assertEqual(token_grads, tokens, rtol=0, atol=0)
                    self.assertEqual(weight_grads, weights, rtol=0, atol=0)
                    self.assertEqual(expert_grads, outputs[0], rtol=0, atol=0)
                    sources = [
                        source
                        for source in range(self.world_size)
                        if (
                            phase % self.world_size
                            if phase % 2
                            else (source + phase) % self.world_size
                        )
                        == self.rank
                    ]
                    self._check_received(outputs[0], sources)
                    # One local expert, so both layouts count every received token.
                    self.assertEqual(
                        counts,
                        torch.full_like(counts, 16 * len(sources)),
                        rtol=0,
                        atol=0,
                    )
                del graph
                gc.collect()
                switch.close()
                gc.collect()


@requires_nccl_ep()
class TokenSwitchCompileTest(MultiProcContinuousTest):
    hw_classification = HardwareClassification.CUDA

    @classmethod
    def backend_str(cls):
        return "nccl"

    @skip_if_lt_x_gpu(2)
    def test_fullgraph_changing_route_training_replay(self):
        torch.cuda.set_device(self.rank)
        for layout in ("flat", "expert_major"):
            torch._dynamo.reset()
            switch = TokenSwitchNCCL(
                dist.distributed_c10d._get_default_group(),
                self.world_size,
                16,
                self.world_size * 16,
                128,
            )
            self.addCleanup(switch.close)
            ids = torch.full((16, 1), self.rank, device="cuda", dtype=torch.int32)
            counts = torch.zeros(1, device="cuda", dtype=torch.int32)
            routing = switch.create_routing(ids, counts, layout=layout)
            initial_work_count = torch._C._distributed_c10d._get_work_registry_size()
            # Row-distinct values; the loss below makes every gradient row
            # depend on which token was routed into it.
            rows = _source_rows(self.rank)[:, None]
            tokens = rows.expand(16, 64).to(torch.bfloat16).requires_grad_()
            weights = (rows.float() / 128).requires_grad_()

            def forward(tokens, weights, ids):
                updated = switch.update_routing(routing, ids, counts)
                received, scores, _ = switch.dispatch(
                    updated, tokens, weights, self.world_size * 16
                )
                return switch.combine(updated, received), scores, received

            compiled = torch.compile(forward, fullgraph=True)
            tokens.grad = torch.zeros_like(tokens)
            weights.grad = torch.zeros_like(weights)

            def step():
                tokens.grad.zero_()
                weights.grad.zero_()
                output, scores, received = compiled(tokens, weights, ids)
                token_scale = tokens.detach().float()
                score_scale = received.detach()[:, 0].float().view_as(scores)
                loss = (output.float() * token_scale).sum() + (
                    scores * score_scale
                ).sum()
                loss.backward()
                return output, received

            warmup = torch.cuda.Stream()
            warmup.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup):
                for _ in range(3):
                    step()
            torch.cuda.current_stream().wait_stream(warmup)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output, received = step()
            for phase in range(self.world_size + 2):
                destination = (
                    phase % self.world_size
                    if phase % 2
                    else (self.rank + phase) % self.world_size
                )
                ids.fill_(destination)
                graph.replay()
                self.assertEqual(output, tokens, rtol=0, atol=0)
                self.assertEqual(tokens.grad, tokens, rtol=0, atol=0)
                self.assertEqual(weights.grad, rows.float(), rtol=0, atol=0)
                sources = [
                    source
                    for source in range(self.world_size)
                    if (
                        phase % self.world_size
                        if phase % 2
                        else (source + phase) % self.world_size
                    )
                    == self.rank
                ]
                values = received[:, 0].float()
                values = values[values != 0].sort().values
                empty = torch.empty(0, dtype=torch.int64, device="cuda")
                expected = torch.cat([empty, *map(_source_rows, sources)]).float()
                self.assertEqual(values, expected.sort().values, rtol=0, atol=0)
                # One local expert, so both layouts count every received token.
                self.assertEqual(
                    counts,
                    torch.full_like(counts, 16 * len(sources)),
                    rtol=0,
                    atol=0,
                )
            self.assertEqual(
                torch._C._distributed_c10d._get_work_registry_size(),
                initial_work_count,
                rtol=0,
                atol=0,
            )
            del graph
            gc.collect()
            switch.close()


def _quantized_rows(rank):
    # Exact FP8 mantissas 1..8 and a per-row E8M0 exponent make each
    # (payload byte, scale byte) pair unique per source row.
    row = torch.arange(16, device="cuda")
    payload = (row % 8 + 1).float()[:, None].expand(16, 512).to(torch.float8_e4m3fn)
    scales = (127 + 2 * rank + row // 8).to(torch.uint8)[:, None].expand(16, 16)
    return payload, scales.contiguous()


def _row_keys(payload, scales):
    keys = scales[:, 0].int() * 256 + payload.view(torch.uint8)[:, 0].int()
    return keys[keys != 0].sort().values


@requires_nccl_ep()
class TokenSwitchMXFP8Test(MultiProcContinuousTest):
    hw_classification = HardwareClassification.CUDA

    @classmethod
    def backend_str(cls):
        return "nccl"

    def _switch(self, flow):
        torch.cuda.set_device(self.rank)
        switch = TokenSwitchNCCL(
            dist.distributed_c10d._get_default_group(),
            self.world_size,
            16,
            self.world_size * 16,
            1024,
            ht_flow=flow,
        )
        self.addCleanup(switch.close)
        return switch

    @skip_if_lt_x_gpu(2)
    def test_quantized_fullgraph_changing_routes(self):
        # The explicit SDK flow must override the environment's incompatible flow.
        with patch.dict(os.environ, {"NCCL_EP_HT_PULL_PUSH": "1"}):
            switch = self._switch("push_pull")
        capacity = self.world_size * 16
        ids = torch.full((16, 1), self.rank, device="cuda", dtype=torch.int32)
        counts = torch.zeros(1, device="cuda", dtype=torch.int32)
        routing = switch.create_routing(
            ids, counts, layout="expert_major", alignment=32
        )
        payload, scales = _quantized_rows(self.rank)
        weights = torch.arange(16, device="cuda", dtype=torch.float32).reshape(16, 1)
        weights = weights / 1000 + self.rank / 17
        initial_work_count = torch._C._distributed_c10d._get_work_registry_size()

        def decode(data, scale):
            factors = torch.exp2(scale.float() - 127).repeat_interleave(32, dim=1)
            return (data.float() * factors).bfloat16()

        expected = decode(payload, scales)

        def protocol(payload, scales, weights, ids):
            updated = switch.update_routing(routing, ids, counts)
            received, scores, _, received_scales = switch.dispatch_quantized(
                updated, payload, scales, weights, capacity
            )
            decoded = decode(received, received_scales)
            combined = switch.combine_quantized(updated, decoded)
            grad_data, grad_scales = switch.dispatch_backward_quantized(
                updated, payload, scales, capacity
            )
            token_grads, score_grads = switch.combine_backward_quantized(
                updated, decode(grad_data, grad_scales), scores
            )
            return (
                combined,
                token_grads,
                score_grads,
                decoded,
                received,
                received_scales,
                grad_data,
                grad_scales,
            )

        compiled = torch.compile(protocol, fullgraph=True)
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(3):
                compiled(payload, scales, weights, ids)
        torch.cuda.current_stream().wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = compiled(payload, scales, weights, ids)
        for phase in range(self.world_size + 2):
            destination = (
                phase % self.world_size
                if phase % 2
                else (self.rank + phase) % self.world_size
            )
            ids.fill_(destination)
            graph.replay()
            (
                combined,
                token_grads,
                score_grads,
                decoded,
                received,
                received_scales,
                grad_data,
                grad_scales,
            ) = outputs
            torch.testing.assert_close(combined, expected, rtol=0, atol=0)
            torch.testing.assert_close(token_grads, expected, rtol=0, atol=0)
            torch.testing.assert_close(score_grads, weights, rtol=0, atol=0)
            torch.testing.assert_close(
                received.view(torch.uint8), grad_data.view(torch.uint8), rtol=0, atol=0
            )
            torch.testing.assert_close(received_scales, grad_scales, rtol=0, atol=0)
            sources = [
                source
                for source in range(self.world_size)
                if (
                    phase % self.world_size
                    if phase % 2
                    else (source + phase) % self.world_size
                )
                == self.rank
            ]
            empty = torch.empty(0, dtype=torch.int32, device="cuda")
            source_keys = [_row_keys(*_quantized_rows(source)) for source in sources]
            expected_keys = torch.cat([empty, *source_keys])
            self.assertEqual(
                _row_keys(received, received_scales), expected_keys.sort().values
            )
            self.assertEqual(
                decoded.float().sum(),
                sum(
                    decode(*_quantized_rows(source)).float().sum() for source in sources
                ),
                rtol=0,
                atol=0,
            )
        self.assertEqual(
            torch._C._distributed_c10d._get_work_registry_size(), initial_work_count
        )
        del graph
        gc.collect()

    @skip_if_lt_x_gpu(2)
    def test_pull_push_rejects_mxfp8_combine(self):
        switch = self._switch("pull_push")
        ids = torch.full((16, 1), self.rank, device="cuda", dtype=torch.int64)
        counts = torch.zeros(1, device="cuda", dtype=torch.int32)
        routing = switch.create_routing(
            ids, counts, layout="expert_major", alignment=32
        )
        expert_tokens = torch.zeros(
            self.world_size * 16, 512, device="cuda", dtype=torch.bfloat16
        )
        # The library reports only a generic status; the BF16 combine on the same
        # inputs is the control showing that only the MXFP8 recipe is rejected.
        switch.combine(routing, expert_tokens)
        with self.assertRaisesRegex(RuntimeError, "invalid argument"):
            switch.combine_quantized(routing, expert_tokens)


if __name__ == "__main__":
    run_tests()
