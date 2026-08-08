"""Tests for the CODA backward epilogue-fusion prototype.

Run:
    python coda_backward_epilogue_fusion_test.py
"""

import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import coda_backward_epilogue_fusion as coda
from coda_backward_epilogue_fusion import (
    apply_epilogue_fusion,
    DeferredGradTensor,
    FusibleFunction,
    LOG,
    MMRelu,
    MMTanh,
    RULES,
)

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


# An op with no fusion rule registered for any pair, used to exercise the
# "forgot to register" diagnostics.
class MMSquare(FusibleFunction):
    @staticmethod
    def forward(main_ctx, epilogue_ctx, x, w):
        a = x @ w
        main_ctx.save_for_backward(x, w)
        main_ctx.set_output_meta(a)
        epilogue_ctx.save_for_backward(a)
        return a * a

    @staticmethod
    def main_backward(main_ctx, g):
        x, w = main_ctx.saved_tensors
        gx = g @ w.transpose(-1, -2) if main_ctx.needs_input_grad[0] else None
        gw = x.transpose(-1, -2) @ g if main_ctx.needs_input_grad[1] else None
        return gx, gw

    @staticmethod
    def epilogue_backward(epilogue_ctx, g):
        (a,) = epilogue_ctx.saved_tensors
        return g * 2 * a


# A two-activation-input op, used to exercise the "more than one fusible input"
# guard. Both inputs can be epilogue outputs.
class Add2(FusibleFunction):
    # Only forward is needed: its test raises at planning time (two fusible inputs),
    # before any backward runs.
    @staticmethod
    def forward(main_ctx, epilogue_ctx, a, b):
        main_ctx.set_output_meta(a)
        return a + b


# Same matmul+relu, but the activation is the SECOND input (input 0 is the
# weight). Used to check the fusible edge is found at a non-zero position.
class MMReluRHS(FusibleFunction):
    @staticmethod
    def forward(main_ctx, epilogue_ctx, w, x):
        a = x @ w
        main_ctx.save_for_backward(x, w)
        main_ctx.set_output_meta(a)
        epilogue_ctx.save_for_backward(a)
        return torch.relu(a)

    @staticmethod
    def main_backward(main_ctx, g):  # inputs are (w, x): weight 0, activation 1
        x, w = main_ctx.saved_tensors
        grad_w = x.transpose(-1, -2) @ g if main_ctx.needs_input_grad[0] else None
        grad_x = g @ w.transpose(-1, -2) if main_ctx.needs_input_grad[1] else None
        return grad_w, grad_x

    @staticmethod
    def epilogue_backward(epilogue_ctx, g):
        (a,) = epilogue_ctx.saved_tensors
        return g * (a > 0).to(a.dtype)


# A shape-CHANGING epilogue (GLU): a = x @ w is [.., 2K]; the epilogue gates the
# two halves into [.., K]. The intermediate (a) is wider than the output, so this
# only works because the user declares the intermediate metadata.
class MMGlu(FusibleFunction):
    @staticmethod
    def forward(main_ctx, epilogue_ctx, x, w):
        a = x @ w  # [.., 2K]
        main_ctx.save_for_backward(x, w)
        main_ctx.set_output_meta(a)  # intermediate is [.., 2K], wider than the output
        epilogue_ctx.save_for_backward(a)
        d = a.shape[-1] // 2
        return a[..., :d] * torch.sigmoid(a[..., d:])  # [.., K]

    @staticmethod
    def main_backward(main_ctx, grad_a):
        x, w = main_ctx.saved_tensors
        grad_x = grad_a @ w.transpose(-1, -2) if main_ctx.needs_input_grad[0] else None
        grad_w = x.transpose(-1, -2) @ grad_a if main_ctx.needs_input_grad[1] else None
        return grad_x, grad_w

    @staticmethod
    def epilogue_backward(epilogue_ctx, grad_out):
        (a,) = epilogue_ctx.saved_tensors
        d = a.shape[-1] // 2
        u, g = a[..., :d], a[..., d:]
        sig = torch.sigmoid(g)
        grad_u = grad_out * sig
        grad_g = grad_out * u * sig * (1 - sig)
        return torch.cat([grad_u, grad_g], dim=-1)  # [.., 2K] = grad_a (dim-expanding)


# Eager reference activation for each op type.
ACT = {MMRelu: torch.relu, MMTanh: torch.tanh, MMSquare: lambda t: t * t}

CHAINS = {
    "single_relu": [MMRelu],
    "relu3": [MMRelu, MMRelu, MMRelu],
    "tanh3": [MMTanh, MMTanh, MMTanh],
    "mixed": [MMTanh, MMRelu, MMTanh],
}


def _rule_keys(rules):
    return {(producer, consumer) for producer, consumer, _ in rules}


def _grads(tensors):
    return [t.grad for t in tensors]


def _run_coda(ops, x, ws, do_fuse=True):
    LOG.reset()
    out = x
    for op, w in zip(ops, ws):
        out = op.apply(out, w)
    loss = out.sum()
    plan = (
        apply_epilogue_fusion(loss.grad_fn, RULES, _internal_debug=True)
        if do_fuse
        else None
    )
    loss.backward()
    return plan


def _run_eager(ops, x, ws):
    out = x
    for op, w in zip(ops, ws):
        out = ACT[op](out @ w)
    out.sum().backward()


def _fresh_inputs(ops, device, dtype=torch.double, seed=0):
    torch.manual_seed(seed)
    base_x = torch.randn(4, 6, dtype=dtype, device=device)
    base_ws = [torch.randn(6, 6, dtype=dtype, device=device) for _ in ops]
    return base_x, base_ws


class TestCodaBwdFusionNumerics(TestCase):
    """Fused backward must match plain eager autograd, on any device."""

    @parametrize("chain", list(CHAINS))
    def test_matches_eager(self, device, chain):
        ops = CHAINS[chain]
        base_x, base_ws = _fresh_inputs(ops, device)

        x = base_x.clone().requires_grad_()
        ws = [w.clone().requires_grad_() for w in base_ws]
        _run_coda(ops, x, ws, do_fuse=True)

        ex = base_x.clone().requires_grad_()
        ews = [w.clone().requires_grad_() for w in base_ws]
        _run_eager(ops, ex, ews)

        self.assertEqual(x.grad, ex.grad)
        for w, ew in zip(ws, ews):
            self.assertEqual(w.grad, ew.grad)

    @parametrize("chain", list(CHAINS))
    def test_fusion_changes_no_numerics(self, device, chain):
        """Same op chain, with and without applying the fusion plan, agrees."""
        ops = CHAINS[chain]
        base_x, base_ws = _fresh_inputs(ops, device)

        x = base_x.clone().requires_grad_()
        ws = [w.clone().requires_grad_() for w in base_ws]
        _run_coda(ops, x, ws, do_fuse=True)

        nx = base_x.clone().requires_grad_()
        nws = [w.clone().requires_grad_() for w in base_ws]
        _run_coda(ops, nx, nws, do_fuse=False)

        self.assertEqual(x.grad, nx.grad)
        for w, nw in zip(ws, nws):
            self.assertEqual(w.grad, nw.grad)


class TestCodaBwdFusionPlanning(TestCase):
    """Structural behavior of the fusion planner and kernel-path accounting."""

    def _inputs(self, n):
        torch.manual_seed(0)
        x = torch.randn(4, 6, dtype=torch.double, requires_grad=True)
        ws = [
            torch.randn(6, 6, dtype=torch.double, requires_grad=True) for _ in range(n)
        ]
        return x, ws

    def test_chain_fuses_all_but_tail(self):
        ops = [MMRelu, MMRelu, MMRelu]
        x, ws = self._inputs(len(ops))
        plan = _run_coda(ops, x, ws)
        # ep1<-main2, ep2<-main3 fuse; the tail epilogue (ep3) has no producer main.
        self.assertEqual(len(plan._pairs_fused), 2)
        self.assertEqual(LOG.c["fused_impl"], 2)
        self.assertEqual(LOG.c["epilogue_unfused"], 1)
        self.assertEqual(LOG.c["main_params_only"], 2)
        self.assertEqual(LOG.c["main_full"], 1)  # main1's input is a leaf

    def test_no_duplicate_grad_input_matmul(self):
        """A deferred producer must not also compute grad_input (no wasted matmul)."""
        ops = [MMRelu, MMRelu, MMRelu]
        x, ws = self._inputs(len(ops))
        _run_coda(ops, x, ws)
        # Each deferred producer runs params-only; the grad_input matmul happens
        # exactly once, inside the fused kernel.
        self.assertEqual(LOG.c["main_params_only"], 2)
        self.assertEqual(LOG.c["fused_impl"], 2)

    def test_saved_tensors_partitioned_per_node(self):
        x, ws = self._inputs(1)
        out = MMRelu.apply(x, ws[0])
        epi_node = out.grad_fn  # _EpilogueNode
        main_node = epi_node.next_functions[0][0]  # _MainNode
        self.assertEqual(len(main_node.saved_tensors), 2)  # x, w
        self.assertEqual(len(epi_node.saved_tensors), 1)  # preactivation a

    def test_no_reference_cycle_leak(self):
        import gc
        import weakref

        refs = []

        def run():
            x = torch.randn(4, 6, dtype=torch.double, requires_grad=True)
            ws = [
                torch.randn(6, 6, dtype=torch.double, requires_grad=True)
                for _ in range(3)
            ]
            refs.append(weakref.ref(x))
            refs.extend(weakref.ref(w) for w in ws)
            out = MMRelu.apply(MMRelu.apply(MMRelu.apply(x, ws[0]), ws[1]), ws[2])
            loss = out.sum()
            apply_epilogue_fusion(loss, RULES, expect_num_fusions=2)
            loss.backward()

        # With gc off, surviving tensors mean a reference cycle is pinning them
        # (not refcount-freed), which would accumulate activations across a loop.
        gc.disable()
        try:
            run()
            alive = sum(r() is not None for r in refs)
        finally:
            gc.collect()
            gc.enable()
        self.assertEqual(
            alive,
            0,
            f"{alive} input/weight tensors still alive after backward "
            f"(held by a reference cycle, not freed by refcount)",
        )

    def test_branching_consumer_raises(self):
        # An epilogue output consumed by more than one main node cannot be fused:
        # its backward grad accumulates across the branches, which a single deferral
        # placeholder cannot represent. The planner must reject it loudly.
        torch.manual_seed(1)
        x = torch.randn(4, 6, dtype=torch.double, requires_grad=True)
        w1 = torch.randn(6, 6, dtype=torch.double, requires_grad=True)
        wa = torch.randn(6, 6, dtype=torch.double, requires_grad=True)
        wb = torch.randn(6, 6, dtype=torch.double, requires_grad=True)

        b = MMRelu.apply(x, w1)
        loss = MMRelu.apply(b, wa).sum() + MMRelu.apply(b, wb).sum()
        with self.assertRaisesRegex(RuntimeError, "exactly one consumer"):
            apply_epilogue_fusion(loss, RULES)

    def test_unregistered_pair_not_fused(self):
        self.assertNotIn(
            (MMSquare.main_backward, MMSquare.epilogue_backward), _rule_keys(RULES)
        )

        x, ws = self._inputs(2)
        LOG.reset()
        out = MMSquare.apply(MMSquare.apply(x, ws[0]), ws[1])
        loss = out.sum()
        plan = apply_epilogue_fusion(loss, RULES, _internal_debug=True)
        self.assertEqual(len(plan._pairs_fused), 0)
        # The one main->epilogue adjacency is recorded as a missing-rule candidate.
        self.assertEqual(len(plan._pairs_missing_rules), 1)
        loss.backward()
        self.assertEqual(LOG.c["fused_impl"], 0)

        ex = x.detach().clone().requires_grad_()
        ew0 = ws[0].detach().clone().requires_grad_()
        ew1 = ws[1].detach().clone().requires_grad_()
        a1 = ex @ ew0
        a2 = (a1 * a1) @ ew1
        (a2 * a2).sum().backward()
        self.assertEqual(x.grad, ex.grad)
        self.assertEqual(ws[0].grad, ew0.grad)
        self.assertEqual(ws[1].grad, ew1.grad)

    def test_missing_rule_reported_in_assert(self):
        """A fusible adjacency with no rule is surfaced when the count falls short."""
        x, ws = self._inputs(2)
        # The MMRelu main feeds MMSquare's epilogue, and the pair
        # (MMRelu.main_backward, MMSquare.epilogue_backward) is not in RULES.
        loss = MMRelu.apply(MMSquare.apply(x, ws[0]), ws[1]).sum()
        plan = apply_epilogue_fusion(loss, RULES, _internal_debug=True)
        self.assertEqual(len(plan._pairs_fused), 0)
        self.assertEqual(len(plan._pairs_missing_rules), 1)
        self.assertEqual(
            plan._pairs_missing_rules[0]._label(),
            "MMRelu.main_backward -> MMSquare.epilogue_backward",
        )
        with self.assertRaisesRegex(AssertionError, "forget to pass a rule"):
            plan.assert_num_fusions(1)
        # The pair label appears in the message too.
        with self.assertRaisesRegex(
            AssertionError, r"MMRelu\.main_backward -> MMSquare\.epilogue_backward"
        ):
            plan.assert_num_fusions(1)

    def test_mixed_chain_fuses_both(self):
        """Self-contained ops; the mixed chain fuses both epilogues across types."""
        self.assertIn(
            (MMTanh.main_backward, MMRelu.epilogue_backward), _rule_keys(RULES)
        )
        self.assertIn(
            (MMRelu.main_backward, MMTanh.epilogue_backward), _rule_keys(RULES)
        )

        ops = [MMTanh, MMRelu, MMTanh]
        x, ws = self._inputs(len(ops))
        plan = _run_coda(ops, x, ws)
        self.assertEqual(len(plan._pairs_fused), 2)
        producers = {p.producer.cls for p in plan._pairs_fused}
        self.assertEqual(producers, {MMRelu, MMTanh})  # both op types fuse as producers
        self.assertEqual(LOG.c["fused_impl"], 2)

    def test_assert_num_fusions_contract(self):
        ops = [MMRelu, MMRelu, MMRelu]
        x, ws = self._inputs(len(ops))
        out = x
        for op, w in zip(ops, ws):
            out = op.apply(out, w)
        plan = apply_epilogue_fusion(out.sum(), RULES, _internal_debug=True)
        self.assertIs(plan.assert_num_fusions(2), plan)  # returns self on success
        with self.assertRaises(AssertionError):
            plan.assert_num_fusions(3)

    def test_placeholder_rejects_real_ops(self):
        # The placeholder must never be computed on; any dispatched op errors.
        t = DeferredGradTensor(
            (4, 6),
            torch.double,
            torch.device("cpu"),
            torch.zeros(4, 6, dtype=torch.double),
            (),
        )
        with self.assertRaisesRegex(RuntimeError, "metadata-only placeholder"):
            _ = t + 1
        # The happy-path tests above already assert the engine never dispatches on
        # it (they'd raise here otherwise), so the placeholder truly just rides the
        # edge in normal backward.

    def test_shape_changing_epilogue(self):
        """A dim-changing epilogue (GLU): intermediate [.., 2K] wider than output."""
        torch.manual_seed(0)
        K = 6
        x = torch.randn(4, K, dtype=torch.double, requires_grad=True)
        w = torch.randn(K, 2 * K, dtype=torch.double, requires_grad=True)

        out = MMGlu.apply(x, w)
        self.assertEqual(out.shape, (4, K))  # output is narrower than the intermediate
        out.sum().backward()

        ex = x.detach().clone().requires_grad_()
        ew = w.detach().clone().requires_grad_()
        a = ex @ ew
        d = a.shape[-1] // 2
        (a[..., :d] * torch.sigmoid(a[..., d:])).sum().backward()
        self.assertEqual(x.grad, ex.grad)
        self.assertEqual(w.grad, ew.grad)

    def test_multiple_fusible_inputs_errors(self):
        torch.manual_seed(0)
        x = torch.randn(4, 6, dtype=torch.double, requires_grad=True)
        w1 = torch.randn(6, 6, dtype=torch.double, requires_grad=True)
        w2 = torch.randn(6, 6, dtype=torch.double, requires_grad=True)
        # Add2's two inputs are both epilogue outputs (MMRelu): two fusible edges.
        out = Add2.apply(MMRelu.apply(x, w1), MMRelu.apply(x, w2))
        with self.assertRaisesRegex(RuntimeError, "deferring more than one grad_input"):
            apply_epilogue_fusion(out.sum(), RULES)

    def test_epilogue_input_not_first(self):
        """The fusible activation edge can be at a non-zero input position."""
        torch.manual_seed(0)
        x = torch.randn(4, 6, dtype=torch.double, requires_grad=True)
        w1 = torch.randn(6, 6, dtype=torch.double, requires_grad=True)
        w2 = torch.randn(6, 6, dtype=torch.double, requires_grad=True)
        rules = RULES + [
            (
                MMReluRHS.main_backward,
                MMRelu.epilogue_backward,
                coda.mm_relu_fused_backward,
            ),
        ]

        a = MMRelu.apply(x, w1)
        out = MMReluRHS.apply(w2, a)  # activation `a` is the 2nd input
        loss = out.sum()
        plan = apply_epilogue_fusion(loss, rules, _internal_debug=True)
        self.assertEqual(len(plan._pairs_fused), 1)
        self.assertEqual(plan._pairs_fused[0].producer.defer_input_idx, 1)
        loss.backward()

        ex = x.detach().clone().requires_grad_()
        ew1 = w1.detach().clone().requires_grad_()
        ew2 = w2.detach().clone().requires_grad_()
        torch.relu(torch.relu(ex @ ew1) @ ew2).sum().backward()
        self.assertEqual(x.grad, ex.grad)
        self.assertEqual(w1.grad, ew1.grad)
        self.assertEqual(w2.grad, ew2.grad)

    def test_expect_num_fusions_kwarg(self):
        ops = [MMRelu, MMRelu, MMRelu]
        x, ws = self._inputs(len(ops))

        out = x
        for op, w in zip(ops, ws):
            out = op.apply(out, w)
        plan = apply_epilogue_fusion(
            out.sum(), RULES, expect_num_fusions=2, _internal_debug=True
        )
        self.assertEqual(len(plan._pairs_fused), 2)

        # A wrong expectation raises during apply, on a fresh graph.
        out2 = x
        for op, w in zip(ops, ws):
            out2 = op.apply(out2, w)
        with self.assertRaises(AssertionError):
            apply_epilogue_fusion(out2.sum(), RULES, expect_num_fusions=3)

    def test_works_with_nonreentrant_checkpoint(self):
        """Fusion composes with torch.utils.checkpoint(use_reentrant=False).

        Two AC regions, (M1 E1) and (M2 E2), one op each. Non-reentrant checkpoint
        builds the real autograd graph -- our main/epilogue nodes are visible to the
        planner -- and recomputes saved tensors lazily in each node's backward. The
        planner is purely structural (it never reads saved_tensors), so arming does
        not force a premature recompute; the producer (main2, region 2) still defers
        its activation grad into the consumer (epilogue1, region 1) across the region
        boundary, and the producer's saved set rides the placeholder to the fused
        kernel even though region 2's tensors are recomputed independently.
        """
        from torch.utils.checkpoint import checkpoint

        torch.manual_seed(0)
        x = torch.randn(4, 6, dtype=torch.double, requires_grad=True)
        w1 = torch.randn(6, 6, dtype=torch.double, requires_grad=True)
        w2 = torch.randn(6, 6, dtype=torch.double, requires_grad=True)

        LOG.reset()
        h = checkpoint(MMRelu.apply, x, w1, use_reentrant=False)  # region 1: M1 E1
        out = checkpoint(MMRelu.apply, h, w2, use_reentrant=False)  # region 2: M2 E2
        loss = out.sum()
        plan = apply_epilogue_fusion(loss, RULES, _internal_debug=True)
        self.assertEqual(
            len(plan._pairs_fused), 1
        )  # epilogue1 (region 1) <- main2 (region 2)
        loss.backward()
        self.assertEqual(LOG.c["fused_impl"], 1)
        self.assertEqual(LOG.c["epilogue_unfused"], 1)  # the tail epilogue (region 2)
        self.assertEqual(
            LOG.c["main_params_only"], 1
        )  # main2 deferred its activation grad
        self.assertEqual(LOG.c["main_full"], 1)  # main1's input is a leaf

        ex = x.detach().clone().requires_grad_()
        ew1 = w1.detach().clone().requires_grad_()
        ew2 = w2.detach().clone().requires_grad_()
        torch.relu(torch.relu(ex @ ew1) @ ew2).sum().backward()
        self.assertEqual(x.grad, ex.grad)
        self.assertEqual(w1.grad, ew1.grad)
        self.assertEqual(w2.grad, ew2.grad)


instantiate_parametrized_tests(TestCodaBwdFusionPlanning)
instantiate_device_type_tests(TestCodaBwdFusionNumerics, globals())

if __name__ == "__main__":
    run_tests()
