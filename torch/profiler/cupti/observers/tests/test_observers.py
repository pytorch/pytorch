# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for torch.profiler.cupti observers (adapted from the original
CuptiMonitor tests onto the mux interface).

The mux's libcupti version (and HES timing) depend on load order, so these
run their bodies in subprocesses that load the v2 libcupti before any CUDA
context -- the same approach the original HES test used.
"""

import subprocess
import sys
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


def _run(code: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestCuptiObservers(TestCase):
    def test_enable_hes_early_guard(self):
        # Before any CUDA context, enable_hes_early() must succeed.
        ok = _run(
            "import torch\n"
            "from torch.profiler import cupti\n"
            "cupti.enable_hes_early()\n"
            "print('ok')\n"
        )
        self.assertEqual(ok.returncode, 0, ok.stderr)

        # After a CUDA context exists, it must raise with a clear message.
        late = _run(
            "import torch\n"
            "from torch.profiler import cupti\n"
            "torch.randn(1, device='cuda')\n"
            "cupti.enable_hes_early()\n"
        )
        self.assertNotEqual(late.returncode, 0)
        self.assertIn(
            "enable_hes_early() must be called before CUDA context creation",
            late.stderr,
        )

    def test_node_timer_observer(self):
        # Load the v2 libcupti before torch so the mux is available, then
        # time kernels through the observer (default: CONCURRENT_KERNEL only).
        p = _run(
            "from cupti import cupti as _c\n"
            "_s = _c.subscribe(lambda *a: None, 0); _c.unsubscribe(_s)\n"
            "import torch\n"
            "from torch.profiler.cupti import instance\n"
            "from torch.profiler.cupti.observers import NodeTimerObserver\n"
            "if not instance().available:\n"
            "    print('MUX_UNAVAILABLE'); raise SystemExit(0)\n"
            "obs = NodeTimerObserver()\n"
            "assert obs.available\n"
            "x = torch.zeros(1, device='cuda')\n"
            "for _ in range(32):\n"
            "    x.add_(1.0)\n"
            "torch.cuda.synchronize()\n"
            "instance().poll(force=True)\n"
            "totals = obs.drain(); obs.close()\n"
            "assert sum(c for _, c in totals.values()) > 0, totals\n"
            "print('OK')\n"
        )
        if "MUX_UNAVAILABLE" in p.stdout:
            self.skipTest("CUPTI mux unavailable (needs CUPTI >= 13.2)")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("OK", p.stdout)

    def test_node_timer_observer_multi_kind(self):
        # Opt into MEMCPY alongside kernels; both are timed. The eager copy
        # lands in the node-0 bucket, so total count covers kernels + copy.
        p = _run(
            "from cupti import cupti as _c\n"
            "_s = _c.subscribe(lambda *a: None, 0); _c.unsubscribe(_s)\n"
            "import torch\n"
            "from torch.profiler.cupti import instance, types\n"
            "from torch.profiler.cupti.observers import NodeTimerObserver\n"
            "if not instance().available:\n"
            "    print('MUX_UNAVAILABLE'); raise SystemExit(0)\n"
            "k = types.ActivityKind\n"
            "obs = NodeTimerObserver([k.CONCURRENT_KERNEL, k.MEMCPY])\n"
            "assert obs.available\n"
            "x = torch.zeros(1, device='cuda')\n"
            "for _ in range(16):\n"
            "    x.add_(1.0)\n"
            "_ = x.cpu()\n"
            "torch.cuda.synchronize()\n"
            "instance().poll(force=True)\n"
            "totals = obs.drain(); obs.close()\n"
            "assert sum(c for _, c in totals.values()) > 0, totals\n"
            "print('OK')\n"
        )
        if "MUX_UNAVAILABLE" in p.stdout:
            self.skipTest("CUPTI mux unavailable (needs CUPTI >= 13.2)")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("OK", p.stdout)

    def test_layout_era_transition(self):
        # A kernel-only timer (vectorizable, single-kind era) must keep timing
        # correctly across layout changes: registering a second observer that
        # adds MEMCPY moves the mux into a multi-kind era, and closing it
        # returns to the vectorizable era. The era-boundary force-flush should
        # mean no buffer straddles eras and the timer's totals stay sane.
        p = _run(
            "from cupti import cupti as _c\n"
            "_s = _c.subscribe(lambda *a: None, 0); _c.unsubscribe(_s)\n"
            "import torch\n"
            "from torch.profiler.cupti import instance, types\n"
            "from torch.profiler.cupti.observers import NodeTimerObserver\n"
            "if not instance().available:\n"
            "    print('MUX_UNAVAILABLE'); raise SystemExit(0)\n"
            "k = types.ActivityKind\n"
            "x = torch.zeros(1, device='cuda')\n"
            "timer = NodeTimerObserver()\n"  # single-kind / vectorizable era
            "assert timer.available\n"
            "for _ in range(8):\n"
            "    x.add_(1.0)\n"
            "torch.cuda.synchronize()\n"
            "multi = NodeTimerObserver([k.CONCURRENT_KERNEL, k.MEMCPY])\n"  # -> multi-kind era
            "for _ in range(8):\n"
            "    x.add_(1.0)\n"
            "_ = x.cpu()\n"
            "torch.cuda.synchronize()\n"
            "multi.close()\n"  # -> back to vectorizable era
            "for _ in range(8):\n"
            "    x.add_(1.0)\n"
            "torch.cuda.synchronize()\n"
            "totals = timer.drain(); timer.close()\n"
            # All durations must be non-negative finite ns (a misparse from a
            # straddling buffer would surface as garbage durations).
            "assert sum(c for _, c in totals.values()) >= 24, totals\n"
            "assert all(d >= 0 for d, _ in totals.values()), totals\n"
            "print('OK')\n"
        )
        if "MUX_UNAVAILABLE" in p.stdout:
            self.skipTest("CUPTI mux unavailable (needs CUPTI >= 13.2)")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("OK", p.stdout)

    def test_profiler_observer_records_kinds(self):
        # ProfilerObserver should record kernel + memcpy columns.
        p = _run(
            "from cupti import cupti as _c\n"
            "_s = _c.subscribe(lambda *a: None, 0); _c.unsubscribe(_s)\n"
            "import torch\n"
            "from torch.profiler.cupti import instance, ProfilerSession, types\n"
            "if not instance().available:\n"
            "    print('MUX_UNAVAILABLE'); raise SystemExit(0)\n"
            "sess = ProfilerSession()\n"
            "assert sess.start()\n"
            "x = torch.zeros(1, device='cuda')\n"
            "for _ in range(16):\n"
            "    x.add_(1.0)\n"
            "_ = x.cpu()\n"
            "torch.cuda.synchronize()\n"
            "data = sess.stop()\n"
            "k = types.ActivityKind\n"
            "assert k.CONCURRENT_KERNEL in data, list(data)\n"
            "assert k.MEMCPY in data, list(data)\n"
            "assert int(data[k.MEMCPY][types.MemcpyField.BYTES][0]) == 4\n"
            "print('OK')\n"
        )
        if "MUX_UNAVAILABLE" in p.stdout:
            self.skipTest("CUPTI mux unavailable (needs CUPTI >= 13.2)")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("OK", p.stdout)


class MuxPaddingTest(TestCase):
    """Pure unit tests for the discovery-driven uniform-padding planner
    (no GPU/CUPTI)."""

    def setUp(self):
        from torch.profiler.cupti.mux import _plan_padding, _subset_sum_indices

        self._plan = _plan_padding
        self._ss = _subset_sum_indices

    def test_subset_sum(self):
        self.assertEqual(self._ss([], 0), [])
        self.assertIsNone(self._ss([], 4))
        idx = self._ss([8, 8, 4], 16)
        self.assertIsNotNone(idx)
        self.assertEqual(sum([8, 8, 4][i] for i in idx), 16)
        self.assertIsNone(self._ss([8, 8], 4))  # only 8s, can't make 4

    def test_plan_pads_when_widths_known(self):
        # base via KIND(0)+selected widths, 8-aligned: kind1 = align(4+8+8)=24,
        # kind2 = align(4+8+8+8)=32 -> gap 8; candidate 90 (8B) closes it.
        union = {1: frozenset({1, 2}), 2: frozenset({1, 2, 3})}
        fsize = {1: {0: 4, 1: 8, 2: 8, 90: 8, 91: 4}, 2: {0: 4, 1: 8, 2: 8, 3: 8}}
        cands = {1: (90, 91), 2: ()}
        plan = self._plan(union, fsize, cands)
        self.assertIn(90, plan[1])  # 8B filler closes the 8B gap
        self.assertTrue({1, 2}.issubset(plan[1]))
        self.assertEqual(set(plan[2]), {1, 2, 3})  # max kind unchanged

    def test_plan_widens_for_discovery_when_widths_unknown(self):
        # Same gap, but candidate widths not yet discovered -> return the
        # discovery selection (union + all candidates) to learn them.
        union = {1: frozenset({1, 2}), 2: frozenset({1, 2, 3})}
        fsize = {1: {0: 4, 1: 8, 2: 8}, 2: {0: 4, 1: 8, 2: 8, 3: 8}}
        cands = {1: (90, 91), 2: ()}
        plan = self._plan(union, fsize, cands)
        self.assertTrue({90, 91}.issubset(plan[1]))  # widened for discovery

    def test_plan_mixed_widths_subset_sum(self):
        # gap 8 closed by two 4B fillers when no 8B filler is available.
        # kind1 base = align(4+8)=16, kind2 base = align(4+8+8)=24 -> gap 8.
        union = {1: frozenset({1}), 2: frozenset({1, 2})}
        fsize = {1: {0: 4, 1: 8, 92: 4, 93: 4}, 2: {0: 4, 1: 8, 2: 8}}
        cands = {1: (92, 93), 2: ()}
        plan = self._plan(union, fsize, cands)
        self.assertEqual({92, 93}, set(plan[1]) - {1})  # 4+4 == 8 gap

    def test_plan_noop_when_uniform_or_unknown_or_single(self):
        # Already uniform.
        self.assertEqual(
            self._plan(
                {1: frozenset({1}), 2: frozenset({1})},
                {1: {0: 4, 1: 8}, 2: {0: 4, 1: 8}},
                {1: (90,), 2: (90,)},
            ),
            {1: frozenset({1}), 2: frozenset({1})},
        )
        # Single kind.
        u = {1: frozenset({1})}
        self.assertEqual(self._plan(u, {1: {0: 4, 1: 8}}, {1: (90,)}), u)
        # Widths not discovered yet -> unchanged.
        u2 = {1: frozenset({1}), 2: frozenset({1, 2})}
        self.assertEqual(self._plan(u2, {}, {1: (90,), 2: ()}), u2)


if __name__ == "__main__":
    run_tests()
