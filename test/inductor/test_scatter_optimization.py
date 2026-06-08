# Owner(s): ["module: inductor"]

import copy
import os
import unittest

import torch
from torch import nn
from torch._dynamo.utils import counters, same
from torch._inductor import config, metrics
from torch._inductor.fx_passes.reduced_atomic_contention import _compute_num_partitions
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.test_case import TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


# set so that metrics appear
torch._logging.set_logs(inductor_metrics=True)

DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"


class TestScatterOpt(TestCase):
    def setUp(self):
        super().setUp()
        metrics.reset()
        counters.clear()

    def check_metric(self, val=1):
        self.assertEqual(val, metrics.num_matches_for_scatter_upon_const_tensor)

    def do_acc_test(self, f, *args):
        expect = f(*args)
        actual = torch.compile(f)(*args)
        self.assertTrue(same(expect, actual, tol=1e-3), f"{expect=}\n{actual=}\n")

    def test_3d_tensor(self):
        L, M, N = 2, 1024, 2048

        def f(x):
            y = torch.full([L, M, N], 3.14, dtype=torch.float)
            y.scatter_(2, x.unsqueeze(2), 2.718)
            return y

        x = torch.randint(0, N, (L, M), dtype=torch.int64)
        self.do_acc_test(f, x)
        expected_num_bytes = (
            L * M * N * torch.float.itemsize + L * M * torch.int64.itemsize
        )
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes)

    def test_non_last_dim(self):
        """
        Test the case that the scatter dimension is not the last one.
        """
        M, N = 1024, 2048

        def f(x):
            y = torch.full([M, N], 3.14, dtype=torch.float)
            y.scatter_(0, x.unsqueeze(0), 2.718)
            return y

        x = torch.randint(0, M, (N,), dtype=torch.int64)
        self.do_acc_test(f, x)
        expected_num_bytes = M * N * torch.float.itemsize + N * torch.int64.itemsize
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes)

    def test_neg_scatter_dim(self):
        M, N = 1024, 2048

        def f(x):
            y = torch.full([M, N], 3.14, dtype=torch.float)
            y.scatter_(-1, x.unsqueeze(1), 2.718)
            return y

        x = torch.randint(0, N, (M,), dtype=torch.int64)
        self.do_acc_test(f, x)
        expected_num_bytes = M * N * torch.float.itemsize + M * torch.int64.itemsize
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes)

    def test_shorter_index_tensor(self):
        M, N = 1024, 2048

        def f(x):
            y = torch.full([M, N], 3.14, dtype=torch.float)
            y.scatter_(1, x.unsqueeze(1), 2.718)
            return y

        x = torch.randint(0, N, (M // 2,), dtype=torch.int64)
        self.do_acc_test(f, x)

        # no match since the index tensor is shorter. May support it in future.
        self.assertEqual(0, counters["inductor"]["pattern_matcher_count"])

    def test_nonzero_const_tensor(self):
        M, N = 1024, 2048

        def f(x):
            y = torch.full([M, N], 3.14, dtype=torch.float)
            y.scatter_(1, x.unsqueeze(1), 2.718)
            return y

        x = torch.randint(0, N, (M,), dtype=torch.int64)
        self.do_acc_test(f, x)
        expected_num_bytes = M * N * torch.float.itemsize + M * torch.int64.itemsize
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes)

    def test_can_not_optimize_due_to_dense(self):
        M, N = 1024, 2048

        def f(x):
            y = torch.full([M, N], 0, dtype=torch.float)
            y.scatter_(1, x, 0.618)
            return y

        x = torch.randint(0, N, (M, N // 2), dtype=torch.int64)
        self.do_acc_test(f, x)
        expected_num_bytes = M * N * torch.float.itemsize + M * (N // 2) * (
            torch.int64.itemsize + torch.float.itemsize
        )
        # Use assertGreaterEqual rather than assertEqual due to the issue related
        # to StarDep mentioned here: https://github.com/pytorch/pytorch/pull/129043#discussion_r1651699706
        self.assertGreaterEqual(metrics.num_bytes_accessed, expected_num_bytes)

    def test_can_not_optimize_due_to_non_const(self):
        M, N = 1024, 2048

        def f(x, y):
            y.scatter_(1, x, 0.618)
            return y

        x = torch.randint(0, N, (M, 1), dtype=torch.int64)
        y = torch.randn([M, N])
        self.do_acc_test(f, x, y)

        # The generated code is quite in-efficient.
        # There are 3 kernels
        # 1. copy from arg to buf
        # 2. scatter upon buf
        # 3. copy buf back to arg
        # Link to the wrapper: https://gist.github.com/shunting314/d43b74e680b3e5b514f7c28160c39f40
        expected_num_bytes = 4 * M * N * torch.float.itemsize + M * (
            torch.int64.itemsize + torch.float.itemsize
        )
        self.assertGreaterEqual(metrics.num_bytes_accessed, expected_num_bytes)

        # the second kernel and third kernel are both mutation kernel. So we
        # overestimated the memory accessed
        # Update the test once the overestimiation is fixed.
        over_estimate = M * torch.float.itemsize + M * N * torch.float.itemsize
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes + over_estimate)

    def test_cross_entropy_loss(self):
        """
        Match full+scatter in CEL and replaces it with a pointwise.

        Perf data on an A100 GPU:
        Without the scatter optimization:
          ms=47.340, peak_mem=10.524 GB
        With the scatter optimization:
          ms=42.768, peak_mem=7.227 GB
        """
        B, T, D, V = 32, 1024, 768, 50257
        if not DO_PERF_TEST:
            # use a smaller V if not doing perf test to avoid OOM
            # in CI
            V = V // 100
        ref_model = nn.Linear(D, V).to(torch.bfloat16)
        opt_model = copy.deepcopy(ref_model)
        ce = nn.CrossEntropyLoss()

        def f(m, x, label):
            ce(m(x).view(-1, V), label.view(-1)).backward()

        opt_f = torch.compile(f)

        x = torch.randn(B, T, D).to(torch.bfloat16)
        label = torch.randint(0, V, (B, T)).to(torch.int64)

        f(ref_model, x, label)
        ref_grad = ref_model.weight.grad
        opt_f(opt_model, x, label)
        act_grad = opt_model.weight.grad
        if not torch.allclose(ref_grad, act_grad, atol=1e-3, rtol=1e-3):
            raise AssertionError(f"{ref_grad=}\n{act_grad=}")

        self.check_metric()

        if DO_PERF_TEST:
            if GPU_TYPE == "xpu":
                raise unittest.SkipTest(
                    "torch.xpu.reset_peak_memory_stats not implemented."
                )
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                opt_f(opt_model, x, label)
            ms = benchmarker.benchmark_gpu(lambda: opt_f(opt_model, x, label))
            peak_mem = torch.cuda.max_memory_allocated() / 10**9
            print(f"{ms=:.3f}, {peak_mem=:.3f} GB")


class TestPartitionedScatterOpt(TestCase):
    """Tests for the partitioned scatter FX pass (reduced_atomic_contention.py)."""

    def setUp(self):
        super().setUp()
        metrics.reset()
        counters.clear()
        torch._dynamo.reset()
        self._saved_enabled = config.partitioned_scatter_enabled
        self._saved_force = config.partitioned_scatter_force
        config.partitioned_scatter_enabled = True

    def tearDown(self):
        config.partitioned_scatter_enabled = self._saved_enabled
        config.partitioned_scatter_force = self._saved_force
        super().tearDown()

    def _check_accuracy(self, f, args, *, atol=1e-1, rtol=1e-2, exact=False):
        """Run f eagerly and through Inductor, assert the results agree."""
        with torch.no_grad():
            expected = f(*args)
            compiled_f = torch.compile(f, backend="inductor", fullgraph=True)
            actual = compiled_f(*args)

        if isinstance(expected, (list, tuple)):
            outputs = list(zip(expected, actual))
        else:
            outputs = [(expected, actual)]
        for e, a in outputs:
            if exact:
                self.assertTrue(torch.equal(e, a), f"expected={e}\nactual={a}")
            else:
                self.assertTrue(
                    torch.allclose(e.float(), a.float(), atol=atol, rtol=rtol),
                    f"expected={e}\nactual={a}",
                )

    def _make_scatter_inputs(
        self, N, output_shape, dtype=torch.float32, index_high=None, dim=0
    ):
        """
        Create (out, idx, vals) for a 1-index scatter-add along `dim`.
        index_high controls contention: smaller = more writes per slot.
        """
        if index_high is None:
            index_high = output_shape[dim] // 2 or 1
        idx = torch.randint(0, index_high, (N,), dtype=torch.int64)
        val_shape = list(output_shape)
        val_shape[dim] = N
        if dtype in (torch.int32,):
            vals = torch.randint(0, 4, val_shape, dtype=dtype)
        else:
            vals = torch.randn(val_shape, dtype=dtype)
        out = torch.zeros(output_shape, dtype=dtype)
        return out, idx, vals

    def test_pr_reference_accuracy(self):
        """Scaled-down PR benchmark: three scatter-adds with high contention (4 slots)."""
        torch.manual_seed(42)
        N, D, n_small = 10_000, 10, 51

        def f(out0, out1, out2, idx0, idx1, idx2, vals):
            out0 = out0.index_put([idx0], vals, accumulate=True)
            out1 = out1.index_put([idx1], vals, accumulate=True)
            out2 = out2.index_put([idx2], vals, accumulate=True)
            return out0, out1, out2

        vals = torch.randn(N, D, dtype=torch.float32)
        out0 = torch.zeros(n_small, D, dtype=torch.float32)
        out1 = torch.zeros(n_small, D, dtype=torch.float32)
        out2 = torch.zeros(n_small, D, dtype=torch.float32)
        idx0 = torch.randint(0, 4, (N,), dtype=torch.int64)
        idx1 = torch.randint(0, 4, (N,), dtype=torch.int64)
        idx2 = torch.randint(0, 4, (N,), dtype=torch.int64)

        self._check_accuracy(
            f, (out0, out1, out2, idx0, idx1, idx2, vals), atol=1.0, rtol=1e-2
        )
        self.assertGreaterEqual(counters["inductor"]["partitioned_scatter_applied"], 3)

    def test_accuracy_int32_exact(self):
        """Integer scatter-add must be bit-for-bit identical to eager (addition is associative)."""
        torch.manual_seed(4)
        N = 8192

        def f(out, idx, vals):
            return out.index_put([idx], vals, accumulate=True)

        out, idx, vals = self._make_scatter_inputs(N, (8,), torch.int32, index_high=8)
        self._check_accuracy(f, (out, idx, vals), exact=True)

    def test_accuracy_inplace_index_put(self):
        """index_put_ (in-place) is also matched and produces correct output."""
        torch.manual_seed(7)
        N, n, D = 8192, 8, 4

        def f(out, idx, vals):
            out.index_put_([idx], vals, accumulate=True)
            return out

        out = torch.zeros(n, D, dtype=torch.float32)
        idx = torch.randint(0, 4, (N,), dtype=torch.int64)
        vals = torch.randn(N, D, dtype=torch.float32)

        self._check_accuracy(f, (out, idx, vals), atol=1.0, rtol=1e-2)

    def test_skip_accumulate_false(self):
        """index_put with accumulate=False doesn't match the registered patterns."""
        n = 256

        def f(out, idx, vals):
            return out.index_put([idx], vals, accumulate=False)

        torch.manual_seed(10)
        out = torch.zeros(n, dtype=torch.float32)
        idx = torch.randperm(n, dtype=torch.int64)
        vals = torch.randn(n, dtype=torch.float32)

        self._check_accuracy(f, (out, idx, vals), exact=True)
        self.assertEqual(counters["inductor"]["partitioned_scatter_applied"], 0)

    def test_pass_disabled(self):
        """When partitioned_scatter_enabled=False the pass is a no-op."""
        config.partitioned_scatter_enabled = False

        N, n = 8192, 8

        def f(out, idx, vals):
            return out.index_put([idx], vals, accumulate=True)

        out = torch.zeros(n, dtype=torch.float32)
        idx = torch.randint(0, 4, (N,), dtype=torch.int64)
        vals = torch.randn(N, dtype=torch.float32)

        with torch.no_grad():
            expected = f(out, idx, vals)
            actual = torch.compile(f, backend="inductor", fullgraph=True)(
                out, idx, vals
            )
        self.assertTrue(same(expected, actual, tol=1e-3))
        self.assertEqual(counters["inductor"]["partitioned_scatter_applied"], 0)

    def test_force_mode_bypasses_heuristic_gates(self):
        """
        partitioned_scatter_force=True bypasses the min_index_size and
        min_contention_ratio gates while still enforcing correctness constraints.
        """

        def f(out_a, idx_a, vals_a, out_b, idx_b, vals_b):
            out_a = out_a.index_put([idx_a], vals_a, accumulate=True)
            out_b = out_b.index_put([idx_b], vals_b, accumulate=True)
            return out_a, out_b

        torch.manual_seed(15)
        # Op A: index_size=100 < min_index_size → normally skipped by gate 6
        out_a = torch.zeros(10, dtype=torch.float32)
        idx_a = torch.randint(0, 5, (100,), dtype=torch.int64)
        vals_a = torch.randn(100, dtype=torch.float32)

        # Op B: contention_ratio=0.25 < 1.0 → normally skipped by gate 7
        out_b = torch.zeros(16384, dtype=torch.float32)
        idx_b = torch.randint(0, 16384, (4096,), dtype=torch.int64)
        vals_b = torch.randn(4096, dtype=torch.float32)

        args = (out_a, idx_a, vals_a, out_b, idx_b, vals_b)

        with torch.no_grad():
            expected = f(*args)
            torch.compile(f, backend="inductor", fullgraph=True)(*args)
        self.assertEqual(counters["inductor"]["partitioned_scatter_applied"], 0)
        self.assertGreater(
            counters["inductor"]["partitioned_scatter_skipped_index_too_small"]
            + counters["inductor"]["partitioned_scatter_skipped_low_contention"],
            0,
        )

        counters.clear()
        torch._dynamo.reset()
        saved_force = config.partitioned_scatter_force
        try:
            config.partitioned_scatter_force = True
            with torch.no_grad():
                forced = torch.compile(f, backend="inductor", fullgraph=True)(*args)
        finally:
            config.partitioned_scatter_force = saved_force

        self.assertEqual(
            counters["inductor"]["partitioned_scatter_applied"],
            2,
            "force mode must apply to both ops regardless of heuristic gates",
        )
        for e, a in zip(expected, forced):
            self.assertTrue(
                torch.allclose(e, a, atol=1e-1, rtol=1e-2),
                f"force mode output mismatch: expected={e[:5]} actual={a[:5]}",
            )

    def test_compute_num_partitions_tight_budget(self):
        """Memory constraint picks P=4: P=8 overhead (28 MB) exceeds 20 MB budget."""
        # P=4: overhead = 1M * 4 * 3 = 12 MB ≤ 20 MB
        # P=8: overhead = 1M * 4 * 7 = 28 MB > 20 MB
        result = _compute_num_partitions(20_000_000, 1_000_000, 4, min_p=2, max_p=128)
        self.assertEqual(result, 4)

    def test_compute_num_partitions_diminishing_returns_cap(self):
        """Diminishing-returns cap limits P when memory is not the bottleneck."""
        available = 10**12  # effectively unlimited

        # writes_per_slot=64 → cap=256, min(256, max_p=128) = 128
        result = _compute_num_partitions(
            available,
            1024,
            4,
            min_p=2,
            max_p=128,
            index_size=1024,
            scatter_dim_size=16,
        )
        self.assertEqual(result, 128)

        # writes_per_slot=4 → cap=16
        result = _compute_num_partitions(
            available,
            1024,
            4,
            min_p=2,
            max_p=128,
            index_size=64,
            scatter_dim_size=16,
        )
        self.assertEqual(result, 16)

        # writes_per_slot=0.5 → cap=max(2, 2)=2
        result = _compute_num_partitions(
            available,
            1024,
            4,
            min_p=2,
            max_p=128,
            index_size=8,
            scatter_dim_size=16,
        )
        self.assertEqual(result, 2)

    def test_skip_low_contention_ratio_multidim(self):
        """
        Contention gate uses index_size / scatter_dim_size, not index_size / output_numel.

        For [vocab=4096, dim=64] with N=5000 indices:
          wrong: 5000 / (4096*64) = 0.019 → skip
          right: 5000 / 4096 = 1.22 → apply
        """
        vocab, dim, N = 4096, 64, 5000

        def f(out, idx, vals):
            return out.index_put([idx], vals, accumulate=True)

        out = torch.zeros(vocab, dim, dtype=torch.float32)
        idx = torch.randint(0, vocab, (N,), dtype=torch.int64)
        vals = torch.randn(N, dim, dtype=torch.float32)

        with torch.no_grad():
            expected = f(out, idx, vals)
            actual = torch.compile(f, backend="inductor", fullgraph=True)(
                out.clone(), idx, vals
            )

        self.assertTrue(
            torch.allclose(expected, actual, atol=1e-1, rtol=1e-2),
            f"multidim scatter mismatch: {expected[:3]} vs {actual[:3]}",
        )
        self.assertGreater(
            counters["inductor"]["partitioned_scatter_applied"],
            0,
            "pass skipped for [vocab, dim] output — contention gate likely still uses "
            "output_numel instead of scatter_dim_size",
        )

    @unittest.skipUnless(HAS_GPU, "requires GPU for CUDA-aware memory tracking")
    def test_memory_aware_partition_count(self):
        """
        Verify that live tensor memory constrains num_partitions.

        Graph: out(40 MB) + idx(88 MB) + vals(44 MB) + persistent(400 MB) inputs.
        At the index_put node, baseline live memory ≈ 440 MB.

        Sub-test 1: floor leaves 600 MB of headroom → available ≈ 160 MB → P=4 applied.
        Sub-test 2: floor = total_gpu → no headroom → pass skips.
        """
        torch.manual_seed(12)
        N = 11_000_000
        output_size = 10_000_000
        persist_n = 100_000_000

        def f(out, idx, vals, persistent):
            scattered = out.index_put([idx], vals, accumulate=True)
            return scattered + persistent.sum()

        out = torch.zeros(output_size, dtype=torch.float32, device=GPU_TYPE)
        idx = torch.randint(0, output_size, (N,), dtype=torch.int64, device=GPU_TYPE)
        vals = torch.randn(N, dtype=torch.float32, device=GPU_TYPE)
        persistent = torch.randn(persist_n, dtype=torch.float32, device=GPU_TYPE)

        with torch.no_grad():
            expected = f(out, idx, vals, persistent)

        _, total_gpu = torch.cuda.mem_get_info()

        saved_floor = config.partitioned_scatter_non_model_floor_bytes
        try:
            config.partitioned_scatter_non_model_floor_bytes = total_gpu - 600_000_000
            with torch.no_grad():
                actual = torch.compile(f, backend="inductor", fullgraph=True)(
                    out, idx, vals, persistent
                )
        finally:
            config.partitioned_scatter_non_model_floor_bytes = saved_floor

        self.assertGreater(
            counters["inductor"]["partitioned_scatter_applied"],
            0,
            "real memory pressure: pass should apply (available ≈ 160 MB fits P=4)",
        )
        self.assertTrue(
            torch.allclose(expected, actual, atol=1e-2, rtol=1e-2),
            f"real memory pressure: output mismatch "
            f"expected[:5]={expected[:5]}\nactual[:5]={actual[:5]}",
        )

        counters.clear()
        torch._dynamo.reset()

        saved_floor = config.partitioned_scatter_non_model_floor_bytes
        try:
            config.partitioned_scatter_non_model_floor_bytes = total_gpu
            with torch.no_grad():
                actual_skipped = torch.compile(f, backend="inductor", fullgraph=True)(
                    out, idx, vals, persistent
                )
        finally:
            config.partitioned_scatter_non_model_floor_bytes = saved_floor

        self.assertEqual(
            counters["inductor"]["partitioned_scatter_applied"],
            0,
            "floor=total_gpu: no headroom, pass must skip",
        )
        self.assertGreater(
            counters["inductor"]["partitioned_scatter_skipped_memory_budget"],
            0,
        )
        self.assertTrue(
            torch.allclose(expected, actual_skipped, atol=1e-2, rtol=1e-2),
            "floor=total_gpu: output should still be correct (pass gracefully skips)",
        )

    @unittest.skipUnless(HAS_GPU, "requires GPU")
    def test_perf_atomic_contention(self):
        """
        Reproduce the PR benchmark: three index_put(accumulate=True) ops with
        moderate contention (8 slots). CI on ROCm MI300X delivers ≈1.8×, so the
        1.5× floor catches regressions with enough slack for hardware variance.
        """
        torch.manual_seed(42)
        N, D, n = 1_000_000, 100, 501
        MIN_SPEEDUP = 1.5

        def scatter_fn(out0, out1, out2, idx, vals):
            out0 = out0.index_put([idx], vals, accumulate=True)
            out1 = out1.index_put([idx], vals, accumulate=True)
            out2 = out2.index_put([idx], vals, accumulate=True)
            return out0, out1, out2

        vals = torch.randn(N, D, dtype=torch.float32, device=GPU_TYPE)
        out0 = torch.zeros(n, D, dtype=torch.float32, device=GPU_TYPE)
        out1 = torch.zeros(n, D, dtype=torch.float32, device=GPU_TYPE)
        out2 = torch.zeros(n, D, dtype=torch.float32, device=GPU_TYPE)
        idx = torch.randint(0, 8, (N,), dtype=torch.int64, device=GPU_TYPE)
        inputs = (out0, out1, out2, idx, vals)

        config.partitioned_scatter_enabled = False
        torch._dynamo.reset()
        baseline_fn = torch.compile(scatter_fn, backend="inductor", fullgraph=True)
        with torch.no_grad():
            for _ in range(3):
                baseline_fn(*inputs)
        baseline_ms = benchmarker.benchmark_gpu(lambda: baseline_fn(*inputs))

        config.partitioned_scatter_enabled = True
        torch._dynamo.reset()
        counters.clear()
        partitioned_fn = torch.compile(scatter_fn, backend="inductor", fullgraph=True)
        with torch.no_grad():
            for _ in range(3):
                partitioned_fn(*inputs)
        partitioned_ms = benchmarker.benchmark_gpu(lambda: partitioned_fn(*inputs))

        speedup = baseline_ms / partitioned_ms
        n_applied = counters["inductor"]["partitioned_scatter_applied"]

        print(
            f"\nbaseline={baseline_ms:.3f} ms  "
            f"partitioned={partitioned_ms:.3f} ms  "
            f"speedup={speedup:.2f}×  applied={n_applied}"
        )

        self.assertGreater(
            n_applied,
            0,
            "pass did not apply to any op — check gates/config",
        )
        self.assertGreater(
            speedup,
            MIN_SPEEDUP,
            f"expected ≥{MIN_SPEEDUP:.1f}× speedup (moderate contention, 8 slots), "
            f"got {speedup:.2f}×  "
            f"(baseline={baseline_ms:.3f} ms, partitioned={partitioned_ms:.3f} ms)",
        )

        config.partitioned_scatter_enabled = True


if HAS_GPU:
    torch.set_default_device(GPU_TYPE)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests()
