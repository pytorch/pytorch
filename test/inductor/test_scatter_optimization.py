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
    """
    Tests for the partitioned scatter FX pass
    (torch/_inductor/fx_passes/reduced_atomic_contention.py).

    The pass replaces high-contention index_put(accumulate=True) ops with a
    partitioned scatter-add that distributes writes across num_partitions
    expanded buffers to reduce atomic contention, then sums them back.

    Tests are structured as:
      1. Accuracy  — compiled output matches eager within numerical tolerance
      2. Gate skip — pass correctly bypasses unsupported/low-benefit operations
      3. Counters  — inductor counter bookkeeping reflects pass decisions
      4. Unit      — pure-function math for _compute_num_partitions
    """

    # -----------------------------------------------------------------------
    # Fixtures
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _check_accuracy(self, f, args, *, atol=1e-1, rtol=1e-2, exact=False):
        """
        Run f eagerly and through Inductor, assert the results agree.

        exact=True uses torch.equal (integer accumulation is associative so
        the partitioned result must be bit-for-bit identical to eager).
        """
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
        Defaults to output_shape[dim] // 2 (moderate contention).
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

    # -----------------------------------------------------------------------
    # Accuracy — PR reference example (scaled down for CI)
    # -----------------------------------------------------------------------

    def test_pr_reference_accuracy(self):
        """
        Scaled-down version of the PR benchmark gist (jataylo/dd3a6353...).

        Original dimensions: N=1_000_000, D=100, n_small=501.
        CI dimensions:       N=10_000,   D=10,  n_small=51.

        Three independent scatter-adds into three output buffers with
        worst-case contention (index range = 4 slots, ratio ≈ 19.6×).
        All three ops should be transformed by the pass.
        """
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
        # index range [0, 4) mirrors the highest-contention row in the PR table
        idx0 = torch.randint(0, 4, (N,), dtype=torch.int64)
        idx1 = torch.randint(0, 4, (N,), dtype=torch.int64)
        idx2 = torch.randint(0, 4, (N,), dtype=torch.int64)

        self._check_accuracy(
            f, (out0, out1, out2, idx0, idx1, idx2, vals), atol=1.0, rtol=1e-2
        )
        self.assertGreaterEqual(counters["inductor"]["partitioned_scatter_applied"], 3)

    def test_accuracy_int32_exact(self):
        """
        Integer scatter-add must produce an exact match to eager.

        Integer addition is associative, so the partitioned reordering of
        writes is mathematically exact: result must be bit-for-bit identical.
        """
        torch.manual_seed(4)
        N = 8192

        def f(out, idx, vals):
            return out.index_put([idx], vals, accumulate=True)

        out, idx, vals = self._make_scatter_inputs(N, (8,), torch.int32, index_high=8)
        self._check_accuracy(f, (out, idx, vals), exact=True)

    # -----------------------------------------------------------------------
    # Accuracy — in-place variant (index_put_)
    # -----------------------------------------------------------------------

    def test_accuracy_inplace_index_put(self):
        """
        in-place index_put_ with accumulate=True is also pattern-matched.
        The pass replaces it with the same partitioned algorithm.
        """
        torch.manual_seed(7)
        N, n, D = 8192, 8, 4  # ratio = 8192/(8*4) = 256

        def f(out, idx, vals):
            out.index_put_([idx], vals, accumulate=True)
            return out

        out = torch.zeros(n, D, dtype=torch.float32)
        idx = torch.randint(0, 4, (N,), dtype=torch.int64)
        vals = torch.randn(N, D, dtype=torch.float32)

        self._check_accuracy(f, (out, idx, vals), atol=1.0, rtol=1e-2)

    # -----------------------------------------------------------------------
    # Skip gate tests — pass correctly bypasses ineligible ops
    # -----------------------------------------------------------------------

    def test_skip_accumulate_false(self):
        """
        index_put with accumulate=False does not match the registered patterns
        (patterns hard-code True in the 4th position) so the pass never fires.

        We use a permutation index so there are no duplicate writes and the
        eager result is deterministic (accumulate=False with duplicates would
        be non-deterministic and untestable for exact equality).
        """
        n = 256  # small enough to use a full permutation

        def f(out, idx, vals):
            # accumulate=False with a permutation index: each slot written exactly once
            return out.index_put([idx], vals, accumulate=False)

        torch.manual_seed(10)
        out = torch.zeros(n, dtype=torch.float32)
        idx = torch.randperm(n, dtype=torch.int64)
        vals = torch.randn(n, dtype=torch.float32)

        # Accuracy: pass skips so compiled should match eager exactly
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
        partitioned_scatter_force=True bypasses gates 6 (min_index_size) and 7
        (min_contention_ratio) and the diminishing-returns cap on num_partitions,
        while still enforcing the hard memory-budget and correctness gates.

        Uses two ops that the normal heuristics would skip:
          Op A — index_size=100 < min_index_size=4096  (gate 6 skip)
          Op B — index_size=4096, output_size=16384,
                 contention_ratio=0.25 < 1.0            (gate 7 skip)

        Both should be applied under force mode, and the output must still be
        numerically correct.
        """
        def f(out_a, idx_a, vals_a, out_b, idx_b, vals_b):
            out_a = out_a.index_put([idx_a], vals_a, accumulate=True)
            out_b = out_b.index_put([idx_b], vals_b, accumulate=True)
            return out_a, out_b

        torch.manual_seed(15)
        # Op A: tiny index (would normally be skipped by gate 6)
        out_a  = torch.zeros(10, dtype=torch.float32)
        idx_a  = torch.randint(0, 5, (100,), dtype=torch.int64)
        vals_a = torch.randn(100, dtype=torch.float32)

        # Op B: low contention ratio (would normally be skipped by gate 7)
        out_b  = torch.zeros(16384, dtype=torch.float32)
        idx_b  = torch.randint(0, 16384, (4096,), dtype=torch.int64)
        vals_b = torch.randn(4096, dtype=torch.float32)

        args = (out_a, idx_a, vals_a, out_b, idx_b, vals_b)

        # Without force: both ops skipped
        with torch.no_grad():
            expected = f(*args)
            normal = torch.compile(f, backend="inductor", fullgraph=True)(*args)
        self.assertEqual(counters["inductor"]["partitioned_scatter_applied"], 0)
        self.assertGreater(
            counters["inductor"]["partitioned_scatter_skipped_index_too_small"]
            + counters["inductor"]["partitioned_scatter_skipped_low_contention"],
            0,
        )

        # With force: both ops applied
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
            counters["inductor"]["partitioned_scatter_applied"], 2,
            "force mode must apply to both ops regardless of heuristic gates",
        )
        for e, a in zip(expected, forced):
            self.assertTrue(
                torch.allclose(e, a, atol=1e-1, rtol=1e-2),
                f"force mode output mismatch: expected={e[:5]} actual={a[:5]}",
            )

    # -----------------------------------------------------------------------
    # Unit tests — _compute_num_partitions pure function
    # -----------------------------------------------------------------------

    def test_compute_num_partitions_tight_budget(self):
        """Exactly enough memory for 4 partitions of a 1 MB buffer."""
        # overhead = output_size * element_bytes * (P-1)
        # 4 partitions: overhead = 1M * 4 * 3 = 12 MB
        # 8 partitions: overhead = 1M * 4 * 7 = 28 MB → won't fit in 20 MB
        output_size = 1_000_000
        element_bytes = 4
        available = 20_000_000  # 20 MB → fits P=4 (12 MB) but not P=8 (28 MB)
        result = _compute_num_partitions(
            available, output_size, element_bytes, min_p=2, max_p=128
        )
        self.assertEqual(result, 4)

    def test_compute_num_partitions_diminishing_returns_cap(self):
        """
        With ample memory, the contention cap limits P so the sum-reduction
        kernel never dominates the scatter kernel.

        Cap formula: P <= 4 * (index_size / scatter_dim_size), floored to power-of-2.

        index_size=1024, scatter_dim_size=16 → writes_per_slot=64
        → cap = 4 * 64 = 256 → floor(log2(256)) = 8 → 2^8 = 256
        But max_p=128, so result = min(256, 128) = 128. Memory is ample (not the bottleneck).

        index_size=64, scatter_dim_size=16 → writes_per_slot=4
        → cap = 4 * 4 = 16 → floor(log2(16)) = 4 → 2^4 = 16
        Even with ample memory, P is capped at 16.
        """
        available = 10**12  # effectively unlimited

        # writes_per_slot = 1024/16 = 64 → cap = 256, floored to 256, then min(256, 128) = 128
        result = _compute_num_partitions(
            available, 1024, 4, min_p=2, max_p=128,
            index_size=1024, scatter_dim_size=16,
        )
        self.assertEqual(result, 128)

        # writes_per_slot = 64/16 = 4 → cap = 16; memory is not the limit
        result = _compute_num_partitions(
            available, 1024, 4, min_p=2, max_p=128,
            index_size=64, scatter_dim_size=16,
        )
        self.assertEqual(result, 16)

        # writes_per_slot = 8/16 = 0.5 → cap = max(2, 2^int(log2(2))) = 2
        result = _compute_num_partitions(
            available, 1024, 4, min_p=2, max_p=128,
            index_size=8, scatter_dim_size=16,
        )
        self.assertEqual(result, 2)

    def test_skip_low_contention_ratio_multidim(self):
        """
        For a multidimensional output [n, D], the contention gate uses
        index_size / scatter_dim_size (n), NOT index_size / output_numel (n*D).

        Classic failing case: embedding weight gradient [vocab, dim] with D >> 1.
          vocab=4096, dim=64, N=5000:
            Old (wrong): 5000 / (4096*64) = 0.019 < 1.0 → incorrectly skip
            New (correct): 5000 / 4096 = 1.22 >= 1.0 → apply

        Parameters chosen so that:
          index_size=5000 >= min_index_size=4096  (gate 6 passes)
          OLD ratio = N/output_numel = 5000/262144 = 0.019 < 1.0  (old gate skips)
          NEW ratio = N/scatter_dim  = 5000/4096  = 1.22  ≥ 1.0  (new gate applies)
        """
        vocab, dim, N = 4096, 64, 5000   # ratio_old=0.019 < 1.0, ratio_new=1.22 ≥ 1.0

        def f(out, idx, vals):
            return out.index_put([idx], vals, accumulate=True)

        out  = torch.zeros(vocab, dim, dtype=torch.float32)
        idx  = torch.randint(0, vocab, (N,), dtype=torch.int64)
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
        # If the gate is correct the pass applied; if it still uses numel the counter stays 0.
        self.assertGreater(
            counters["inductor"]["partitioned_scatter_applied"], 0,
            "pass skipped for [vocab, dim] output — contention gate likely still uses "
            "output_numel instead of scatter_dim_size",
        )


    @unittest.skipUnless(HAS_GPU, "requires GPU for CUDA-aware memory tracking")
    def test_memory_aware_partition_count(self):
        """
        Verifies that live tensor memory in the FX graph constrains num_partitions,
        and that setting floor=total_gpu causes the pass to skip entirely.

        Graph (572 MB of actual GPU allocations)
        -----------------------------------------
        placeholder: out        ( 10 M float32 =  40 MB)
        placeholder: idx        ( 11 M int64   =  88 MB)
        placeholder: vals       ( 11 M float32 =  44 MB)
        placeholder: persistent (100 M float32 = 400 MB)  ← large, live throughout

        index_put(out, [idx], vals) → scattered   ← persistent still live: 440 MB baseline
        sum(persistent)             → psum         ← persistent freed here
        add(scattered, psum)        → output

        MemoryTracker baseline at index_put (after scheduling):
            initial live = 40+88+44+400 = 572 MB
            +40 MB (index_put output), -(40+88+44) MB (out/idx/vals freed)
            = 440 MB

        Sub-test 1 — real memory pressure, reasonable floor (allowed_peak = 600 MB):
            available = 600 - 440 = 160 MB
            P=4 overhead = 10M * 4 * 3 = 120 MB ≤ 160 MB → fits
            P=8 overhead = 10M * 4 * 7 = 280 MB > 160 MB → doesn't fit
            → num_partitions = 4, applied > 0

        Sub-test 2 — floor = total_gpu → allowed_peak = 0 → available < 0 → pass skips:
            Setting non_model_floor_bytes = total_gpu means "reserve all GPU memory
            as off-limits floor", leaving zero headroom for expanded scatter buffers.
            _compute_num_partitions returns 0 < min_p=2 → skip entirely.
        """
        torch.manual_seed(12)
        N = 11_000_000  # index size: ratio=1.1 >= 1.0, size >= 4096
        output_size = 10_000_000  # 10 M float32 = 40 MB
        persist_n = 100_000_000  # 100 M float32 = 400 MB

        def f(out, idx, vals, persistent):
            scattered = out.index_put([idx], vals, accumulate=True)
            # persistent's last use is here, after index_put:
            # MemoryTracker counts it as live (440 MB) at the index_put node.
            return scattered + persistent.sum()

        out = torch.zeros(output_size, dtype=torch.float32, device=GPU_TYPE)
        idx = torch.randint(0, output_size, (N,), dtype=torch.int64, device=GPU_TYPE)
        vals = torch.randn(N, dtype=torch.float32, device=GPU_TYPE)
        persistent = torch.randn(persist_n, dtype=torch.float32, device=GPU_TYPE)

        with torch.no_grad():
            expected = f(out, idx, vals, persistent)

        _, total_gpu = torch.cuda.mem_get_info()

        # ---- Sub-test 1: real memory pressure → applied > 0 ---------------
        # floor = total_gpu - 600 MB simulates a GPU with 600 MB of usable headroom.
        # baseline ≈ 440 MB → available ≈ 160 MB → num_partitions = 4.
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
            counters["inductor"]["partitioned_scatter_applied"], 0,
            "real memory pressure: pass should apply (available ≈ 160 MB fits P=4)",
        )
        self.assertTrue(
            torch.allclose(expected, actual, atol=1e-2, rtol=1e-2),
            f"real memory pressure: output mismatch "
            f"expected[:5]={expected[:5]}\nactual[:5]={actual[:5]}",
        )

        # ---- Sub-test 2: floor = total_gpu → allowed_peak = 0 → skip ------
        # Setting non_model_floor_bytes = total_gpu leaves zero headroom for
        # expanded scatter buffers, so _compute_num_partitions returns 0 < min_p=2.
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
            counters["inductor"]["partitioned_scatter_applied"], 0,
            "floor=total_gpu: no headroom, pass must skip",
        )
        self.assertGreater(
            counters["inductor"]["partitioned_scatter_skipped_memory_budget"], 0,
        )
        self.assertTrue(
            torch.allclose(expected, actual_skipped, atol=1e-2, rtol=1e-2),
            "floor=total_gpu: output should still be correct (pass gracefully skips)",
        )



    # -----------------------------------------------------------------------
    # Performance — requires DO_PERF_TEST=1
    # -----------------------------------------------------------------------

    @unittest.skipUnless(HAS_GPU, "requires GPU")
    def test_perf_atomic_contention(self):
        """
        Reproduces the PR #168073 benchmark.

        Three independent index_put(accumulate=True) ops into three float32[n,D]
        buffers with N=1_000_000 scatter operations.  The index range is kept
        deliberately small so many threads write to the same output slots,
        creating heavy atomic contention — exactly as the original PR gist:
          https://gist.github.com/jataylo/dd3a6353ad2859efd65fa87b28aa3ebd

        PR benchmark results (N=1M, D=100, n=501, float32):
          uniform_range | MI300 no-pass | MI300 partitioned | speedup
          0-3           |  8.31 ms      |  2.20 ms          | 3.78×
          0-7           |  4.32 ms      |  1.63 ms          | 2.66×
          0-15          |  4.24 ms      |  1.60 ms          | 2.66×

        The single-slot case (0-0) is excluded here.  In that pathological case
        the *baseline* scatter is effectively cache-resident (all N writes hit
        the same tiny D-element row that fits in L1), while the partitioned
        version touches a much larger expanded buffer (P×n×D elements strided
        across memory), making cache pressure dominate over atomic savings.
        The win is strongest at 4–16 slots, which is the regime the PR targets.

        The test compiles the function twice — once with the pass disabled
        (baseline) and once with it enabled — benchmarks both with
        benchmarker.benchmark_gpu (which handles its own warmup), then asserts
        the partitioned version is at least MIN_SPEEDUP faster.  A threshold
        of 2.0× is intentionally conservative: the real gain on MI300/H100 at
        4-slot contention is 3–4×, so 2.0× gives headroom for variance while
        still catching regressions.
        """
        torch.manual_seed(42)
        N, D, n = 1_000_000, 100, 501
        # Conservative floor: actual MI300X speedup is 3-4× at 4 slots.
        # 2.0× still catches regressions without being fragile on different hardware.
        MIN_SPEEDUP = 2.0

        def scatter_fn(out0, out1, out2, idx, vals):
            """Three independent scatter-adds — the exact PR benchmark workload."""
            out0 = out0.index_put([idx], vals, accumulate=True)
            out1 = out1.index_put([idx], vals, accumulate=True)
            out2 = out2.index_put([idx], vals, accumulate=True)
            return out0, out1, out2

        # Contention levels from the PR table.  index_slots = index_high + 1.
        # All ops share the same idx tensor so contention is maximally uniform.
        contention_cases = [
            # (label,      index_high, min_speedup)
            ("high",     3,  MIN_SPEEDUP),   # 4 slots  — PR table 3.78× on MI300
            ("moderate", 7,  MIN_SPEEDUP),   # 8 slots  — PR table 2.66× on MI300
            ("low-med",  15, MIN_SPEEDUP),   # 16 slots — PR table 2.66× on MI300
        ]

        print(
            f"\n{'contention':<12} {'slots':<8} "
            f"{'baseline_ms':<14} {'partitioned_ms':<16} {'speedup':<10} {'applied'}"
        )
        print("-" * 68)

        for label, index_high, min_spd in contention_cases:
            # Build inputs once; reuse for both compiled variants.
            vals = torch.randn(N, D, dtype=torch.float32, device=GPU_TYPE)
            out0 = torch.zeros(n, D, dtype=torch.float32, device=GPU_TYPE)
            out1 = torch.zeros(n, D, dtype=torch.float32, device=GPU_TYPE)
            out2 = torch.zeros(n, D, dtype=torch.float32, device=GPU_TYPE)
            # All three ops share one index tensor: every write to a slot is
            # a true atomic conflict between thread blocks on all three ops.
            idx = torch.randint(
                0, index_high + 1, (N,), dtype=torch.int64, device=GPU_TYPE
            )
            inputs = (out0, out1, out2, idx, vals)

            # ---- baseline: pass disabled ----------------------------------------
            config.partitioned_scatter_enabled = False
            torch._dynamo.reset()
            baseline_fn = torch.compile(
                scatter_fn, backend="inductor", fullgraph=True
            )
            with torch.no_grad():
                for _ in range(3):
                    baseline_fn(*inputs)
            baseline_ms = benchmarker.benchmark_gpu(
                lambda: baseline_fn(*inputs)  # noqa: B023
            )

            # ---- partitioned: pass enabled --------------------------------------
            config.partitioned_scatter_enabled = True
            torch._dynamo.reset()
            counters.clear()
            partitioned_fn = torch.compile(
                scatter_fn, backend="inductor", fullgraph=True
            )
            with torch.no_grad():
                for _ in range(3):
                    partitioned_fn(*inputs)
            partitioned_ms = benchmarker.benchmark_gpu(
                lambda: partitioned_fn(*inputs)  # noqa: B023
            )

            speedup = baseline_ms / partitioned_ms
            n_applied = counters["inductor"]["partitioned_scatter_applied"]

            print(
                f"{label:<12} {index_high + 1:<8} "
                f"{baseline_ms:<14.3f} {partitioned_ms:<16.3f} "
                f"{speedup:<10.2f}x {n_applied}"
            )

            self.assertGreater(
                n_applied, 0,
                f"contention={label}: pass did not apply — check gates/config",
            )
            self.assertGreater(
                speedup, min_spd,
                f"contention={label}: expected ≥{min_spd:.1f}× speedup, "
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
