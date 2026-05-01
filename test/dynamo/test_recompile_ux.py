# Owner(s): ["module: dynamo"]
import unittest
import weakref
from functools import cache

import torch
import torch._dynamo
import torch._dynamo.config
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._logging
from torch._dynamo.eval_frame import (
    _get_cache_entries_for_region,
    _get_total_cache_entry_count,
)
from torch._dynamo.exc import FailOnRecompileLimitHit
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.logging_utils import kwargs_to_settings, log_settings


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)


class RecompileUxTests(torch._dynamo.test_case.TestCase):
    # TODO(whc) dynamo actually recompiles one more time than the cache limit
    cache_limit = 1

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack.enter_context(
            torch._dynamo.config.patch("recompile_limit", cls.cache_limit)
        )

    def test_drop_cache_on_skip(self):
        def model(x, i):
            return x + i

        attached = False
        triggered = False

        def trigger():
            nonlocal triggered
            triggered = True

        def compiler(gm, input):
            nonlocal attached
            f = gm.forward
            if attached:
                raise AssertionError("Expected not attached")
            # NB: making this a weakref.ref causes the cycle to no
            # longer be promptly GC'ed
            weakref.finalize(f, trigger)
            attached = True
            return f

        x = torch.randn(2)
        for i in range(2):
            opt_model = torch.compile(model, backend=compiler)
            opt_model(x, i)

        self.assertTrue(triggered)

    def test_loop_torture(self):
        def loop_torture(input, iters):
            out = input
            # randint itself causes one graph break
            for _ in range(iters):
                out += input
            return out

        compile_counter = torch._dynamo.testing.CompileCounter()
        for _ in range(10):
            x = torch.randn(3)
            iters = torch.randint(low=0, high=1000, size=())
            opt_loop_torture = torch.compile(loop_torture, backend=compile_counter)
            opt_loop_torture(x, iters)

        # Currently, we recompile each time,
        # We'd probably like to bail out quickly and warn
        # TODO(whc) these checks fail on py37.  Why?
        # self.assertEqual(counters["frames"]["total"], 2 + self.cache_limit)
        # self.assertEqual(counters["frames"]["ok"], 1 + self.cache_limit)

        # compile_counter only sees frames that were fed to the backend compiler,
        # which is a subset of counters["frames"]["ok"] -- probably because
        # counters["frames"]["ok"] includes frames not containing torch ops?
        self.assertEqual(compile_counter.frame_count, self.cache_limit)

    @torch._dynamo.config.patch("automatic_dynamic_shapes", False)
    def test_dynamic_input(self):
        def model(input):
            return input + input

        expected_recompiles = 2
        compile_counter = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch("recompile_limit", expected_recompiles):
            with self.assertLogs(logger="torch._dynamo", level="WARNING") as logs:
                for _ in range(10):
                    bsz = torch.randint(low=0, high=1000, size=())
                    x = torch.randn((bsz, 3, 4))
                    opt_model = torch.compile(model, backend=compile_counter)
                    opt_model(x)

        self.assertEqual(compile_counter.frame_count, expected_recompiles)
        self.assertEqual(len(logs.records), 1)
        print(logs.records[0])
        self.assertTrue(
            logs.records[0]
            .getMessage()
            .startswith("torch._dynamo hit config.recompile_limit")
        )

    @unittest.skipIf(
        not torch.cuda.is_available() and not torch.xpu.is_available(),
        "requires cuda or xpu",
    )
    def test_nvfuser_guards(self):
        # we may want to model dynamo's guards sufficiently after nvfuser's ProfilingExecutor guards
        # such that we ensure dynamo is in charge of all the recompilations at the top level,
        # and we could thus simplify the underlying torchscript executor
        def func(a, b, c):
            return a + b * c

        a = torch.rand(3, 4, 5, device=device_type)
        b = torch.rand(3, 4, 5, device=device_type)
        b_v = torch.rand(3, 5, 4, device=device_type).view(3, 4, 5)
        b_p = torch.rand(3, 5, 4, device=device_type).permute(0, 2, 1)
        c = torch.rand(3, 4, 5, device=device_type)
        compile_counter = torch._dynamo.testing.CompileCounter()

        with torch._dynamo.config.patch("recompile_limit", 2):
            opt_func = torch.compile(func, backend=compile_counter)
            opt_func(a, b, c)  # warmup
            self.assertEqual(compile_counter.frame_count, 1)

            opt_func(a, b, c)  # no guard fail or recompile
            self.assertEqual(compile_counter.frame_count, 1)

            opt_func(a, b_v, c)  # a view should not cause nvfuser recompile
            self.assertEqual(compile_counter.frame_count, 1)

            opt_func(a, b_p, c)  # a permutation should cause recompile
            self.assertEqual(compile_counter.frame_count, 2)

    def assert_single_log_contains(self, logs, contains_str):
        self.assertEqual(len(logs.records), 1)
        self.assertTrue(
            logs.records[0].getMessage().find(contains_str) > 0,
            msg=f'Expected to find "{contains_str}" in log "{logs.records[0].getMessage()}"',
        )

    def test_verbose_tensor_check(self):
        def func(a):
            # Warning: choose a function here whose meta implementation lives
            # entirely in C++.  If you do a Python one, Dynamo will dive into
            # torch._refs which is OK but it will muddy up the warnings
            return torch.add(a, 4)

        def cache_fail_test(cached_input, missed_input, expected_failure):
            # TODO(whc) maybe its hacky to have a 'test within a test' but this seemed convenient
            torch._dynamo.reset()
            torch._dynamo.utils.counters.clear()
            opt_func = torch.compile(func, backend="eager")
            # warmup
            opt_func(cached_input)

            with self.assertLogs(logger="torch._dynamo", level="WARNING") as logs:
                opt_func = torch.compile(func, backend="eager")
                opt_func(missed_input)
            self.assert_single_log_contains(logs, expected_failure)

        a = torch.rand(3, 4, 5)
        cache_fail_test(
            a,
            a[0:2, :, :],
            "tensor 'a' size mismatch at index 0. expected 3, actual 2",
        )
        cache_fail_test(
            a,
            a.clone().as_strided((3, 4, 5), stride=(1, 3, 12)),
            "tensor 'a' stride mismatch at index 0. expected 20, actual 1",
        )
        cache_fail_test(a, a[0, :, :], "tensor 'a' rank mismatch. expected 3, actual 2")
        cache_fail_test(a, a.to("meta"), "tensor 'a' dispatch key set mismatch.")
        cache_fail_test(
            a,
            a.to(torch.float16),
            "tensor 'a' dtype mismatch. expected Float, actual Half",
        )
        a_grad = a.clone()
        a_grad.requires_grad = True
        cache_fail_test(
            a,
            a_grad,
            "tensor 'a' requires_grad mismatch. expected requires_grad=0",
        )

    def test_mismatched_type(self):
        a = torch.rand(3, 4, 5)
        b = torch.rand(3, 4, 5)

        def func(a, b):
            return a + b

        opt_func = torch.compile(func, backend="eager")
        # warmup
        opt_func(a, b)

        with self.assertLogs(logger="torch._dynamo", level="WARNING") as logs:
            opt_func = torch.compile(func, backend="eager")
            opt_func(a, 1)
        self.assert_single_log_contains(
            logs,
            "expected type of 'b' to be a tensor type, ' but found <class 'int'>",
        )

    @torch._dynamo.config.patch(recompile_limit=1, fail_on_recompile_limit_hit=True)
    def test_fail_on_recompile_limit_hit(self):
        @torch.compile(backend="eager")
        def func(b, a):
            if a:
                return b * 2
            else:
                return b + 1

        func(torch.randn(5), True)
        with self.assertRaises(FailOnRecompileLimitHit):
            func(torch.randn(5), False)

    @torch._dynamo.config.patch("recompile_limit", 32)
    def test_multiple_guard_fails(self):
        failure_reasons = []

        def guard_fail_fn(failure):
            failure_reasons.append(failure[0])

        def f(x):
            return torch.relu(x)

        opt_f = torch._dynamo.optimize(
            backend="eager", guard_fail_fn=guard_fail_fn, dynamic=False
        )(f)

        for i in range(5):
            failure_reasons.clear()
            opt_f(torch.randn(8 + i))

        failure_str = "\n".join(failure_reasons)
        for line in [
            "tensor 'x' size mismatch at index 0. expected 11, actual 12",
            "tensor 'x' size mismatch at index 0. expected 10, actual 12",
            "tensor 'x' size mismatch at index 0. expected 9, actual 12",
            "tensor 'x' size mismatch at index 0. expected 8, actual 12",
        ]:
            self.assertIn(
                line,
                failure_str,
            )

    @torch._dynamo.config.patch("recompile_limit", 32)
    def test_multiple_guard_fails_report_all(self):
        with log_settings(kwargs_to_settings(recompiles_verbose=True)):
            failure_reasons = []

            def guard_fail_fn(failure):
                failure_reasons.append(failure[0])

            def f(x):
                return torch.ones(len(x), x[-1])

            opt_f = torch._dynamo.optimize(
                backend="eager", guard_fail_fn=guard_fail_fn, dynamic=False
            )(f)

            opt_f([4, 5, 6])

            def filter_reasons():
                return "\n".join(
                    [
                        line
                        for line in "\n".join(failure_reasons).splitlines()
                        if not line.startswith("___check_type_id")
                    ]
                )

            failure_reasons.clear()
            opt_f([7, 8])

            for line in ["len(x) == 3"]:
                self.assertIn(line, filter_reasons())

            failure_reasons.clear()
            opt_f([9])

            for line in ["len(x) == 2", "len(x) == 3"]:
                self.assertIn(line, filter_reasons())

    @torch._dynamo.config.patch(recompile_limit=1)
    def test_recompile_child_run_only(self):
        def f(x, n):
            if torch.compiler.is_compiling():
                x = x + 1
            x = g(x)
            return h(x) + n

        def g(x):
            if torch.compiler.is_compiling():
                return x + 2
            return x

        def h(x):
            if torch.compiler.is_compiling():
                return x + 4
            return x

        torch.compile(g, backend="eager")(torch.randn(3))
        inp = torch.randn(3)
        opt_f = torch.compile(f, backend="eager")
        opt_f(inp, 0)

        # expect f to run eager, g compiled (from previous invocatino), h eager
        res = opt_f(inp, 1)

        self.assertEqual(res, inp + 3)


class RecompileLimitKwargTests(torch._dynamo.test_case.TestCase):
    @staticmethod
    def _num_cache_entries(code):
        return len(torch._dynamo.eval_frame._debug_get_cache_entry_list(code))

    def test_recompile_limit_basic(self):
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x, y):
            return x + y

        opt_f = torch.compile(f, backend=cnt, recompile_limit=2)

        opt_f(torch.randn(3), torch.randn(3))
        self.assertEqual(self._num_cache_entries(f), 1)

        opt_f(torch.randn(3, dtype=torch.float64), torch.randn(3, dtype=torch.float64))
        self.assertEqual(self._num_cache_entries(f), 2)

        # Third dtype should NOT trigger recompilation (recompile_limit=2)
        opt_f(torch.randn(3, dtype=torch.float16), torch.randn(3, dtype=torch.float16))
        self.assertEqual(self._num_cache_entries(f), 2)

    def test_recompile_limit_none_uses_global(self):
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x, y):
            return x + y

        # Without recompile_limit kwarg, uses global config (default 8)
        opt_f = torch.compile(f, backend=cnt)

        for i in range(10):
            dtype = [
                torch.float32,
                torch.float64,
                torch.float16,
                torch.bfloat16,
                torch.int32,
                torch.int64,
                torch.int16,
                torch.int8,
                torch.uint8,
                torch.complex64,
            ][i]
            opt_f(torch.ones(3, dtype=dtype), torch.ones(3, dtype=dtype))

        self.assertEqual(
            self._num_cache_entries(f), torch._dynamo.config.recompile_limit
        )

    def test_recompile_limit_fullgraph_raises(self):
        """With fullgraph=True, hitting the recompile_limit kwarg raises
        FailOnRecompileLimitHit, consistent with the fullgraph contract."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_f = torch.compile(f, backend=cnt, fullgraph=True, recompile_limit=1)

        opt_f(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

        with self.assertRaises(FailOnRecompileLimitHit):
            opt_f(torch.randn(3, dtype=torch.float64))

    @torch._dynamo.config.patch(automatic_dynamic_shapes=True)
    def test_recompile_limit_resume_function_auto_dynamic(self):
        """With automatic dynamic shapes and recompile_limit=2, the resume
        function recompiles via dimension changes on a global tensor while
        the main function gets cache hits. The resume function should stop
        at 2 entries and fall back to eager."""
        cnt = torch._dynamo.testing.CompileCounter()

        y_holder = {"tensor": torch.randn(4, 8, 2)}

        def f(x):
            x.sin()
            print("graph break")
            return y_holder["tensor"].cos()

        opt_f = torch.compile(f, backend=cnt, recompile_limit=2)

        # Call 1: static compile
        y_holder["tensor"] = torch.randn(4, 8, 2)
        opt_f(torch.randn(4, 8, 2))

        # Call 2: y dim0 changes -> f cache hit, resume recompiles
        y_holder["tensor"] = torch.randn(5, 8, 2)
        opt_f(torch.randn(4, 8, 2))
        frame_count_after_2 = cnt.frame_count

        # Call 3: y dim1 changes -> resume should NOT recompile
        # (resume already has 2 entries = recompile_limit)
        y_holder["tensor"] = torch.randn(5, 9, 2)
        opt_f(torch.randn(4, 8, 2))
        self.assertEqual(cnt.frame_count, frame_count_after_2)

        # Verify f has 1 entry, resume has 2
        num_f_entries = len(torch._dynamo.eval_frame._debug_get_cache_entry_list(f))
        self.assertEqual(num_f_entries, 1)

        from torch._dynamo.resume_execution import ContinueExecutionCache

        resume_codes = list(ContinueExecutionCache.cache[f.__code__].values())
        self.assertTrue(len(resume_codes) > 0, "No resume functions found")
        for resume_code in resume_codes:
            num_resume_entries = len(
                torch._dynamo.eval_frame._debug_get_cache_entry_list(resume_code)
            )
            self.assertEqual(num_resume_entries, 2)


class IsolateRecompilesTests(torch._dynamo.test_case.TestCase):
    """Tests for isolate_recompiles=True on torch.compile().

    Each torch.compile() call with isolate_recompiles=True gets its own
    isolated cache bucket via the per-compile cache map in ExtraState.
    Without isolation, all compile calls on the same code object share a
    single cache — entries from one call interfere with another's lookup,
    recompile limit, and FrameExecStrategy.
    """

    @staticmethod
    def _num_cache_entries(code):
        return len(torch._dynamo.eval_frame._debug_get_cache_entry_list(code))

    # ===== Basic isolation: independent caches per compile call =====

    @torch._dynamo.config.patch(
        recompile_limit=1,
        fail_on_recompile_limit_hit=True,
        automatic_dynamic_shapes=False,
    )
    def test_isolate_recompiles_basic(self):
        """A single isolated region respects its per-region recompile limit."""

        def f(x):
            return x.sin()

        opt_f = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )

        opt_f(torch.randn(3))

        with self.assertRaises(FailOnRecompileLimitHit):
            opt_f(torch.randn(4))

    @torch._dynamo.config.patch(
        recompile_limit=1,
        fail_on_recompile_limit_hit=True,
        automatic_dynamic_shapes=False,
    )
    def test_isolate_recompiles_same_function_different_regions(self):
        """Two compile calls on the same function get independent caches.
        Each can compile once without the other's entry causing a limit hit."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_a = torch.compile(f, backend=cnt, dynamic=False, isolate_recompiles=True)
        opt_b = torch.compile(f, backend=cnt, dynamic=False, isolate_recompiles=True)

        opt_a(torch.randn(3))
        opt_b(torch.randn(4))

        self.assertEqual(cnt.frame_count, 2)

    @torch._dynamo.config.patch(
        recompile_limit=1,
        fail_on_recompile_limit_hit=True,
        automatic_dynamic_shapes=False,
    )
    def test_isolate_recompiles_factory_pattern(self):
        """Factory creates multiple torch.compile wrappers around the same
        inner function. Each gets its own isolated cache bucket."""

        def core(x):
            return x.sum()

        @cache
        def factory(key):
            @torch.compile(fullgraph=True, dynamic=False, isolate_recompiles=True)
            def frontend(x, n):
                return core(x) + n

            return frontend

        factory("foo")(torch.ones(3), 3)
        factory("bar")(torch.ones(4), 3)
        factory("baz")(torch.ones(5), 3)

    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    def test_isolate_recompiles_same_backend_different_regions(self):
        """Two isolated regions sharing the SAME CompileCounter backend.
        Without per-region bucketing, the second region would get a cache
        hit from the first (same backend, same guards). Verifies the
        per-region cache map routes entries to the correct bucket."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_a = torch.compile(f, backend=cnt, dynamic=False, isolate_recompiles=True)
        opt_b = torch.compile(f, backend=cnt, dynamic=False, isolate_recompiles=True)

        opt_a(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

        # Must compile again — different region, even though same backend + input
        opt_b(torch.randn(3))
        self.assertEqual(cnt.frame_count, 2)

        # Cache hits within each region
        opt_a(torch.randn(3))
        opt_b(torch.randn(3))
        self.assertEqual(cnt.frame_count, 2)

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_isolate_recompiles_string_backends(self, backend):
        """Two isolated regions with the same string backend compile
        independently — verified by total cache entry count."""

        def f(x):
            return x.sin()

        opt_a = torch.compile(f, backend=backend, isolate_recompiles=True)
        opt_b = torch.compile(f, backend=backend, isolate_recompiles=True)

        opt_a(torch.randn(3))
        self.assertEqual(self._num_cache_entries(f), 1)

        opt_b(torch.randn(3))
        self.assertEqual(self._num_cache_entries(f), 2)

        opt_a(torch.randn(3))
        opt_b(torch.randn(3))
        self.assertEqual(self._num_cache_entries(f), 2)

    # ===== Static vs dynamic: independent compilation strategies =====

    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    def test_isolate_recompiles_static_and_dynamic(self):
        """Two regions on the same function: one static, one dynamic.
        isolate_recompiles keeps their cache entries separate so static
        recompiles don't count against the dynamic region."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sum()

        opt_static = torch.compile(
            f, backend=cnt, dynamic=False, isolate_recompiles=True
        )
        opt_dynamic = torch.compile(
            f, backend=cnt, dynamic=True, isolate_recompiles=True
        )

        opt_static(torch.randn(4, 8))
        self.assertEqual(cnt.frame_count, 1)

        opt_dynamic(torch.randn(5, 9))
        self.assertEqual(cnt.frame_count, 2)

        # Static cache hit
        opt_static(torch.randn(4, 8))
        self.assertEqual(cnt.frame_count, 2)

        # Dynamic cache hit with different shape
        opt_dynamic(torch.randn(6, 10))
        self.assertEqual(cnt.frame_count, 2)

        # Static recompile with new shape
        opt_static(torch.randn(5, 9))
        self.assertEqual(cnt.frame_count, 3)

    def test_isolate_recompiles_mark_dynamic_vs_static(self):
        """Two regions: one with mark_static, one with mark_dynamic.
        Their guards don't interfere across regions."""
        cnt_static = torch._dynamo.testing.CompileCounter()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_static = torch.compile(f, backend=cnt_static, isolate_recompiles=True)
        opt_dynamic = torch.compile(f, backend=cnt_dynamic, isolate_recompiles=True)

        x_static = torch.randn(4, 8)
        torch._dynamo.mark_static(x_static, 0)
        opt_static(x_static)
        self.assertEqual(cnt_static.frame_count, 1)

        x_dynamic = torch.randn(4, 8)
        torch._dynamo.mark_dynamic(x_dynamic, 0)
        opt_dynamic(x_dynamic)
        self.assertEqual(cnt_dynamic.frame_count, 1)

        # Static cache hit — same shape
        x_static2 = torch.randn(4, 8)
        torch._dynamo.mark_static(x_static2, 0)
        opt_static(x_static2)
        self.assertEqual(cnt_static.frame_count, 1)

        # Dynamic cache hit — different shape, same dynamic dim
        x_dynamic2 = torch.randn(7, 8)
        opt_dynamic(x_dynamic2)
        self.assertEqual(cnt_dynamic.frame_count, 1)

        # Static recompile — different shape
        x_static3 = torch.randn(7, 8)
        torch._dynamo.mark_static(x_static3, 0)
        opt_static(x_static3)
        self.assertEqual(cnt_static.frame_count, 2)

    @torch._dynamo.config.patch(automatic_dynamic_shapes=True)
    def test_isolate_recompiles_auto_dynamic_shared_pgo(self):
        """PGO (frame_state) is shared across isolated regions. Region B
        benefits from region A's shape observations — compiles with dynamic
        shapes immediately without redundant static-then-dynamic recompilation."""
        cnt_a = torch._dynamo.testing.CompileCounter()
        cnt_b = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_a = torch.compile(f, backend=cnt_a, isolate_recompiles=True)
        opt_b = torch.compile(f, backend=cnt_b, isolate_recompiles=True)

        opt_a(torch.randn(3, 4))
        opt_a(torch.randn(5, 4))
        self.assertEqual(cnt_a.frame_count, 2)

        # Region B benefits from A's PGO — compiles dynamic immediately
        opt_b(torch.randn(7, 4))
        self.assertEqual(cnt_b.frame_count, 1)

        opt_b(torch.randn(9, 4))
        self.assertEqual(cnt_b.frame_count, 1)

    @torch._dynamo.config.patch(
        recompile_limit=2,
        fail_on_recompile_limit_hit=True,
        automatic_dynamic_shapes=False,
    )
    def test_isolate_recompiles_same_backend_different_dynamic_independent_limits(self):
        """Two regions with the same backend, one static and one dynamic.
        Each exhausts its recompile_limit independently."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_static = torch.compile(
            f, backend=cnt, dynamic=False, isolate_recompiles=True
        )
        opt_dynamic = torch.compile(
            f, backend=cnt, dynamic=True, isolate_recompiles=True
        )

        # Static region: two shapes fill recompile_limit=2
        opt_static(torch.randn(3))
        opt_static(torch.randn(4))
        self.assertEqual(cnt.frame_count, 2)

        # Dynamic region: compiles once, different shapes are cache hits
        opt_dynamic(torch.randn(5))
        self.assertEqual(cnt.frame_count, 3)
        opt_dynamic(torch.randn(6))
        self.assertEqual(cnt.frame_count, 3)

        # Static region hits its limit
        with self.assertRaises(FailOnRecompileLimitHit):
            opt_static(torch.randn(5))

        # Dynamic region still works — independent limit
        opt_dynamic(torch.randn(7))
        self.assertEqual(cnt.frame_count, 3)

    # ===== Recompile limits: per-region and accumulated =====

    @torch._dynamo.config.patch(recompile_limit=1)
    def test_isolate_recompiles_fullgraph_raises(self):
        """With fullgraph=True, hitting the recompile limit raises
        FailOnRecompileLimitHit regardless of fail_on_recompile_limit_hit."""

        def f(x):
            return x.sin()

        opt_f = torch.compile(
            f, backend="eager", fullgraph=True, dynamic=False, isolate_recompiles=True
        )

        opt_f(torch.randn(3))
        with self.assertRaisesRegex(FailOnRecompileLimitHit, "fullgraph=True"):
            opt_f(torch.randn(4))

    @torch._dynamo.config.patch(
        accumulated_recompile_limit=6,
        recompile_limit=4,
        automatic_dynamic_shapes=False,
    )
    def test_isolate_recompiles_accumulated_limit(self):
        """accumulated_recompile_limit is a global safety cap across all
        regions on the same code object. Three regions collectively contribute
        6 entries (2 each), hitting the global cap even though each region
        is below its per-region recompile_limit of 4. New shapes fall back
        to eager via RUN_ONLY."""

        def f(x):
            return x.sin()

        opt_a = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )
        opt_b = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )
        opt_c = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )

        id_a = opt_a._isolate_recompiles_id
        id_b = opt_b._isolate_recompiles_id
        id_c = opt_c._isolate_recompiles_id

        # Region A: 2 compilations (total 2)
        opt_a(torch.randn(1))
        opt_a(torch.randn(2))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_a)), 2)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_b)), 0)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_c)), 0)
        self.assertEqual(_get_total_cache_entry_count(f), 2)

        # Region B: 2 compilations (total 4)
        opt_b(torch.randn(3))
        opt_b(torch.randn(4))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_a)), 2)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_b)), 2)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_c)), 0)
        self.assertEqual(_get_total_cache_entry_count(f), 4)

        # Region C: 2 compilations (total 6 = accumulated_recompile_limit)
        opt_c(torch.randn(5))
        opt_c(torch.randn(6))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_a)), 2)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_b)), 2)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_c)), 2)
        self.assertEqual(_get_total_cache_entry_count(f), 6)

        # All three regions blocked — new shapes fall back to eager
        x7 = torch.randn(7)
        self.assertEqual(opt_a(x7), f(x7))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_a)), 2)

        x8 = torch.randn(8)
        self.assertEqual(opt_b(x8), f(x8))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_b)), 2)

        x9 = torch.randn(9)
        self.assertEqual(opt_c(x9), f(x9))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_c)), 2)
        self.assertEqual(_get_total_cache_entry_count(f), 6)

        # Existing cached shapes still hit cache
        x1 = torch.randn(1)
        self.assertEqual(opt_a(x1), f(x1))

    @torch._dynamo.config.patch(
        accumulated_recompile_limit=4,
        recompile_limit=8,
        automatic_dynamic_shapes=False,
        fail_on_recompile_limit_hit=True,
    )
    def test_isolate_recompiles_accumulated_limit_hard_fail(self):
        """With fail_on_recompile_limit_hit=True, exceeding accumulated_recompile_limit
        across isolated regions raises FailOnRecompileLimitHit."""

        def f(x):
            return x.cos()

        opt_a = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )
        opt_b = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )

        id_a = opt_a._isolate_recompiles_id
        id_b = opt_b._isolate_recompiles_id

        # 2 entries each = 4 total = accumulated_recompile_limit
        opt_a(torch.randn(1))
        opt_a(torch.randn(2))
        opt_b(torch.randn(3))
        opt_b(torch.randn(4))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_a)), 2)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_b)), 2)

        with self.assertRaises(FailOnRecompileLimitHit):
            opt_a(torch.randn(5))
        with self.assertRaises(FailOnRecompileLimitHit):
            opt_b(torch.randn(6))

    # ===== RUN_ONLY strategy: per-region persistence after limit hit =====

    @torch._dynamo.config.patch(
        recompile_limit=1,
        automatic_dynamic_shapes=False,
    )
    def test_isolate_recompiles_limit_does_not_skip_other_regions(self):
        """When one region hits its recompile limit and goes RUN_ONLY,
        other regions (both isolated and non-isolated) can still compile."""

        def f(x):
            return x.sin()

        opt_a = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )
        opt_b = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )
        opt_default = torch.compile(f, backend="eager", dynamic=False)

        id_a = opt_a._isolate_recompiles_id
        id_b = opt_b._isolate_recompiles_id

        # Region A compiles once, then hits limit
        opt_a(torch.randn(3))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_a)), 1)
        opt_a(torch.randn(4))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_a)), 1)

        # Region B still compiles
        opt_b(torch.randn(5))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_b)), 1)

        # Default (non-isolated) region still compiles
        opt_default(torch.randn(6))
        self.assertEqual(len(_get_cache_entries_for_region(f, -1)), 1)

    @torch._dynamo.config.patch(
        recompile_limit=1,
        automatic_dynamic_shapes=False,
    )
    def test_isolate_recompiles_region_run_only_persists(self):
        """After hitting the recompile limit, RUN_ONLY is persisted per-region
        in ExtraState.region_strategy_map. Subsequent calls skip the callback
        entirely (no repeated limit-hit warnings). Cached shapes still work."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_a = torch.compile(f, backend=cnt, dynamic=False, isolate_recompiles=True)
        opt_b = torch.compile(f, backend=cnt, dynamic=False, isolate_recompiles=True)
        id_a = opt_a._isolate_recompiles_id
        id_b = opt_b._isolate_recompiles_id

        # Region A compiles shape 3
        opt_a(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_a)), 1)

        # Region A hits limit — RUN_ONLY set for this region
        opt_a(torch.randn(4))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_a)), 1)

        # RUN_ONLY persists — callback not re-entered on new shapes
        opt_a(torch.randn(5))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_a)), 1)

        # Cached shape 3 still produces correct result
        x3 = torch.randn(3)
        self.assertEqual(opt_a(x3), f(x3))
        self.assertEqual(cnt.frame_count, 1)

        # Region B is unaffected
        opt_b(torch.randn(6))
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_b)), 1)

    # ===== Default strategy × region: SKIP inherited, RUN_ONLY not =====

    def test_isolate_recompiles_inherits_default_skip(self):
        """Global SKIP (from skip_code / @torch._dynamo.skip / FX plumbing /
        TorchScript __init__ / etc.) is a correctness decision — the code
        must not be traced. Isolated regions inherit this SKIP, so neither
        the default nor isolated wrapper compiles a skip_code-marked code
        object. Only the automatic RUN_ONLY (from a prior non-isolated
        recompile-limit hit) is prevented from bleeding into regions."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        torch._dynamo.eval_frame.skip_code(f.__code__)

        opt_default = torch.compile(f, backend=cnt)
        opt_default(torch.randn(3))
        self.assertEqual(cnt.frame_count, 0)
        self.assertEqual(len(_get_cache_entries_for_region(f, -1)), 0)

        opt_iso = torch.compile(f, backend=cnt, isolate_recompiles=True)
        id_iso = opt_iso._isolate_recompiles_id

        x = torch.randn(3)
        self.assertEqual(opt_iso(x), f(x))
        self.assertEqual(cnt.frame_count, 0)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_iso)), 0)

    def test_isolate_recompiles_ignores_default_run_only(self):
        """Regression for the RUN_ONLY-bleed case: a prior non-isolated
        recompile-limit hit sets RUN_ONLY on extra->strategy. A later
        isolated region on the same code object must not inherit that
        RUN_ONLY and must compile normally."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        # Trip the non-isolated recompile limit so RUN_ONLY is persisted
        # to extra->strategy (non-isolated bucket).
        with torch._dynamo.config.patch(
            recompile_limit=1, automatic_dynamic_shapes=False
        ):
            opt_default = torch.compile(f, backend=cnt, dynamic=False)
            opt_default(torch.randn(3))
            opt_default(torch.randn(4))  # hits limit → RUN_ONLY persisted
            opt_default(torch.randn(5))  # RUN_ONLY path, no compile
        self.assertEqual(cnt.frame_count, 1)

        # Isolated region must ignore the persisted RUN_ONLY.
        opt_iso = torch.compile(f, backend=cnt, isolate_recompiles=True)
        id_iso = opt_iso._isolate_recompiles_id
        opt_iso(torch.randn(6))
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(len(_get_cache_entries_for_region(f, id_iso)), 1)

    # ===== Cache internals: insertion order, fallback, shared bucket =====

    @torch._dynamo.config.patch(recompile_limit=2, automatic_dynamic_shapes=False)
    def test_isolate_recompiles_insertion_order_per_region(self):
        """New entries are added at the front of their region's list.
        Interleaved compilations across regions don't mix ordering.
        After hitting recompile_limit, entries are frozen."""

        def f(x):
            return x.sin()

        opt_a = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )
        opt_b = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )
        id_a = opt_a._isolate_recompiles_id
        id_b = opt_b._isolate_recompiles_id

        # Interleave compilations
        opt_a(torch.randn(3))
        opt_b(torch.randn(10))
        opt_a(torch.randn(4))
        opt_b(torch.randn(11))

        # Newest at front in each region
        entries_a = _get_cache_entries_for_region(f, id_a)
        self.assertEqual(len(entries_a), 2)
        self.assertGreater(
            entries_a[0].compile_id.frame_compile_id,
            entries_a[1].compile_id.frame_compile_id,
        )

        entries_b = _get_cache_entries_for_region(f, id_b)
        self.assertEqual(len(entries_b), 2)
        self.assertGreater(
            entries_b[0].compile_id.frame_compile_id,
            entries_b[1].compile_id.frame_compile_id,
        )

        # Both at limit — no new entries
        opt_a(torch.randn(5))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_a)), 2)
        opt_b(torch.randn(12))
        self.assertEqual(len(_get_cache_entries_for_region(f, id_b)), 2)

    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    def test_isolate_recompiles_lru_move_to_front(self):
        """On a cache hit, the matched entry moves to the front of its
        region's list (LRU). Verify by inspecting compile_id ordering
        before and after the hit. Also verify cross-region independence."""

        def f(x):
            return x.sin()

        opt_a = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )
        opt_b = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )
        id_a = opt_a._isolate_recompiles_id
        id_b = opt_b._isolate_recompiles_id

        # Region A: compile shapes 3, 4, 5.
        # Insertion order (newest at front): [5, 4, 3]
        opt_a(torch.randn(3))
        opt_a(torch.randn(4))
        opt_a(torch.randn(5))

        entries_a = _get_cache_entries_for_region(f, id_a)
        self.assertEqual(len(entries_a), 3)
        ids_before = [e.compile_id for e in entries_a]

        # Region B: compile shapes 6, 7.
        opt_b(torch.randn(6))
        opt_b(torch.randn(7))
        entries_b = _get_cache_entries_for_region(f, id_b)

        # Hit region A with shape 3 (oldest entry, at back) — LRU moves to front
        opt_a(torch.randn(3))
        entries_a_after = _get_cache_entries_for_region(f, id_a)
        ids_after = [e.compile_id for e in entries_a_after]

        # shape-3 entry was last, now first
        self.assertEqual(ids_after[0], ids_before[-1])
        self.assertEqual(ids_after[1], ids_before[0])
        self.assertEqual(ids_after[2], ids_before[1])

        # No new entries — cache hit, not recompilation
        self.assertEqual(len(entries_a_after), 3)

        # Region B order unchanged — LRU on A doesn't affect B
        entries_b_after = _get_cache_entries_for_region(f, id_b)
        self.assertEqual(
            [e.compile_id for e in entries_b_after],
            [e.compile_id for e in entries_b],
        )

    def test_non_isolated_entries_visible_to_isolated(self):
        """Non-isolated entries (bucket -1) are visible to isolated regions
        via read-only fallback when the backend matches. BC friendly —
        isolated compiles reuse existing non-isolated compilations."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.exp()

        opt_global = torch.compile(f, backend=cnt)
        opt_global(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

        opt_isolated = torch.compile(f, backend=cnt, isolate_recompiles=True)
        opt_isolated(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

    @torch._dynamo.config.patch(recompile_limit=8)
    def test_isolate_recompiles_reasons_include_default_bucket(self):
        """Recompile-reason logging for an isolated region must also walk
        the default (-1) bucket. lookup() checks default entries as a
        fallback for isolated regions, so their guard failures are real
        and must not be dropped from the recompile-reason log."""
        default_fails: list[str] = []
        region_fails: list[str] = []

        def record_default(failure):
            default_fails.append(failure.reason)

        def record_region(failure):
            region_fails.append(failure.reason)

        def f(x):
            return x.sum()

        opt_default = torch._dynamo.optimize(
            "eager", guard_fail_fn=record_default, dynamic=False
        )(f)
        opt_isolated = torch._dynamo.optimize(
            "eager",
            guard_fail_fn=record_region,
            dynamic=False,
            isolate_recompiles=True,
        )(f)

        # Populate default bucket with a shape-3 entry.
        opt_default(torch.randn(3))
        # Populate region bucket with a shape-4 entry.
        opt_isolated(torch.randn(4))
        # Recompile in the region: shape-5 misses both buckets.
        # The logging path must report guard failures for BOTH the
        # region's shape-4 entry and the default bucket's shape-3 entry.
        opt_isolated(torch.randn(5))

        self.assertTrue(
            region_fails, f"region entries' guard failures missing: {region_fails}"
        )
        self.assertTrue(
            default_fails,
            f"default-bucket entries' guard failures dropped from "
            f"recompile reasons (bug): {default_fails}",
        )

    @torch._dynamo.config.patch(recompile_limit=8)
    def test_non_isolated_reasons_unchanged(self):
        """Regression: recompile-reason logging for non-isolated compiles
        (id=-1) must still work. The split between cache_entries and
        cache_entries_for_reasons should not have changed this path."""
        fails: list[str] = []

        def f(x):
            return x.sum()

        opt = torch._dynamo.optimize(
            "eager",
            guard_fail_fn=lambda failure: fails.append(failure.reason),
            dynamic=False,
        )(f)

        opt(torch.randn(3))
        opt(torch.randn(4))
        self.assertTrue(fails, f"no recompile reasons logged: {fails}")

    @torch._dynamo.config.patch(recompile_limit=8)
    def test_reasons_include_all_default_entries(self):
        """When the default bucket has multiple entries, recompile-reason
        logging from an isolated region must report guard failures from
        each of them, not just one."""
        default_fails: list[str] = []

        def f(x):
            return x.sum()

        opt_default = torch._dynamo.optimize(
            "eager",
            guard_fail_fn=lambda failure: default_fails.append(failure.reason),
            dynamic=False,
        )(f)
        opt_default(torch.randn(3))
        opt_default(torch.randn(4))

        # Prime the region with one entry — recompile-reason logging only
        # fires on subsequent calls (is_recompilation requires the region
        # to already have ≥1 entry).
        opt_iso = torch._dynamo.optimize(
            "eager", dynamic=False, isolate_recompiles=True
        )(f)
        opt_iso(torch.randn(5))
        default_fails.clear()

        # Recompile in the region — shape 6 misses the region entry and
        # both default entries. All three should contribute guard-failure
        # reasons; default_fails must receive both default entries' fails.
        opt_iso(torch.randn(6))

        self.assertGreaterEqual(
            len(default_fails),
            2,
            f"expected guard failures for both default entries, got {default_fails}",
        )

    def test_reset_clears_region_strategy(self):
        """torch._dynamo.reset() must clear region_strategy_map on
        ExtraState. Otherwise a RUN_ONLY persisted by a prior region
        would survive reset and prevent the new region from compiling."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        with torch._dynamo.config.patch(
            recompile_limit=1, automatic_dynamic_shapes=False
        ):
            opt_a = torch.compile(
                f, backend=cnt, dynamic=False, isolate_recompiles=True
            )
            opt_a(torch.randn(3))
            opt_a(torch.randn(4))  # hits limit → region RUN_ONLY persisted
            self.assertEqual(cnt.frame_count, 1)

        torch._dynamo.reset()

        opt_b = torch.compile(f, backend=cnt, isolate_recompiles=True)
        opt_b(torch.randn(5))
        self.assertEqual(cnt.frame_count, 2)

    @torch._dynamo.config.patch(
        recompile_limit=2,
        fail_on_recompile_limit_hit=True,
        automatic_dynamic_shapes=False,
    )
    def test_non_isolated_compiles_share_cache(self):
        """Without isolate_recompiles, two compile calls share bucket -1.
        They share cache hits AND recompile limits."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.exp()

        opt_a = torch.compile(f, backend=cnt, dynamic=False)
        opt_b = torch.compile(f, backend=cnt, dynamic=False)

        opt_a(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

        # Cache hit from opt_a's entry
        opt_b(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

        # New shape from opt_b counts toward shared limit
        opt_b(torch.randn(4))
        self.assertEqual(cnt.frame_count, 2)

        opt_a(torch.randn(4))
        self.assertEqual(cnt.frame_count, 2)

        # Shared limit exceeded
        with self.assertRaises(FailOnRecompileLimitHit):
            opt_a(torch.randn(5))

    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    def test_no_isolate_recompiles_shared_cache(self):
        """Baseline: without isolate_recompiles, compile calls share the
        cache. A recompile from one is visible to the other."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sum()

        opt_a = torch.compile(f, backend=cnt, dynamic=False)
        opt_b = torch.compile(f, backend=cnt, dynamic=False)

        opt_a(torch.randn(4))
        self.assertEqual(cnt.frame_count, 1)

        opt_b(torch.randn(4))
        self.assertEqual(cnt.frame_count, 1)

        opt_a(torch.randn(5))
        self.assertEqual(cnt.frame_count, 2)

        opt_b(torch.randn(5))
        self.assertEqual(cnt.frame_count, 2)

    @torch._dynamo.config.patch(
        recompile_limit=2,
        fail_on_recompile_limit_hit=True,
        automatic_dynamic_shapes=False,
    )
    def test_different_backends_shared_cache_without_isolate(self):
        """Baseline: without isolate_recompiles, different backends share the
        cache. Entries from backend A count against backend B's limit."""
        cnt_a = torch._dynamo.testing.CompileCounter()
        cnt_b = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sum()

        opt_a = torch.compile(f, backend=cnt_a, dynamic=False)
        opt_b = torch.compile(f, backend=cnt_b, dynamic=False)

        opt_a(torch.randn(3))
        opt_a(torch.randn(4))
        self.assertEqual(cnt_a.frame_count, 2)

        with self.assertRaises(FailOnRecompileLimitHit):
            opt_b(torch.randn(5))

    @torch._dynamo.config.patch(
        recompile_limit=2,
        fail_on_recompile_limit_hit=True,
        automatic_dynamic_shapes=False,
    )
    def test_different_backends_independent_with_isolate(self):
        """With isolate_recompiles, different backends get independent buckets."""
        cnt_a = torch._dynamo.testing.CompileCounter()
        cnt_b = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sum()

        opt_a = torch.compile(f, backend=cnt_a, dynamic=False, isolate_recompiles=True)
        opt_b = torch.compile(f, backend=cnt_b, dynamic=False, isolate_recompiles=True)

        opt_a(torch.randn(3))
        opt_a(torch.randn(4))
        self.assertEqual(cnt_a.frame_count, 2)

        # B compiles independently
        opt_b(torch.randn(5))
        self.assertEqual(cnt_b.frame_count, 1)
        opt_b(torch.randn(6))
        self.assertEqual(cnt_b.frame_count, 2)

    # ===== Lifecycle: reset, resume functions, GC =====

    def test_isolate_recompiles_reset(self):
        """torch._dynamo.reset() clears all regions."""
        cnt_a = torch._dynamo.testing.CompileCounter()
        cnt_b = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.cos()

        opt_a = torch.compile(f, backend=cnt_a, isolate_recompiles=True)
        opt_b = torch.compile(f, backend=cnt_b, isolate_recompiles=True)

        opt_a(torch.randn(3))
        opt_b(torch.randn(4))
        self.assertEqual(cnt_a.frame_count, 1)
        self.assertEqual(cnt_b.frame_count, 1)

        torch._dynamo.reset()

        opt_a(torch.randn(3))
        opt_b(torch.randn(4))
        self.assertEqual(cnt_a.frame_count, 2)
        self.assertEqual(cnt_b.frame_count, 2)

    @torch._dynamo.config.patch(recompile_limit=3)
    def test_isolate_recompiles_resume_function(self):
        """Resume functions from a graph break inherit the region's
        isolate_recompiles_id and respect its per-region recompile limit."""
        cnt = torch._dynamo.testing.CompileCounter()

        mode = {"value": "a"}

        def f(x):
            a = x.sin()
            torch._dynamo.graph_break()
            if mode["value"] == "a":
                return a.cos()
            elif mode["value"] == "b":
                return a.tan()
            elif mode["value"] == "c":
                return a.exp()
            else:
                return a + 1

        opt_f = torch.compile(f, backend=cnt, isolate_recompiles=True)

        opt_f(torch.randn(4))
        frame_count_after_1 = cnt.frame_count

        mode["value"] = "b"
        opt_f(torch.randn(4))
        frame_count_after_2 = cnt.frame_count
        self.assertGreater(frame_count_after_2, frame_count_after_1)

        mode["value"] = "c"
        opt_f(torch.randn(4))
        frame_count_after_3 = cnt.frame_count
        self.assertGreater(frame_count_after_3, frame_count_after_2)

        # Resume function has 3 entries = recompile_limit. Fourth blocked.
        mode["value"] = "d"
        opt_f(torch.randn(4))
        self.assertEqual(cnt.frame_count, frame_count_after_3)

    def test_isolate_recompiles_gc_wrapper(self):
        """When an isolated compile wrapper is GC'd, orphaned cache entries
        remain. A new torch.compile gets a fresh region and compiles
        independently. reset() clears everything including orphans."""
        import gc

        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_a = torch.compile(f, backend=cnt, isolate_recompiles=True)
        opt_a(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(self._num_cache_entries(f), 1)

        del opt_a
        gc.collect()

        # Orphaned entry persists
        self.assertEqual(self._num_cache_entries(f), 1)

        # Fresh region compiles independently
        opt_b = torch.compile(f, backend=cnt, isolate_recompiles=True)
        opt_b(torch.randn(3))
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(self._num_cache_entries(f), 2)

        torch._dynamo.reset()
        self.assertEqual(self._num_cache_entries(f), 0)

    # ===== Debug / introspection =====

    def test_debug_get_cache_entry_list_deterministic_order(self):
        """_debug_get_cache_entry_list returns entries sorted by
        isolate_recompiles_id for deterministic output."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opts = [
            torch.compile(f, backend=cnt, isolate_recompiles=True) for _ in range(6)
        ]
        for opt in reversed(opts):
            opt(torch.randn(3))

        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(f)
        self.assertEqual(len(entries), 6)

        ids = [e.isolate_recompiles_id for e in entries]
        self.assertEqual(ids, sorted(ids))


instantiate_parametrized_tests(IsolateRecompilesTests)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
