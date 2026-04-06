# Owner(s): ["module: dynamo"]
import unittest
import weakref

import torch
import torch._dynamo
import torch._dynamo.config
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._logging
from torch._dynamo.exc import FailOnRecompileLimitHit
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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
