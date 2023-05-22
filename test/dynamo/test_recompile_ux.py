# Owner(s): ["module: dynamo"]
import unittest
import weakref
from unittest.mock import patch

import torch

import torch._dynamo
import torch._dynamo.config
import torch._dynamo.test_case
import torch._dynamo.testing


class RecompileUxTests(torch._dynamo.test_case.TestCase):
    # TODO(whc) dynamo actualy recompiles one more time than the cache limit
    cache_limit = 1

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack.enter_context(
            torch._dynamo.config.patch("cache_size_limit", cls.cache_limit),
        )
        cls._exit_stack.enter_context(
            torch._dynamo.config.patch("dynamic_shapes", False),
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
            assert not attached
            # NB: making this a weakref.ref causes the cycle to no
            # longer be promptly GC'ed
            weakref.finalize(f, trigger)
            attached = True
            return f

        x = torch.randn(2)
        for i in range(2):
            opt_model = torch._dynamo.optimize(compiler)(model)
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
            opt_loop_torture = torch._dynamo.optimize(compile_counter)(loop_torture)
            opt_loop_torture(x, iters)

        # Currently, we recompile each time,
        # We'd probably like to bail out quickly and warn
        # TODO(whc) these checks fail on py37.  Why?
        # self.assertEqual(counters["frames"]["total"], 2 + self.cache_limit)
        # self.assertEqual(counters["frames"]["ok"], 1 + self.cache_limit)

        # compile_counter only sees frames that were fed to the backend compiler,
        # which is a subset of counters["frames"]["ok"] -- probably becuase
        # counters["frames"]["ok"] includes frames not containing torch ops?
        self.assertEqual(compile_counter.frame_count, self.cache_limit)

    def test_dynamic_input(self):
        def model(input):
            return input + input

        expected_recompiles = 2
        compile_counter = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch("cache_size_limit", expected_recompiles):
            with self.assertLogs(logger="torch._dynamo", level="WARNING") as logs:
                for _ in range(10):
                    bsz = torch.randint(low=0, high=1000, size=())
                    x = torch.randn((bsz, 3, 4))
                    opt_model = torch._dynamo.optimize(compile_counter)(model)
                    opt_model(x)

        self.assertEqual(compile_counter.frame_count, expected_recompiles)
        self.assertEqual(len(logs.records), 1)
        print(logs.records[0])
        self.assertTrue(
            logs.records[0]
            .getMessage()
            .startswith("torch._dynamo hit config.cache_size_limit")
        )

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_nvfuser_guards(self):
        # we may want to model dynamo's guards sufficiently after nvfuser's ProfilingExecutor guards
        # such that we ensure dynamo is in charge of all the recompilations at the top level,
        # and we could thus simplfy the underlying torchscript executor
        def func(a, b, c):
            return a + b * c

        a = torch.rand(3, 4, 5, device="cuda")
        b = torch.rand(3, 4, 5, device="cuda")
        b_v = torch.rand(3, 5, 4, device="cuda").view(3, 4, 5)
        b_p = torch.rand(3, 5, 4, device="cuda").permute(0, 2, 1)
        c = torch.rand(3, 4, 5, device="cuda")
        compile_counter = torch._dynamo.testing.CompileCounter()

        with torch._dynamo.config.patch("cache_size_limit", 2):
            opt_func = torch._dynamo.optimize(compile_counter)(func)
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

    @patch.object(torch._dynamo.config, "report_guard_failures", True)
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
            opt_func = torch._dynamo.optimize("eager")(func)
            # warmup
            opt_func(cached_input)

            with self.assertLogs(logger="torch._dynamo", level="WARNING") as logs:
                opt_func = torch._dynamo.optimize("eager")(func)
                opt_func(missed_input)
            self.assert_single_log_contains(logs, expected_failure)

        a = torch.rand(3, 4, 5)
        cache_fail_test(
            a,
            a[0:2, :, :],
            "tensor 'L['a']' size mismatch at index 0. expected 3, actual 2",
        )
        cache_fail_test(
            a,
            a.clone().as_strided((3, 4, 5), stride=(1, 3, 12)),
            "tensor 'L['a']' stride mismatch at index 0. expected 20, actual 1",
        )
        cache_fail_test(
            a, a[0, :, :], "tensor 'L['a']' rank mismatch. expected 3, actual 2"
        )
        cache_fail_test(a, a.to("meta"), "tensor 'L['a']' dispatch key set mismatch.")
        cache_fail_test(
            a,
            a.to(torch.float16),
            "tensor 'L['a']' dtype mismatch. expected Float, actual Half",
        )
        a_grad = a.clone()
        a_grad.requires_grad = True
        cache_fail_test(
            a,
            a_grad,
            "tensor 'L['a']' requires_grad mismatch. expected requires_grad=0",
        )

    @patch.object(torch._dynamo.config, "report_guard_failures", True)
    def test_mismatched_type(self):
        a = torch.rand(3, 4, 5)
        b = torch.rand(3, 4, 5)

        def func(a, b):
            return a + b

        opt_func = torch._dynamo.optimize("eager")(func)
        # warmup
        opt_func(a, b)

        with self.assertLogs(logger="torch._dynamo", level="WARNING") as logs:
            opt_func = torch._dynamo.optimize("eager")(func)
            opt_func(a, 1)
        self.assert_single_log_contains(
            logs,
            "expected type of 'L['b']' to be a tensor type, ' but found <class 'int'>",
        )


# TODO(jansel): these pass with pytest, but not with pytorch CI
# if __name__ == "__main__":
#     from torch._dynamo.testing import run_tests
#     run_tests()
