# Owner(s): ["module: dynamo"]
from unittest.mock import patch

import torch
import torch._dynamo.test_case
import torch._dynamo.testing


class RecompileTests(torch._dynamo.test_case.TestCase):
    def test_automatic_dynamic_reduce_recompiles(self):
        # Test the counterfactual, lots of recompiles without this config
        def foo(x, y):
            return x * y

        def run_foo_6_times_and_count_recompiles(dynamic=None):
            cnt = torch._dynamo.testing.CompileCounter()

            x = torch.randn([2])
            y = torch.randn([2])
            opt = torch._dynamo.optimize(cnt, dynamic=dynamic)(foo)
            opt(x, y)
            x = torch.randn([3])
            y = torch.randn([3])
            opt(x, y)
            x = torch.randn([4])
            y = torch.randn([4])
            opt(x, y)
            opt(x, y)
            x = torch.randn([5])
            y = torch.randn([5])
            opt(x, y)
            opt(x, y)
            x = torch.randn([6])
            y = torch.randn([6])
            opt(x, y)

            return cnt

        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_without_automatic():
            return run_foo_6_times_and_count_recompiles()

        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", True)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_with_automatic():
            return run_foo_6_times_and_count_recompiles()

        without = run_without_automatic()
        self.assertEqual(without.frame_count, 5)
        self.assertEqual(without.op_count, 5)
        torch._dynamo.reset()
        without = run_foo_6_times_and_count_recompiles(dynamic=False)
        self.assertEqual(without.frame_count, 5)
        self.assertEqual(without.op_count, 5)
        torch._dynamo.reset()
        with_automatic = run_with_automatic()
        self.assertEqual(with_automatic.frame_count, 2)
        self.assertEqual(with_automatic.op_count, 2)
        torch._dynamo.reset()
        with_automatic = run_foo_6_times_and_count_recompiles(dynamic=None)
        self.assertEqual(with_automatic.frame_count, 2)
        self.assertEqual(with_automatic.op_count, 2)
        torch._dynamo.reset()
        with_dynamic = run_foo_6_times_and_count_recompiles(dynamic=True)
        self.assertEqual(with_dynamic.frame_count, 1)
        self.assertEqual(with_dynamic.op_count, 1)

    @patch.object(torch._dynamo.config, "assume_static_by_default", True)
    def test_recompiles_true_false_flop(self):
        # Test the counterfactual, lots of recompiles without this config
        def foo(x, y):
            if x:
                return y * 2
            else:
                return y * y

        def run_foo_6_times_and_count_recompiles():
            cnt = torch._dynamo.testing.CompileCounter()

            opt = torch._dynamo.optimize(cnt, nopython=True)(foo)

            x = True
            y = torch.randn([2])
            opt(x, y)
            x = False
            y = torch.randn([2])
            opt(x, y)
            x = True
            y = torch.randn([3])
            opt(x, y)
            x = True
            y = torch.randn([4])
            opt(x, y)
            x = True
            y = torch.randn([5])
            opt(x, y)

            return cnt

        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_without_automatic():
            return run_foo_6_times_and_count_recompiles()

        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", True)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_with_automatic():
            return run_foo_6_times_and_count_recompiles()

        without = run_without_automatic()
        self.assertEqual(without.frame_count, 5)
        self.assertEqual(without.op_count, 5)
        torch._dynamo.reset()
        with_automatic = run_with_automatic()
        self.assertEqual(with_automatic.frame_count, 3)
        self.assertEqual(with_automatic.op_count, 3)

    def test_automatic_dynamic_tensor_scalar_change(self):
        # Test the counterfactual, lots of recompiles without this config
        def foo(x, y):
            return x * y

        def run_foo_6_times_and_count_recompiles_swap_types():
            cnt = torch._dynamo.testing.CompileCounter()

            x = torch.randn([2])
            y = torch.randn([2])
            opt = torch._dynamo.optimize(cnt)(foo)
            opt(x, y)
            x = torch.randn([3])
            y = 3
            opt(x, y)
            x = torch.randn([4])
            y = torch.randn([4])
            opt(x, y)
            opt(x, y)
            x = torch.randn([5])
            y = 4
            opt(x, y)
            opt(x, y)
            x = torch.randn([6])
            y = torch.randn([6])
            opt(x, y)

            return cnt

        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_without_automatic():
            return run_foo_6_times_and_count_recompiles_swap_types()

        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", True)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_with_automatic():
            return run_foo_6_times_and_count_recompiles_swap_types()

        without = run_without_automatic()
        self.assertEqual(without.frame_count, 5)
        self.assertEqual(without.op_count, 5)
        torch._dynamo.reset()
        with_automatic = run_with_automatic()
        self.assertEqual(with_automatic.frame_count, 3)
        self.assertEqual(with_automatic.op_count, 3)

    def test_aliasing_guard_failures(self):
        def foo(a, b, c):
            a.add_(b)
            return c + 1

        cnt = torch._dynamo.testing.CompileCounter()
        compiled_foo = torch._dynamo.optimize(cnt, nopython=True)(foo)

        x = torch.randn([3])
        y = torch.randn([3])
        z = torch.randn([3])
        cmp_result = compiled_foo(
            x.clone().detach(), y.clone().detach(), z.clone().detach()
        )
        eager_result = foo(x.clone().detach(), y.clone().detach(), z.clone().detach())
        self.assertEqual(cmp_result, eager_result)
        self.assertEqual(cnt.frame_count, 1)

        cmp_result = compiled_foo(
            z.clone().detach(), y.clone().detach(), x.clone().detach()
        )
        eager_result = foo(z.clone().detach(), y.clone().detach(), x.clone().detach())
        self.assertEqual(cmp_result, eager_result)
        # No recompile, alias preserved
        self.assertEqual(cnt.frame_count, 1)

        x_clone = x.clone().detach()
        cmp_result = compiled_foo(x_clone, y.clone().detach(), x_clone)
        x_clone = x.clone().detach()
        eager_result = compiled_foo(x_clone, y.clone().detach(), x_clone)
        self.assertEqual(cmp_result, eager_result)
        # Recompile, alias changed
        self.assertEqual(cnt.frame_count, 2)

    def test_aliasing_guard_failures_with_globals(self):
        g1 = torch.randn([3])
        g2 = torch.randn([3])

        def foo(a):
            a.add_(g1)
            return g2 + 1

        cnt = torch._dynamo.testing.CompileCounter()
        compiled_foo = torch._dynamo.optimize(cnt, nopython=True)(foo)

        z = torch.randn([3])
        cmp_result = compiled_foo(z.clone().detach())
        eager_result = foo(z.clone().detach())
        self.assertEqual(cmp_result, eager_result)
        self.assertEqual(cnt.frame_count, 1)

        g1 = g1.clone().detach()
        cmp_result = compiled_foo(g1)
        g1 = g1.clone().detach()
        eager_result = compiled_foo(g1)
        self.assertEqual(cmp_result, eager_result)
        # Recompile, alias changed
        self.assertEqual(cnt.frame_count, 2)

    def test_dynamic_shape_parameter_recompile(self):
        # Test the matrix multiplication with Parameters.
        # Without the config assume_parameters_shapes_static_by_default,
        # the torch.nn.Parameter shapes are assumed to be static which leads to recompilation

        w = torch.nn.Parameter(torch.randn(3, 2))

        def foo(x):
            return x @ w

        def run_foo_6_times_and_count_recompiles():
            cnt = torch._dynamo.testing.CompileCounter()

            opt = torch._dynamo.optimize(cnt, nopython=True)(foo)

            x = torch.nn.Parameter(torch.randn(1, 3))
            opt(x)
            x = torch.nn.Parameter(torch.randn(10, 3))
            opt(x)
            x = torch.nn.Parameter(torch.randn(11, 3))
            opt(x)
            x = torch.nn.Parameter(torch.randn(15, 3))
            opt(x)
            x = torch.nn.Parameter(torch.randn(15, 3))
            opt(x)

            return cnt

        @patch.object(torch._dynamo.config, "force_parameter_static_shapes", True)
        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_static_comp_default_param():
            return run_foo_6_times_and_count_recompiles()

        @patch.object(torch._dynamo.config, "force_parameter_static_shapes", True)
        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", True)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_dynamic_comp_default_param():
            return run_foo_6_times_and_count_recompiles()

        @patch.object(torch._dynamo.config, "force_parameter_static_shapes", False)
        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_static_comp_dynamic_param():
            return run_foo_6_times_and_count_recompiles()

        @patch.object(torch._dynamo.config, "force_parameter_static_shapes", False)
        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", True)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_dynamic_comp_dynamic_param():
            return run_foo_6_times_and_count_recompiles()

        torch._dynamo.reset()
        static_comp_default_param = run_static_comp_default_param()
        self.assertEqual(static_comp_default_param.frame_count, 4)
        self.assertEqual(static_comp_default_param.op_count, 4)

        torch._dynamo.reset()
        dynamic_comp_default_param = run_dynamic_comp_default_param()
        self.assertEqual(dynamic_comp_default_param.frame_count, 4)
        self.assertEqual(dynamic_comp_default_param.op_count, 4)

        torch._dynamo.reset()
        static_comp_dynamic_param = run_static_comp_dynamic_param()
        self.assertEqual(static_comp_dynamic_param.frame_count, 4)
        self.assertEqual(static_comp_dynamic_param.op_count, 4)

        torch._dynamo.reset()
        dynamic_comp_dynamic_param = run_dynamic_comp_dynamic_param()
        self.assertEqual(dynamic_comp_dynamic_param.frame_count, 2)
        self.assertEqual(dynamic_comp_dynamic_param.op_count, 2)

    def test_simple_module_recompile(self):
        class SimpleDropout(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dropout = torch.nn.Dropout(0.5)
                self.linear = torch.nn.Linear(10, 1)

            def forward(self, x):
                return self.dropout(self.linear(x))

        model = SimpleDropout()
        x = torch.randn(10)
        counter = torch._dynamo.testing.CompileCounter()
        model = torch.compile(model, backend=counter, fullgraph=True)
        for _ in range(20):
            model.eval()
            model(x)
            model.train()
            model(x)
        self.assertEqual(counter.frame_count, 2)

    @patch.object(torch._dynamo.config, "cache_size_limit", 2)
    def test_no_recursive_compile_after_cache_limit_hit(self):
        def f(x, n):
            x = x + n
            return g(x, n)

        def g(x, n):
            x = x + n
            return h(x, n)

        def h(x, n):
            return x + n

        counter = torch._dynamo.testing.CompileCounter()
        opt_f = torch.compile(f, backend=counter, dynamic=False)
        for i in range(10):
            opt_f(torch.ones(3), i)
        self.assertEqual(counter.frame_count, 2)

    def test_automatic_dynamic_on_closed_ints(self):
        def f(x):
            def g(y):
                return y + x

            return g

        counter = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=counter)
        def h(x, g):
            return g(x)

        for i in range(10):
            h(torch.randn(5), f(i))
        self.assertEqual(counter.frame_count, 2)

    @patch.object(torch._dynamo.config, "cache_size_limit", 2)
    def test_run_mode_after_cache_limit_hit(self):
        def f(x, n):
            x = x + n
            if torch._dynamo.is_compiling():
                x = x + 1
            return g(x, n)

        def g(x, n):
            x = x + n
            if torch._dynamo.is_compiling():
                x = x + 2
            return x

        counter = torch._dynamo.testing.CompileCounter()
        opt_f = torch.compile(f, backend=counter, dynamic=False)
        # compiles
        self.assertEqual(opt_f(torch.ones(3), 0), torch.ones(3) + 3)
        self.assertEqual(opt_f(torch.ones(3), 1), torch.ones(3) + 5)
        # cache limit hit
        self.assertEqual(opt_f(torch.ones(3), 2), torch.ones(3) + 4)
        self.assertEqual(opt_f(torch.ones(3), 3), torch.ones(3) + 6)
        # run mode
        self.assertEqual(opt_f(torch.ones(3), 0), torch.ones(3) + 3)
        self.assertEqual(opt_f(torch.ones(3), 1), torch.ones(3) + 5)
        self.assertEqual(counter.frame_count, 2)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
