# Owner(s): ["module: dynamo"]
from unittest.mock import patch

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo import config as dc


class RecompileTests(torch._dynamo.test_case.TestCase):
    def test_inline_inbuilt_nn_modules_candidate(self):
        def hook_flag_on(guard_manager, f_locals, builder):
            self.assertTrue(
                "[inline-inbuilt-nn-modules-candidate]" not in str(guard_manager)
            )

        def hook_flag_off(guard_manager, f_locals, builder):
            self.assertTrue(
                "[inline-inbuilt-nn-modules-candidate]" in str(guard_manager)
            )

        class SubMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            @torch.compile(backend="eager")
            def forward(self, x):
                return self.linear(x)

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sm1 = SubMod()
                self.sm2 = SubMod()

            def forward(self, x):
                return self.sm1(x) + self.sm2(x)

        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook

        with (
            install_guard_manager_testing_hook(hook_flag_on),
            dc.patch(inline_inbuilt_nn_modules=True),
        ):
            mod = Mod()
            mod(torch.randn(2, 2))

        with (
            install_guard_manager_testing_hook(hook_flag_off),
            dc.patch(inline_inbuilt_nn_modules=False),
        ):
            mod = Mod()
            mod(torch.randn(2, 2))

    def test_automatic_dynamic_reduce_recompiles(self):
        # Test the counterfactual, lots of recompiles without this config
        def foo(x, y):
            return x * y

        def run_foo_6_times_and_count_recompiles(dynamic=None):
            cnt = torch._dynamo.testing.CompileCounter()

            x = torch.randn([2])
            y = torch.randn([2])
            opt = torch.compile(foo, backend=cnt, dynamic=dynamic)
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

            opt = torch.compile(foo, backend=cnt, fullgraph=True)

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
            opt = torch.compile(foo, backend=cnt)
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
        compiled_foo = torch.compile(foo, backend=cnt, fullgraph=True)

        x = torch.randn([3])
        y = torch.randn([3])
        z = torch.randn([3])
        cmp_result = compiled_foo(
            x.detach().clone(), y.detach().clone(), z.detach().clone()
        )
        eager_result = foo(x.detach().clone(), y.detach().clone(), z.detach().clone())
        self.assertEqual(cmp_result, eager_result)
        self.assertEqual(cnt.frame_count, 1)

        cmp_result = compiled_foo(
            z.detach().clone(), y.detach().clone(), x.detach().clone()
        )
        eager_result = foo(z.detach().clone(), y.detach().clone(), x.detach().clone())
        self.assertEqual(cmp_result, eager_result)
        # No recompile, alias preserved
        self.assertEqual(cnt.frame_count, 1)

        x_clone = x.detach().clone()
        cmp_result = compiled_foo(x_clone, y.detach().clone(), x_clone)
        x_clone = x.detach().clone()
        eager_result = compiled_foo(x_clone, y.detach().clone(), x_clone)
        self.assertEqual(cmp_result, eager_result)
        # Recompile, alias changed
        self.assertEqual(cnt.frame_count, 2)

    def test_object_alias_relation_guards_without_lambda(self):
        class Box:
            pass

        def foo(box_a, box_b, t):
            entries = {box_a, box_b}
            if len(entries) == 1:
                return t + 1
            return t - 1

        cnt = torch._dynamo.testing.CompileCounter()
        x = torch.tensor(0)

        with dc.patch(use_lamba_guard_for_object_aliasing=False):
            compiled = torch.compile(foo, backend=cnt, fullgraph=True)

            shared = Box()
            res_alias = compiled(shared, shared, x)
            self.assertEqual(res_alias.item(), 1)

            res_unique = compiled(Box(), Box(), x)
            self.assertEqual(res_unique.item(), -1)
            self.assertEqual(cnt.frame_count, 2)

        torch._dynamo.reset()

    def test_aliasing_guard_failures_with_globals(self):
        g1 = torch.randn([3])
        g2 = torch.randn([3])

        def foo(a):
            a.add_(g1)
            return g2 + 1

        cnt = torch._dynamo.testing.CompileCounter()
        compiled_foo = torch.compile(foo, backend=cnt, fullgraph=True)

        z = torch.randn([3])
        cmp_result = compiled_foo(z.detach().clone())
        eager_result = foo(z.detach().clone())
        self.assertEqual(cmp_result, eager_result)
        self.assertEqual(cnt.frame_count, 1)

        g1 = g1.detach().clone()
        cmp_result = compiled_foo(g1)
        g1 = g1.detach().clone()
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

            opt = torch.compile(foo, backend=cnt, fullgraph=True)

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

    @patch.object(torch._dynamo.config, "recompile_limit", 2)
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

    @patch.object(torch._dynamo.config, "recompile_limit", 2)
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

    @torch._dynamo.config.patch(automatic_dynamic_shapes_mark_as="unbacked")
    def test_automatic_dynamic_shapes_mark_as_unbacked(self):
        counter = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=counter)
        def f(x):
            return x * x

        f(torch.randn(3))
        f(torch.randn(2))
        f(torch.randn(1))
        f(torch.randn(0))

        self.assertEqual(counter.frame_count, 2)  # not three or four!

    def test_ambient_autocast_recompile(self):
        weights = torch.randn(10, 10)
        counter = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        @torch.compile(backend=counter, fullgraph=True)
        def fn(x):
            return torch.mm(x, weights)

        x = torch.randn(1, 10)

        self.assertEqual(fn(x).dtype, torch.float32)

        with torch.autocast("cpu", torch.float16):
            self.assertEqual(fn(x).dtype, torch.float16)

        with torch.autocast("cpu", torch.bfloat16):
            self.assertEqual(fn(x).dtype, torch.bfloat16)

        # should recompile each time
        self.assertEqual(counter.frame_count, 3)

    def test_autocast_constant_fold(self):
        # test that constant-folded autocast functions
        # work properly - it should work if the global autocast
        # state is guarded.

        weights = torch.randn(10, 10)
        counter = torch._dynamo.testing.CompileCounterWithBackend("eager")

        def fn(x):
            if torch.get_autocast_dtype("cpu") == torch.float16:
                x = x + 1
            else:
                x = x - 1
            return torch.mm(x, weights)

        opt_fn = torch.compile(fn, backend=counter, fullgraph=True)

        x = torch.randn(1, 10)

        with torch.autocast("cpu", torch.float16):
            self.assertEqual(fn(x), opt_fn(x))

        with torch.autocast("cpu", torch.bfloat16):
            self.assertEqual(fn(x), opt_fn(x))

        self.assertEqual(counter.frame_count, 2)

    def test_dunder_call_recompile(self):
        class Foo:
            def __call__(self, x):
                return x + 1

        counter = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=counter)
        def f(x, foo):
            return foo(x)

        x = torch.ones(2)
        foo1 = Foo()
        foo2 = Foo()

        # no recompilation
        f(x, foo1)
        f(x, foo2)
        self.assertEqual(counter.frame_count, 1)

        # one recompilation
        Foo.__call__ = lambda self, x: x + 2
        f(x, foo1)
        self.assertEqual(counter.frame_count, 2)

    def test_no_recompile_over_unused_objects(self):
        # This is a regression test case that imitates
        # https://github.com/city96/ComfyUI-GGUF/blob/47bec6147569a138dd30ad3e14f190a36a3be456/ops.py#L169-L182
        counter = torch._dynamo.testing.CompileCounter()

        def f(x, key, patches):
            return x * x + 1

        @torch.compile(backend=counter, fullgraph=True)
        def apply_patches(f, x, keys):
            patches = []
            for key, patch in keys:  # noqa: F402
                patches.append(patch)
            x = f(x, key, patches)
            return x

        # no recompilation
        x = torch.rand(10)
        apply_patches(f, x, [("a", 1), ("b", 2)])
        self.assertEqual(counter.frame_count, 1)
        apply_patches(f, x, [("c", 3), ("d", 4)])
        self.assertEqual(counter.frame_count, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
