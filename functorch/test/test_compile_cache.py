# Owner(s): ["module: functorch"]

import torch

import functorch
from torch.testing._internal.common_utils import run_tests, TestCase, IS_WINDOWS
import unittest

from functorch.compile import aot_function, nop


class TestCompileCache(TestCase):
    def check(self, a, b, aot_fn, fn):
        a_clone = a.clone().detach().requires_grad_(True)
        b_clone = b.clone().detach().requires_grad_(True)
        ref = fn(a, b)
        ref.sum().backward()

        res = aot_fn(a_clone, b_clone)
        res.sum().backward()
        assert torch.allclose(res, ref)
        assert torch.allclose(a.grad, a_clone.grad)
        assert torch.allclose(b.grad, b_clone.grad)

    def test_recompilation_on_broadcast(self):
        def fn(x, bias):
            return x + bias

        for hasher_type in ["DynamicShapeHasher", "StaticShapeHasher"]:
            functorch.compile.clear_compile_cache()
            start_num_recomps = functorch.compile.num_of_recompilations()
            aot_autograd_fn = aot_function(fn, nop, nop, hasher_type=hasher_type)

            a = torch.randn(10, 20, requires_grad=True)
            b = torch.randn(20, requires_grad=True)
            self.check(a, b, aot_autograd_fn, fn)

            a = torch.randn(10, 20, requires_grad=True)
            b = torch.randn(10, 20, requires_grad=True)
            self.check(a, b, aot_autograd_fn, fn)

            end_num_recomps = functorch.compile.num_of_recompilations()

            total_recomps = end_num_recomps - start_num_recomps
            assert total_recomps == 2

    def test_compilation_for_dynamic_shape(self):
        def fn(x, bias):
            return x + bias

        for hasher_type in ["DynamicShapeHasher", "StaticShapeHasher"]:
            functorch.compile.clear_compile_cache()
            start_num_recomps = functorch.compile.num_of_recompilations()
            aot_autograd_fn = aot_function(fn, nop, nop, hasher_type=hasher_type)

            for s in range(10, 20):
                a = torch.randn(s, requires_grad=True)
                b = torch.randn(s, requires_grad=True)
                self.check(a, b, aot_autograd_fn, fn)

            for s in range(10, 20):
                a = torch.randn(s, requires_grad=True)
                b = torch.randn(s, requires_grad=True)
                self.check(a, b, aot_autograd_fn, fn)

            end_num_recomps = functorch.compile.num_of_recompilations()

            total_recomps = end_num_recomps - start_num_recomps
            if hasher_type == "DynamicShapeHasher":
                assert total_recomps == 1
            elif hasher_type == "StaticShapeHasher":
                assert total_recomps == 10

            for s in range(10, 20):
                a = torch.randn(s, s, requires_grad=True)
                b = torch.randn(s, s, requires_grad=True)
                self.check(a, b, aot_autograd_fn, fn)

            end_num_recomps = functorch.compile.num_of_recompilations()

            total_recomps = end_num_recomps - start_num_recomps
            if hasher_type == "DynamicShapeHasher":
                assert total_recomps == 2
            elif hasher_type == "StaticShapeHasher":
                assert total_recomps == 20

    def test_global_cache_no_recompilations(self):
        def f(x, bias):
            return x + bias

        def g(x, bias):
            return aot_function(f, nop, nop, hasher_type="DynamicShapeHasher")(x, bias)

        start_num_recomps = functorch.compile.num_of_recompilations()
        for _ in range(10):
            a = torch.randn(10, 20, requires_grad=True)
            b = torch.randn(10, 20, requires_grad=True)
            self.check(a, b, g, f)

        end_num_recomps = functorch.compile.num_of_recompilations()
        total_recomps = end_num_recomps - start_num_recomps
        assert total_recomps == 1

    def test_multiple_functions(self):
        def f(x, bias):
            return x + bias

        def g(x, y):
            return x * y

        for hasher_type in ["DynamicShapeHasher", "StaticShapeHasher"]:
            functorch.compile.clear_compile_cache()
            aot_autograd_f = aot_function(f, nop, nop, hasher_type=hasher_type)
            aot_autograd_g = aot_function(g, nop, nop, hasher_type=hasher_type)

            start_num_recomps = functorch.compile.num_of_recompilations()
            a = torch.randn(10, requires_grad=True)
            b = torch.randn(10, requires_grad=True)
            self.check(a, b, aot_autograd_f, f)

            a = torch.randn(10, requires_grad=True)
            b = torch.randn(10, requires_grad=True)
            self.check(a, b, aot_autograd_g, g)

            end_num_recomps = functorch.compile.num_of_recompilations()
            total_recomps = end_num_recomps - start_num_recomps
            assert total_recomps == 2

            # Force recompilation for function f and check num of recompilations again
            a = torch.randn(10, 20, requires_grad=True)
            b = torch.randn(10, 20, requires_grad=True)
            self.check(a, b, aot_autograd_f, f)

            end_num_recomps = functorch.compile.num_of_recompilations()
            total_recomps = end_num_recomps - start_num_recomps
            assert total_recomps == 3

    def test_high_number_of_args(self):
        def f(*args):
            res = args[0]
            for arg in args:
                res = res * arg
            return res

        def check(args, aot_autograd_fn, fn):
            args_clone = [arg.clone().detach().requires_grad_(True) for arg in args]
            ref = fn(*args)
            ref.sum().backward()

            res = aot_autograd_fn(*args_clone)
            res.sum().backward()
            assert torch.allclose(res, ref)
            for (arg, arg_clone) in zip(args, args_clone):
                assert torch.allclose(arg.grad, arg_clone.grad)

        for hasher_type in ["DynamicShapeHasher", "StaticShapeHasher"]:
            functorch.compile.clear_compile_cache()

            aot_autograd_f = aot_function(f, nop, nop, hasher_type=hasher_type)

            args = [torch.randn(10, requires_grad=True) for _ in range(100)]
            check(args, aot_autograd_f, f)

    def test_multiple_compiler(self):
        def fn(x, bias):
            return x + bias

        def nop_duplicate(fx_g, _):
            return fx_g

        for hasher_type in ["DynamicShapeHasher", "StaticShapeHasher"]:
            functorch.compile.clear_compile_cache()
            start_num_recomps = functorch.compile.num_of_recompilations()
            nop_fn = aot_function(fn, nop, nop, hasher_type=hasher_type)
            nop_duplicate_fn = aot_function(
                fn, nop_duplicate, nop_duplicate, hasher_type=hasher_type
            )

            a = torch.randn(10, 20, requires_grad=True)
            b = torch.randn(20, requires_grad=True)
            nop_fn(a, b)
            nop_duplicate_fn(a, b)

            end_num_recomps = functorch.compile.num_of_recompilations()

            total_recomps = end_num_recomps - start_num_recomps
            assert total_recomps == 2


@unittest.skipIf(IS_WINDOWS, 'test broken on windows')
class TestCompileCacheStaticArgs(TestCase):
    def check(self, a, b, aot_autograd_fn, fn):
        a_clone = a.clone().detach().requires_grad_(True)
        ref = fn(a, b)
        ref.sum().backward()

        res = aot_autograd_fn(a_clone, b)
        res.sum().backward()
        assert torch.allclose(res, ref)
        assert torch.allclose(a.grad, a_clone.grad)

    def test_failure(self):
        # Test that not setting up static_argnums should raise exception
        def fn(x, p):
            return x * p

        aot_autograd_f = aot_function(fn, nop, nop)

        a = torch.randn(2, 2, requires_grad=True)
        b = 2
        try:
            # Since b is not marked as static, it should raise exception
            aot_autograd_f(a, b)
            raise AssertionError()
        except RuntimeError:
            pass

    def test_simple(self):
        def fn(x, static_arg):
            return x * static_arg

        functorch.compile.clear_compile_cache()

        start_num_recomps = functorch.compile.num_of_recompilations()

        aot_autograd_f = aot_function(fn, nop, nop, static_argnums=1)

        a = torch.randn(2, 2, requires_grad=True)
        b = 2
        self.check(a, b, aot_autograd_f, fn)

        # Same type of args, so no recompilation
        a = torch.randn(2, 2, requires_grad=True)
        b = 2
        self.check(a, b, aot_autograd_f, fn)

        # Trigger recompilation
        a = torch.randn(2, 2, requires_grad=True)
        b = 3
        self.check(a, b, aot_autograd_f, fn)

        end_num_recomps = functorch.compile.num_of_recompilations()

        total_recomps = end_num_recomps - start_num_recomps
        assert total_recomps == 2

    def test_static_arg_before_tensor_arg(self):
        def fn(static_arg, x):
            return static_arg - x

        def check(a, b, aot_autograd_fn, fn):
            b_clone = b.clone().detach().requires_grad_(True)
            ref = fn(a, b)
            ref.sum().backward()

            res = aot_autograd_fn(a, b_clone)
            res.sum().backward()
            assert torch.allclose(res, ref)
            assert torch.allclose(b.grad, b_clone.grad)

        functorch.compile.clear_compile_cache()

        start_num_recomps = functorch.compile.num_of_recompilations()

        aot_autograd_f = aot_function(fn, nop, nop, static_argnums=0)

        a = 2
        b = torch.randn(2, 2, requires_grad=True)
        check(a, b, aot_autograd_f, fn)

        a = 3
        b = torch.randn(2, 2, requires_grad=True)
        check(a, b, aot_autograd_f, fn)

        end_num_recomps = functorch.compile.num_of_recompilations()

        total_recomps = end_num_recomps - start_num_recomps
        assert total_recomps == 2

    def test_interleaved_static_args(self):
        def fn(static_arg1, x, static_arg2):
            return static_arg1 - x - static_arg2

        def check(a, b, c, aot_autograd_fn, fn):
            b_clone = b.clone().detach().requires_grad_(True)
            ref = fn(a, b, c)
            ref.sum().backward()

            res = aot_autograd_fn(a, b_clone, c)
            res.sum().backward()
            assert torch.allclose(res, ref)
            assert torch.allclose(b.grad, b_clone.grad)

        functorch.compile.clear_compile_cache()

        start_num_recomps = functorch.compile.num_of_recompilations()

        aot_autograd_f = aot_function(fn, nop, nop, static_argnums=(0, 2))

        a = 2
        b = torch.randn(2, 2, requires_grad=True)
        c = 0.1
        check(a, b, c, aot_autograd_f, fn)

        a = 3
        b = torch.randn(2, 2, requires_grad=True)
        c = 0.1
        check(a, b, c, aot_autograd_f, fn)

        end_num_recomps = functorch.compile.num_of_recompilations()

        total_recomps = end_num_recomps - start_num_recomps
        assert total_recomps == 2

    def test_dropout(self):
        def fn(x, prob):
            return torch.nn.functional.dropout(x, p=prob)

        functorch.compile.clear_compile_cache()

        start_num_recomps = functorch.compile.num_of_recompilations()

        aot_autograd_f = aot_function(fn, nop, nop, static_argnums=[1])

        a = torch.randn(2, 2, requires_grad=True)
        b = 0.3
        aot_autograd_f(a, b)

        # Setting the prob to 0. This should cause recompilation.
        a = torch.randn(2, 2, requires_grad=True)
        b = 0
        self.check(a, b, aot_autograd_f, fn)

        end_num_recomps = functorch.compile.num_of_recompilations()

        total_recomps = end_num_recomps - start_num_recomps
        assert total_recomps == 2

    def test_if_condition(self):
        def fn(x, state: bool):
            if state:
                return torch.sin(x)
            else:
                return torch.cos(x)

        functorch.compile.clear_compile_cache()

        start_num_recomps = functorch.compile.num_of_recompilations()

        aot_autograd_f = aot_function(fn, nop, nop, static_argnums=[1])

        a = torch.randn(2, 2, requires_grad=True)
        b = True
        self.check(a, b, aot_autograd_f, fn)

        a = torch.randn(2, 2, requires_grad=True)
        b = True
        self.check(a, b, aot_autograd_f, fn)

        a = torch.randn(2, 2, requires_grad=True)
        b = False
        self.check(a, b, aot_autograd_f, fn)

        end_num_recomps = functorch.compile.num_of_recompilations()

        total_recomps = end_num_recomps - start_num_recomps
        assert total_recomps == 2

    def test_custom(self):
        class Record:
            def __init__(self, name, multiplier):
                self.name = name
                self.multiplier = multiplier

            def __eq__(self, other):
                return self.name == other.name and self.multiplier == other.multiplier

            def __hash__(self):
                return hash((self.name, self.multiplier))

        def fn(x, record):
            return x * record.multiplier

        functorch.compile.clear_compile_cache()

        start_num_recomps = functorch.compile.num_of_recompilations()

        aot_autograd_f = aot_function(fn, nop, nop, static_argnums=[1])

        a = torch.randn(2, 2, requires_grad=True)
        b = Record("Foo", 0.5)
        self.check(a, b, aot_autograd_f, fn)

        a = torch.randn(2, 2, requires_grad=True)
        b = Record("Bar", 10.2)
        self.check(a, b, aot_autograd_f, fn)

        end_num_recomps = functorch.compile.num_of_recompilations()

        total_recomps = end_num_recomps - start_num_recomps
        assert total_recomps == 2

    def test_tuple(self):
        def fn(a_tuple, static_arg):
            return torch.sin(a_tuple[0]) - a_tuple[1] - static_arg

        def check(a_tuple, b, aot_autograd_fn, fn):
            a0 = a_tuple[0]
            a1 = a_tuple[1]

            a0_clone = a0.clone().detach().requires_grad_(True)
            a1_clone = a1.clone().detach().requires_grad_(True)
            ref = fn(a, b)
            ref.sum().backward()

            res = aot_autograd_fn((a0_clone, a1_clone), b)
            res.sum().backward()
            assert torch.allclose(res, ref)
            assert torch.allclose(a0.grad, a0_clone.grad)
            assert torch.allclose(a1.grad, a1_clone.grad)

        functorch.compile.clear_compile_cache()

        start_num_recomps = functorch.compile.num_of_recompilations()

        aot_autograd_f = aot_function(fn, nop, nop, static_argnums=(1,))

        a = (
            torch.randn(2, 2, requires_grad=True),
            torch.randn(2, 2, requires_grad=True),
        )
        b = 0.1
        check(a, b, aot_autograd_f, fn)

        a = (
            torch.randn(2, 2, requires_grad=True),
            torch.randn(2, 2, requires_grad=True),
        )
        b = 1
        check(a, b, aot_autograd_f, fn)

        end_num_recomps = functorch.compile.num_of_recompilations()

        total_recomps = end_num_recomps - start_num_recomps
        assert total_recomps == 2

    def test_tuple_with_first_arg_as_static(self):
        def fn(static_arg, a_tuple):
            return torch.sin(a_tuple[0]) - a_tuple[1] - static_arg

        def check(a, b_tuple, aot_autograd_fn, fn):
            b0 = b_tuple[0]
            b1 = b_tuple[1]

            b0_clone = b0.clone().detach().requires_grad_(True)
            b1_clone = b1.clone().detach().requires_grad_(True)
            ref = fn(a, b_tuple)
            ref.sum().backward()

            res = aot_autograd_fn(a, (b0_clone, b1_clone))
            res.sum().backward()
            assert torch.allclose(res, ref)
            assert torch.allclose(b0.grad, b0_clone.grad)
            assert torch.allclose(b1.grad, b1_clone.grad)

        functorch.compile.clear_compile_cache()

        start_num_recomps = functorch.compile.num_of_recompilations()

        aot_autograd_f = aot_function(fn, nop, nop, static_argnums=(0,))

        a = 0.1
        b = (
            torch.randn(2, 2, requires_grad=True),
            torch.randn(2, 2, requires_grad=True),
        )
        check(a, b, aot_autograd_f, fn)

        a = 1
        b = (
            torch.randn(2, 2, requires_grad=True),
            torch.randn(2, 2, requires_grad=True),
        )
        check(a, b, aot_autograd_f, fn)

        end_num_recomps = functorch.compile.num_of_recompilations()

        total_recomps = end_num_recomps - start_num_recomps
        assert total_recomps == 2

    def test_dict(self):
        def fn(a_dict, static_arg):
            return torch.sin(a_dict["foo"]) - a_dict["bar"] - static_arg

        def check(a_dict, b, aot_autograd_fn, fn):

            a0 = a_dict["foo"]
            a1 = a_dict["bar"]

            a0_clone = a0.clone().detach().requires_grad_(True)
            a1_clone = a1.clone().detach().requires_grad_(True)
            ref = fn(a_dict, b)
            ref.sum().backward()

            a_clone = {}
            a_clone["foo"] = a0_clone
            a_clone["bar"] = a1_clone
            res = aot_autograd_fn(a_clone, b)
            res.sum().backward()
            assert torch.allclose(res, ref)
            assert torch.allclose(a0.grad, a0_clone.grad)
            assert torch.allclose(a1.grad, a1_clone.grad)

        functorch.compile.clear_compile_cache()

        start_num_recomps = functorch.compile.num_of_recompilations()

        aot_autograd_f = aot_function(fn, nop, nop, static_argnums=(1,))

        a = {}
        a["foo"] = torch.zeros(2, 2, requires_grad=True)
        a["bar"] = torch.ones(2, 2, requires_grad=True)
        b = 0
        check(a, b, aot_autograd_f, fn)

        a = {}
        a["foo"] = torch.randn(2, 2, requires_grad=True)
        a["bar"] = torch.randn(2, 2, requires_grad=True)
        b = 0.2
        check(a, b, aot_autograd_f, fn)

        end_num_recomps = functorch.compile.num_of_recompilations()

        total_recomps = end_num_recomps - start_num_recomps
        assert total_recomps == 2

    def test_dict_with_static_arg_before_dict(self):
        def fn(static_arg, a_dict):
            return torch.sin(a_dict["foo"]) - a_dict["bar"] - static_arg

        def check(a, b_dict, aot_autograd_fn, fn):

            ref = fn(a, b_dict)
            res = aot_autograd_fn(a, b_dict)
            assert torch.allclose(res, ref)

            b0 = b_dict["foo"]
            b1 = b_dict["bar"]

            b0_clone = b0.clone().detach().requires_grad_(True)
            b1_clone = b1.clone().detach().requires_grad_(True)
            ref.sum().backward()

            b_clone = {}
            b_clone["foo"] = b0_clone
            b_clone["bar"] = b1_clone
            res = aot_autograd_fn(a, b_clone)
            res.sum().backward()
            assert torch.allclose(res, ref)
            assert torch.allclose(b0.grad, b0_clone.grad)
            assert torch.allclose(b1.grad, b1_clone.grad)

        functorch.compile.clear_compile_cache()

        start_num_recomps = functorch.compile.num_of_recompilations()

        aot_autograd_f = aot_function(fn, nop, nop, static_argnums=(0,))

        a = 0.1
        b = {}
        b["foo"] = torch.randn(2, 2, requires_grad=True)
        b["bar"] = torch.randn(2, 2, requires_grad=True)
        check(a, b, aot_autograd_f, fn)

        a = 0.2
        b = {}
        b["foo"] = torch.randn(2, 2, requires_grad=True)
        b["bar"] = torch.randn(2, 2, requires_grad=True)
        check(a, b, aot_autograd_f, fn)

        end_num_recomps = functorch.compile.num_of_recompilations()

        total_recomps = end_num_recomps - start_num_recomps
        assert total_recomps == 2

    def test_tuple_static_args(self):
        def fn(x, tuple_static_arg):
            return x * tuple_static_arg[0] * tuple_static_arg[1]

        functorch.compile.clear_compile_cache()

        start_num_recomps = functorch.compile.num_of_recompilations()

        aot_autograd_f = aot_function(fn, nop, nop, static_argnums=1)

        a = torch.randn(2, 2, requires_grad=True)
        b = (2, 3)
        self.check(a, b, aot_autograd_f, fn)

        # Same type of args, so no recompilation
        a = torch.randn(2, 2, requires_grad=True)
        b = (2, 3)
        self.check(a, b, aot_autograd_f, fn)

        # Trigger recompilation
        a = torch.randn(2, 2, requires_grad=True)
        b = (3, 4)
        self.check(a, b, aot_autograd_f, fn)

        end_num_recomps = functorch.compile.num_of_recompilations()

        total_recomps = end_num_recomps - start_num_recomps
        assert total_recomps == 2

    def test_arg_none(self):
        def check(a, b, c, aot_autograd_fn, fn):
            def cloner(x):
                if x is not None:
                    return x.clone().detach().requires_grad_(True)
                return None

            def check_grad(x, x_clone):
                if x is not None:
                    return torch.allclose(x.grad, x_clone.grad)
                return True

            ref = fn(a, b, c)
            res = aot_autograd_fn(a, b, c)
            assert torch.allclose(res, ref)

            a_clone = cloner(a)
            b_clone = cloner(b)
            c_clone = cloner(c)
            ref.sum().backward()
            res = aot_autograd_fn(a_clone, b_clone, c_clone)
            res.sum().backward()

            check_grad(a, a_clone)
            check_grad(b, b_clone)
            check_grad(c, c_clone)

        def fn(a, b, c):
            if a is None and b is None:
                return c
            elif a is None and c is None:
                return b
            elif b is None and c is None:
                return a
            elif a is None:
                return b + c
            elif b is None:
                return a + c
            elif c is None:
                return a + b
            return a + b + c

        functorch.compile.clear_compile_cache()

        start_num_recomps = functorch.compile.num_of_recompilations()

        aot_autograd_f = aot_function(fn, nop, nop)

        t1 = torch.randn(2, 2, requires_grad=True)
        check(t1, None, None, aot_autograd_f, fn)
        check(None, t1, None, aot_autograd_f, fn)
        check(None, None, t1, aot_autograd_f, fn)

        t2 = torch.randn(2, 2, requires_grad=True)
        check(t1, t2, None, aot_autograd_f, fn)
        check(t1, None, t2, aot_autograd_f, fn)
        check(None, t1, t2, aot_autograd_f, fn)

        t3 = torch.randn(2, 2, requires_grad=True)
        check(t1, t2, t3, aot_autograd_f, fn)

        # Same type of args, so no recompilation
        check(t1, t2, None, aot_autograd_f, fn)

        end_num_recomps = functorch.compile.num_of_recompilations()

        total_recomps = end_num_recomps - start_num_recomps
        assert total_recomps == 7


if __name__ == "__main__":
    run_tests()
