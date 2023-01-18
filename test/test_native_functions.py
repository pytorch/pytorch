# Owner(s): ["module: unknown"]

from typing import Optional, List
import torch
from torch.testing._internal.common_utils import TestCase, run_tests

# End-to-end tests of features in native_functions.yaml


class FloatListWrapperModule(torch.nn.Module):
    def forward(self, values, incr: Optional[List[float]]):
        return torch._C._nn._test_optional_floatlist(values, incr)


class IntListWrapperModule(torch.nn.Module):
    def forward(self, values, incr: Optional[List[int]]):
        return torch._C._nn._test_optional_intlist(values, incr)


class TestNativeFunctions(TestCase):

    def _lists_with_str(self):
        return [
            ("foo",),
            (2, "foo"),
            ("foo", 3),
            ["foo"],
            [2, "foo"],
            ["foo", 3],
            "foo",
        ]

    def _test_raises_str_typeerror(self, fn):
        for arg in self._lists_with_str():
            self.assertRaisesRegex(TypeError, "str", lambda: fn(arg))
            try:
                fn(arg)
            except TypeError as e:
                print(e)

    def test_symintlist_error(self):
        x = torch.randn(1)
        self._test_raises_str_typeerror(lambda arg: torch._C._nn.pad(x, arg))

    def test_vararg_symintlist_error(self):
        self._test_raises_str_typeerror(lambda arg: torch.rand(arg))
        self._test_raises_str_typeerror(lambda arg: torch.rand(*arg))

    def test_symintlist_error_with_overload_but_is_unique(self):
        x = torch.randn(1)
        y = torch.randn(1)
        self._test_raises_str_typeerror(lambda arg: x.set_(y, 0, arg))

    def test_symintlist_error_with_overload(self):
        x = torch.randn(1)
        self._test_raises_str_typeerror(lambda arg: x.view(arg))

    def test_intlist_error_with_overload(self):
        x = torch.randn(1)
        self._test_raises_str_typeerror(lambda arg: torch._C._nn.pad(x, arg))

    #
    # optional float list
    #

    def do_test_optional_floatlist_with_module(self, module):
        values = torch.tensor([1.5, 2.5], dtype=torch.float)

        returned = module(values, None)
        self.assertEqual(values, returned)
        # Make sure that it's an alias, indicating that the operator saw a nullopt.
        values[0] = 3.5
        self.assertEqual(values, returned)

        returned = module(values, [5.1, 4.1])
        self.assertEqual(values, torch.tensor([3.5, 2.5], dtype=torch.float))
        self.assertEqual(returned, torch.tensor([8.6, 6.6], dtype=torch.float))

    def trace_optional_floatlist(self, const):
        def wrapper(values):
            return torch._C._nn._test_optional_floatlist(values, const)
        return torch.jit.trace(wrapper, torch.tensor([1.5, 2.5], dtype=torch.float))

    def test_optional_floatlist(self):
        self.do_test_optional_floatlist_with_module(FloatListWrapperModule())
        self.do_test_optional_floatlist_with_module(torch.jit.script(FloatListWrapperModule()))

        traced_none = self.trace_optional_floatlist(None)
        traced_list = self.trace_optional_floatlist([5.1, 4.1])

        # Not really a module, just lets us use our two traced functions to handle
        # the specific cases of passing None and [5.1, 4.1].
        def fake_module(values, const):
            if const is None:
                return traced_none(values)
            if const == [5.1, 4.1]:
                return traced_list(values)
            raise Exception("Invalid argument")

        self.do_test_optional_floatlist_with_module(fake_module)

    def test_optional_floatlist_invalid(self):
        with self.assertRaisesRegex(TypeError, "must be tuple of floats, not list"):
            FloatListWrapperModule()(torch.zeros(1), ["hi"])

        with self.assertRaisesRegex(RuntimeError, "value of type .* instead found type"):
            torch.jit.script(FloatListWrapperModule())(torch.zeros(1), ["hi"])

        with self.assertRaisesRegex(TypeError, "must be .* Tensor"):
            FloatListWrapperModule()(torch.zeros(1), torch.zeros(1))

        with self.assertRaisesRegex(RuntimeError, "value of type .* instead found type"):
            torch.jit.script(FloatListWrapperModule())(torch.zeros(1), torch.zeros(1))

    #
    # optional int list
    #

    def do_test_optional_intlist_with_module(self, module):
        values = torch.tensor([1, 2], dtype=torch.int)

        returned = module(values, None)
        self.assertEqual(values, returned)
        # Make sure that it's an alias, indicating that the operator saw a nullopt.
        values[0] = 3
        self.assertEqual(values, returned)

        returned = module(values, [5, 4])
        self.assertEqual(values, torch.tensor([3, 2], dtype=torch.int))
        self.assertEqual(returned, torch.tensor([8, 6], dtype=torch.int))

    def trace_optional_intlist(self, const):
        def wrapper(values):
            return torch._C._nn._test_optional_intlist(values, const)
        return torch.jit.trace(wrapper, torch.tensor([1, 2], dtype=torch.int))

    def test_optional_intlist(self):
        self.do_test_optional_intlist_with_module(IntListWrapperModule())
        self.do_test_optional_intlist_with_module(torch.jit.script(IntListWrapperModule()))

        traced_none = self.trace_optional_intlist(None)
        traced_list = self.trace_optional_intlist([5, 4])

        # Not really a module, just lets us use our two traced functions to handle
        # the specific cases of passing None and [5, 4].
        def fake_module(values, const):
            if const is None:
                return traced_none(values)
            if const == [5, 4]:
                return traced_list(values)
            raise Exception("Invalid argument")

        self.do_test_optional_intlist_with_module(fake_module)

    def test_optional_intlist_invalid(self):
        with self.assertRaisesRegex(TypeError, "must be .* but found"):
            IntListWrapperModule()(torch.zeros(1), [0.5])

        with self.assertRaisesRegex(RuntimeError, "value of type .* instead found type"):
            torch.jit.script(IntListWrapperModule())(torch.zeros(1), [0.5])

        with self.assertRaisesRegex(TypeError, "must be .* Tensor"):
            IntListWrapperModule()(torch.zeros(1), torch.zeros(1))

        with self.assertRaisesRegex(RuntimeError, "value of type .* instead found type"):
            torch.jit.script(IntListWrapperModule())(torch.zeros(1), torch.zeros(1))

    #
    # optional filled int list
    #

    def do_test_optional_filled_intlist_with_module(self, module):
        values = torch.tensor([1, 2], dtype=torch.int)

        returned = module(values, None)
        self.assertEqual(values, returned)
        # Make sure that it's an alias, indicating that the operator saw a nullopt.
        values[0] = 3
        self.assertEqual(values, returned)

        returned = module(values, 10)
        self.assertEqual(values, torch.tensor([3, 2], dtype=torch.int))
        self.assertEqual(returned, torch.tensor([13, 12], dtype=torch.int))

    def trace_optional_filled_intlist(self, const):
        def wrapper(values):
            return torch._C._nn._test_optional_filled_intlist(values, const)
        return torch.jit.trace(wrapper, torch.tensor([1, 2], dtype=torch.int))

    def test_optional_filled_intlist(self):

        def f(n: int):
            x = torch._C._nn._test_optional_filled_intlist(torch.tensor([1, 1], dtype=torch.int), (n, n))
            y = torch._C._nn._test_optional_filled_intlist(torch.tensor([1, 1], dtype=torch.int), n)
            return x, y

        # eager
        returned = f(10)
        self.assertEqual(returned[0], returned[1])

        # scripted
        s = torch.jit.script(f)
        returned = s(10)
        self.assertEqual(returned[0], returned[1])

        # traced
        traced_none = self.trace_optional_filled_intlist(None)
        traced_int = self.trace_optional_filled_intlist(10)

        # Not really a module, just lets us use our two traced functions to handle
        # the specific cases of passing None and 10.
        def fake_module(values, const):
            if const is None:
                return traced_none(values)
            if const == 10:
                return traced_int(values)
            raise Exception("Invalid argument")

        self.do_test_optional_filled_intlist_with_module(fake_module)

    def test_string_defaults(self):
        dummy = torch.rand(1)
        fn = torch._C._nn._test_string_default
        fn(dummy)

        with self.assertRaisesRegex(RuntimeError, "A"):
            fn(dummy, a="")

        with self.assertRaisesRegex(RuntimeError, "B"):
            fn(dummy, b="")

        def f(x):
            torch._C._nn._test_string_default(x)
        scripted_fn = torch.jit.script(f)
        scripted_fn(dummy)


if __name__ == '__main__':
    run_tests()
