# Owner(s): ["module: unknown"]

import torch
from torch.testing._internal.common_utils import TestCase, run_tests, skipIfTorchDynamo, HardwareClassification

# End-to-end tests of features in native_functions.yaml


class FloatListWrapperModule(torch.nn.Module):
    def forward(self, values, incr: list[float] | None):
        return torch._C._nn._test_optional_floatlist(values, incr)


class IntListWrapperModule(torch.nn.Module):
    def forward(self, values, incr: list[int] | None):
        return torch._C._nn._test_optional_intlist(values, incr)


class TestNativeFunctions(TestCase):
    hw_classification = HardwareClassification.GENERIC

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

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_optional_floatlist(self):
        self.do_test_optional_floatlist_with_module(FloatListWrapperModule())

    def test_optional_floatlist_invalid(self):
        with self.assertRaisesRegex(TypeError, "must be tuple of floats, not list"):
            FloatListWrapperModule()(torch.zeros(1), ["hi"])

        with self.assertRaisesRegex(TypeError, "must be .* Tensor"):
            FloatListWrapperModule()(torch.zeros(1), torch.zeros(1))

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

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_optional_intlist(self):
        self.do_test_optional_intlist_with_module(IntListWrapperModule())

    def test_optional_intlist_invalid(self):
        with self.assertRaisesRegex(TypeError, "must be .* but found"):
            IntListWrapperModule()(torch.zeros(1), [0.5])

        with self.assertRaisesRegex(TypeError, "must be .* Tensor"):
            IntListWrapperModule()(torch.zeros(1), torch.zeros(1))

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

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_optional_filled_intlist(self):

        def f(n: int):
            x = torch._C._nn._test_optional_filled_intlist(torch.tensor([1, 1], dtype=torch.int), (n, n))
            y = torch._C._nn._test_optional_filled_intlist(torch.tensor([1, 1], dtype=torch.int), n)
            return x, y

        # eager
        returned = f(10)
        self.assertEqual(returned[0], returned[1])
        self.do_test_optional_filled_intlist_with_module(
            torch._C._nn._test_optional_filled_intlist
        )

    def test_string_defaults(self):
        dummy = torch.rand(1)
        fn = torch._C._nn._test_string_default
        fn(dummy)

        with self.assertRaisesRegex(RuntimeError, "A"):
            fn(dummy, a="")

        with self.assertRaisesRegex(RuntimeError, "B"):
            fn(dummy, b="")


if __name__ == '__main__':
    run_tests()
