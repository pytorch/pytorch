# Owner(s): ["module: dynamo"]

import torch
from torch._dynamo.test_case import run_tests, TestCase


def _compile_fullgraph(fn):
    return torch.compile(fn, fullgraph=True, backend="eager")


class TestTorchCheck(TestCase):
    def test_check_compiles_when_predicate_true_and_message_has_no_closure(self):
        @_compile_fullgraph
        def f(x):
            torch._check(x.shape[0] > 3, lambda: "Shape is not greater than 3")
            return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_compiles_when_predicate_true_constant_and_message_has_no_closure(
        self,
    ):
        @_compile_fullgraph
        def f(x):
            torch._check(x.shape[0] > 3, lambda: "Shape is not greater than 3")
            return x + 1

        x = torch.randn(4)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_compiles_when_predicate_true_constant_and_message_None(self):
        @_compile_fullgraph
        def f(x):
            torch._check(x.shape[0] > 3)
            return x + 1

        x = torch.randn(4)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_compiles_when_predicate_true_and_message_None(self):
        @_compile_fullgraph
        def f(x):
            torch._check(x.shape[0] > 3)
            return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_raises_at_runtime_when_predicate_false_and_message_None(self):
        @_compile_fullgraph
        def f(x):
            torch._check(x.shape[0] > 3)
            return x + 1

        x = torch.randn(3)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(RuntimeError, None):
            f(x)

    def test_check_raises_at_runtime_when_predicate_false_constant_and_message_None(
        self,
    ):
        @_compile_fullgraph
        def f(x):
            torch._check(x.shape[0] > 3)
            return x + 1

        x = torch.randn(3)

        with self.assertRaisesRegex(RuntimeError, None):
            f(x)

    def test_check_raises_at_runtime_when_predicate_false_and_message_has_no_closure(
        self,
    ):
        @_compile_fullgraph
        def f(x):
            torch._check(x.shape[0] > 3, lambda: "Shape is not greater than 3")
            return x + 1

        x = torch.randn(3)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(RuntimeError, "Shape is not greater than 3"):
            f(x)

    def test_check_raises_at_runtime_when_predicate_false_constant_and_message_has_no_closure(
        self,
    ):
        @_compile_fullgraph
        def f(x):
            torch._check(x.shape[0] > 3, lambda: "Shape is not greater than 3")
            return x + 1

        x = torch.randn(3)

        with self.assertRaisesRegex(RuntimeError, "Shape is not greater than 3"):
            f(x)

    def test_check_assert_error_at_runtime_when_predicate_false_and_message_has_closure(
        self,
    ):
        @_compile_fullgraph
        def f(x):
            torch._check(x.shape[0] > 3, lambda: f"{x.shape[0]} is not greater than 3")
            return x + 1

        x = torch.randn(3)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaises(AssertionError):
            f(x)

    def test_check_assert_error_at_runtime_when_predicate_true_and_message_has_closure(
        self,
    ):
        @_compile_fullgraph
        def f(x):
            torch._check(x.shape[0] > 3, lambda: f"{x.shape[0]} is not greater than 3")
            return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaises(AssertionError):
            f(x)


if __name__ == "__main__":
    run_tests()
