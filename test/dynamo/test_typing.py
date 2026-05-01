# Owner(s): ["module: dynamo"]
import collections
import typing
import unittest
from typing_extensions import TypeVar

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import same


T = TypeVar("T")


class TypingTests(TestCase):
    def test_typing_typevar(self):
        def fn(x):
            def sumt(y: torch.Tensor) -> torch.Tensor:
                return torch.sum(y)

            def foo(c: typing.Callable[[T], T], y: T) -> T:
                return c(y)

            return foo(sumt, x)

        x = torch.randn(3)
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 1)

    def test_typing_union_and_optional(self):
        def fn(x):
            a = torch.jit.annotate(dict[str, torch.Tensor | None], {})
            b = torch.jit.annotate(dict[str, torch.Tensor | None], {})
            return a, b, x + 1

        x = torch.randn(3)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=False)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_typing_union_new_syntax(self):
        def fn(x):
            def inner1(y: torch.Tensor | None):
                return y

            def inner2(y: None | torch.Tensor):
                return y

            def inner3(y: torch.Tensor | list[int]):
                return y

            return x + 1

        torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(3))

    @unittest.expectedFailure
    def test_typing_union_new_syntax_reconstruct(self):
        def fn(x):
            return (
                x + 1,
                torch.Tensor | None,
                None | torch.Tensor,
                torch.Tensor | list[int],
            )

        torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(3))

    def test_typing_variable_isinstance(self):
        def fn(x, m):
            if isinstance(m, typing.Mapping):
                return x + 1
            else:
                return x - 1

        x = torch.randn(2, 3)
        m = {"x": torch.randn(3)}
        ref = fn(x, m)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, m)
        self.assertTrue(torch.allclose(ref, res))

    def test_typing_dict(self):
        def fn(d):
            return d[T]

        d = {T: torch.randn(3)}
        r1 = fn(d)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        r2 = opt_fn(d)
        self.assertEqual(r1, r2)

    def test_construct_generic(self):
        class C(typing.Generic[T]):
            pass

        def f(x):
            c = C[int]()
            if c is not None:
                return x.sin()

        x = torch.randn(3, 3)
        y = f(x)
        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        opt_y = opt_f(x)
        self.assertEqual(y, opt_y)

    def test_union_equality(self):
        class C:
            pass

        def f(x):
            t = typing.Union[C, int]  # noqa: UP007
            y = x * 2
            y += t == typing.Union[C, int]  # noqa: UP007
            y += t == C | int
            return y.sin()

        x = torch.randn(3, 3)
        y = f(x)
        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        opt_y = opt_f(x)
        self.assertEqual(y, opt_y)

    def test_type_int(self):
        def f(x):
            return x.sin(), type[int]

        x = torch.randn(3, 3)
        y = f(x)
        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        opt_y = opt_f(x)
        self.assertEqual(y, opt_y)

    def test_generic_any(self):
        def f(x):
            return x.sin(), list[typing.Any]

        x = torch.randn(3, 3)
        y = f(x)
        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        opt_y = opt_f(x)
        self.assertEqual(y, opt_y)

    def test_invalid_subscript(self):
        class C:
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def f(x, y):
            return x.sin(), y[int]

        x = torch.randn(3, 3)
        for y in (C, C(), int, None):
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "is not subscriptable"
            ):
                f(x, y)

    def test_generic_multi_typevar(self):
        U = TypeVar("U")
        V = TypeVar("V", default=int)

        class C(typing.Generic[T, U, V]):
            pass

        def f(x):
            return x.sin(), C[int, int]

        x = torch.randn(3, 3)
        y = f(x)
        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        opt_y = opt_f(x)
        self.assertEqual(y[0], opt_y[0])

        def f(x):
            return x.sin(), C[int]

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "Too few arguments"):
            opt_f(x)

    def test_callable(self):
        def f(x):
            return x.sin(), collections.abc.Callable[[int], int]

        x = torch.randn(3, 3)
        y = f(x)
        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        opt_y = opt_f(x)
        self.assertEqual(y, opt_y)


if __name__ == "__main__":
    run_tests()
