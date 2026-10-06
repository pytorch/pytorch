# Owner(s): ["module: dynamo"]
"""
Tests that torch.compile preserves the IndexError exception contract for
out-of-range dimension/index arguments, matching eager-mode behavior.
"""

import contextlib

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)


OPS = [
    ("softmax", lambda t: torch.softmax(t, dim=10)),
    ("squeeze", lambda t: torch.squeeze(t, 99)),
    ("unsqueeze", lambda t: torch.unsqueeze(t, 99)),
    ("amax", lambda t: torch.amax(t, dim=99)),
    ("argmax", lambda t: torch.argmax(t, dim=99)),
    ("flip", lambda t: torch.flip(t, dims=[99])),
    ("cat", lambda t: torch.cat([t], dim=99)),
    ("stack", lambda t: torch.stack([t], dim=99)),
]


# Ops whose out-of-range argument is rejected during fake tensor propagation.
FAKE_PROP_OPS = {
    "getitem": lambda t: t[10],
    "select": lambda t: t.select(0, 10),
    "sum": lambda t: t.sum(dim=5),
    "transpose": lambda t: t.transpose(0, 5),
}
OUT_OF_RANGE_MSG = "out of (range|bounds)"


class TestIndexErrorContract(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def _make_tensor(self):
        return torch.randn(4)

    def test_compile_preserves_index_error(self):
        t = self._make_tensor()
        for name, op in OPS:
            with self.subTest(op=name):
                compiled = torch.compile(op, backend="eager")
                torch._dynamo.reset()
                with self.assertRaises(IndexError):
                    compiled(t)

    def test_size_negative_out_of_range(self):
        t = self._make_tensor()
        with self.assertRaises(IndexError):
            t.size(-99)

        compiled = torch.compile(lambda x: x.size(-99), backend="eager")
        torch._dynamo.reset()
        with self.assertRaises(IndexError):
            compiled(t)


class TestConstantFoldedExceptionContract(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_constant_folded_exception_messages(self):
        def fn(kind):
            try:
                if kind == "index":
                    "abc".index("z")
                elif kind == "join":
                    ".".join(["a", 3])
                elif kind == "encode":
                    "\xe9".encode("ascii")
                elif kind == "format":
                    return f"{1:,,}"
                elif kind == "int":
                    return int(float("inf"))
                elif kind == "float":
                    return float(10**400)
            except UnicodeEncodeError as e:
                # Its non-string args must not crash the handler; str() of
                # Unicode errors is not modelled yet.
                return type(e).__name__, len(e.args)
            except (TypeError, ValueError, OverflowError) as e:
                return str(e)
            return "no error"

        for kind in ("index", "join", "encode", "format", "int", "float"):
            with self.subTest(kind=kind):
                torch._dynamo.reset()
                self.assertEqual(
                    torch.compile(fn, backend="eager", fullgraph=True)(kind), fn(kind)
                )

    def test_constant_folded_str_format_attribute_error(self):
        def fn():
            try:
                "{0.foo}".format(1)  # noqa: UP030, UP032
            except AttributeError as e:
                return str(e)
            return "no error"

        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(), fn())


@instantiate_parametrized_tests
class TestIndexErrorUserHandlers(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def _check(self, fn, *args):
        expected = fn(*args)
        torch._dynamo.reset()
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(*args), expected)

    @parametrize("op_name", list(FAKE_PROP_OPS))
    def test_except_index_error(self, op_name):
        op = FAKE_PROP_OPS[op_name]

        def fn(x):
            try:
                return op(x)
            except IndexError as e:
                return str(e)

        self._check(fn, torch.ones(3, 4))

    @parametrize("exc_type", [Exception, LookupError], name_fn=lambda t: t.__name__)
    def test_except_base_classes(self, exc_type):
        def fn(x):
            try:
                return x[10]
            except exc_type as e:
                return type(e).__name__

        self._check(fn, torch.ones(3, 4))

    def test_unused_result(self):
        def fn(x):
            try:
                x.sum(dim=5)
            except IndexError:
                return x + 1
            return x

        self._check(fn, torch.ones(3, 4))

    def test_suppress(self):
        def fn(x):
            with contextlib.suppress(IndexError):
                return x.sum(dim=5)
            return x + 1

        self._check(fn, torch.ones(3, 4))

    def test_finally_then_except(self):
        def fn(x):
            log = []
            try:
                try:
                    return x.sum(dim=5)
                finally:
                    log.append("finally")
            except IndexError:
                log.append("except")
                return log

        self._check(fn, torch.ones(3, 4))

    def test_handler_in_caller_of_inlined_function(self):
        def inner(t):
            return t.sum(dim=6)

        def fn(x):
            try:
                return inner(x)
            except IndexError:
                return x + 1

        self._check(fn, torch.ones(3, 4))

    def test_unrelated_handler_does_not_catch(self):
        def fn(x):
            try:
                return x.sum(dim=5)
            except RuntimeError:
                return x + 1

        torch._dynamo.reset()
        compiled = torch.compile(fn, backend="eager")
        with self.assertRaisesRegex(IndexError, "Dimension out of range"):
            compiled(torch.ones(3, 4))

    def test_unrelated_handler_fullgraph_is_unsupported(self):
        # Like any other observed exception that is re-raised out of a handler,
        # this cannot be raised as-is from the compiled region under fullgraph.
        def fn(x):
            try:
                return x.sum(dim=5)
            except RuntimeError:
                return x + 1

        torch._dynamo.reset()
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Observed exception"
        ):
            compiled(torch.ones(3, 4))

    @parametrize("op_name", list(FAKE_PROP_OPS))
    @parametrize("fullgraph", [False, True])
    def test_unhandled_preserves_index_error(self, op_name, fullgraph):
        torch._dynamo.reset()
        compiled = torch.compile(
            FAKE_PROP_OPS[op_name], backend="eager", fullgraph=fullgraph
        )
        with self.assertRaisesRegex(IndexError, OUT_OF_RANGE_MSG):
            compiled(torch.ones(3, 4))


if __name__ == "__main__":
    run_tests()
