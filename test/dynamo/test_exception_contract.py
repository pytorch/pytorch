# Owner(s): ["module: dynamo"]
"""
Tests that torch.compile preserves the IndexError exception contract for
out-of-range dimension/index arguments, matching eager-mode behavior.
"""

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import HardwareClassification


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


if __name__ == "__main__":
    run_tests()
