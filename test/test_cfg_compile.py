# Owner(s): ["module: dynamo"]

import dis
import importlib
import sys
import unittest
import warnings

import torch
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TEST_WITH_CROSSREF,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)


# These tests look at which kernels the dispatcher runs; the crossref and
# dynamo test configurations interpose on exactly that.
UNDISTURBED = not (TEST_WITH_CROSSREF or TEST_WITH_TORCHDYNAMO)

# The Rust runtime is only in builds made with USE_CFG_COMPILE=1, and only
# understands CPython 3.12 bytecode.
try:
    cfg_rs = importlib.import_module("torch._cfg_rs")
except ImportError:
    cfg_rs = None
HAS_RUNTIME = UNDISTURBED and cfg_rs is not None and sys.version_info[:2] == (3, 12)


@unittest.skipUnless(UNDISTURBED, "the test configuration changes dispatch")
class TestDispatchCapture(TestCase):
    """The dispatcher hook that cfg_compile records kernels with."""

    hw_classification = HardwareClassification.GENERIC

    def trace(self, fn):
        return torch._C._cfg_dispatch_trace(fn)[1]

    def test_call_and_redispatch_chain(self):
        x, y = torch.randn(3), torch.randn(3)
        self.assertEqual(
            self.trace(lambda: torch.add(x, y)),
            [
                ("aten::add.Tensor", "AutogradCPU", 0, False),
                ("aten::add.Tensor", "CPU", 1, True),
            ],
        )

    def test_inplace_goes_through_adinplaceorview(self):
        x, y = torch.randn(3), torch.randn(3)
        records = self.trace(lambda: x.add_(y))
        self.assertEqual(
            [r[1] for r in records[:3]], ["AutogradCPU", "ADInplaceOrView", "CPU"]
        )
        self.assertEqual([r[3] for r in records[:3]], [False, True, True])

    def test_composite_kernel_makes_nested_calls(self):
        x, w = torch.randn(2, 3), torch.randn(4, 3)
        records = self.trace(lambda: torch.nn.functional.linear(x, w))
        self.assertEqual(records[0][0], "aten::linear")
        nested = [r for r in records if r[2] == 1 and not r[3]]
        self.assertIn("aten::matmul", [r[0] for r in nested])

    def test_result_is_returned(self):
        x = torch.ones(3)
        result, records = torch._C._cfg_dispatch_trace(lambda: x * 2)
        self.assertEqual(result, torch.full((3,), 2.0))
        self.assertEqual(records[0][0], "aten::mul.Tensor")

    def test_hook_is_removed(self):
        x = torch.ones(3)
        self.assertEqual(self.trace(lambda: 1 + 1), [])
        with self.assertRaisesRegex(RuntimeError, "must match"):
            self.trace(lambda: x + torch.ones(4))
        self.assertEqual(self.trace(lambda: None), [])

    def test_captures_nest(self):
        x = torch.ones(3)

        def outer():
            inner = self.trace(lambda: x + 1)
            self.assertEqual(len([r for r in inner if r[2] == 0]), 1)
            return x * 2

        records = self.trace(outer)
        self.assertEqual([r[0] for r in records if r[2] == 0], ["aten::mul.Tensor"])


@unittest.skipUnless(HAS_RUNTIME, "cfg_compile runtime not built")
class TestCfgFrontend(TestCase):
    """Bytecode -> SSA graph."""

    hw_classification = HardwareClassification.GENERIC

    def test_decoder_matches_dis(self):
        def loop(x, n):
            total = 0
            for i in range(n):
                if i % 2:
                    continue
                total += x[i]
            return total

        # Enough constants and a long enough body for EXTENDED_ARG prefixes.
        namespace = {}
        body = "".join(
            f"    x = x + {i}.5 if x > {i} else x - {i}.5\n" for i in range(300)
        )
        exec(f"def wide(x):\n{body}    return x\n", namespace)

        for fn in (loop, namespace["wide"]):
            expected = [
                (
                    i.offset,
                    i.opname,
                    i.arg or 0,
                    i.argval if i.opcode in dis.hasjrel else None,
                )
                for i in dis.get_instructions(fn.__code__)
                if i.opname != "EXTENDED_ARG"
            ]
            self.assertEqual(cfg_rs.decode(fn.__code__), expected)
        self.assertTrue(
            any(arg > 255 for _, _, arg, _ in cfg_rs.decode(namespace["wide"].__code__))
        )

    def test_whole_function_is_one_graph(self):
        def f(x, n):
            for _ in range(n):
                if x.sum() > 0:
                    x = x - 1
                else:
                    x = x + 2
            return x

        text = torch.cfg_compile(f, fallback=False).dump()
        self.assertIn("phi", text)
        self.assertIn("br_exhausted", text)
        self.assertIn("binary - ", text)
        self.assertIn("binary + ", text)

    def test_unsupported(self):
        def f(x):
            try:
                return x + 1
            except RuntimeError:
                return x

        with self.assertRaisesRegex(torch.cfg_compile.Unsupported, "try/except"):
            torch.cfg_compile(f, fallback=False)
        self.assertTrue(torch.cfg_compile.is_available())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIs(torch.cfg_compile(f), f)
        self.assertIn(
            "left f uncompiled", str(caught[0].message).replace(f.__qualname__, "f")
        )

    def test_methods_and_modules(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.ones(3))

            def forward(self, x):
                return x * self.w + 1

        m = M()
        x = torch.randn(3)
        for compiled in (torch.cfg_compile(m), torch.cfg_compile(m.forward)):
            for _ in range(3):
                self.assertEqual(compiled(x), m(x))


if __name__ == "__main__":
    run_tests()
