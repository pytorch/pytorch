# Owner(s): ["module: dynamo"]

import unittest

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


@unittest.skipUnless(UNDISTURBED, "the test configuration changes dispatch")
class TestCallSite(TestCase):
    """Call sites built from the capture of one Python-level operation."""

    hw_classification = HardwareClassification.GENERIC

    def site(self, kind, run, *operands, **options):
        result, site, reason = torch._C._cfg_make_site(kind, run, operands, **options)
        self.assertIsNotNone(site, reason)
        return result, site

    def rejects(self, reason, kind, run, *operands, **options):
        site, why = torch._C._cfg_make_site(kind, run, operands, **options)[1:]
        self.assertEqual((site, why), (None, reason))

    def test_binary_operator(self):
        x, y = torch.randn(3), torch.randn(3)
        result, site = self.site("binary", lambda: x + y, x, y)
        self.assertEqual(result, torch.add(x, y))
        self.assertEqual(site.info()["op"], "aten::add.Tensor")
        self.assertEqual(site.info()["args"], ["self=$0", "other=$1", "alpha=1"])

    def test_reflected_operator(self):
        x, y = torch.randn(3), torch.randn(3)
        # What Python falls back to for x - y when x does not know how.
        result, site = self.site("binary", lambda: y.__rsub__(x), x, y)
        self.assertEqual(result, x - y)
        self.assertEqual(site.info()["op"], "aten::rsub.Tensor")
        self.assertEqual(site.info()["args"], ["self=$1", "other=$0", "alpha=1"])

    def test_unary_operator(self):
        x = torch.randn(3)
        info = self.site("unary", lambda: -x, x)[1].info()
        self.assertEqual((info["op"], info["args"]), ("aten::neg", ["self=$0"]))

    def test_rejects_what_is_not_one_dispatcher_call(self):
        x, y, bad = torch.randn(3), torch.randn(3), torch.randn(4)

        def nested():
            return torch._C._cfg_dispatch_trace(lambda: x + y)[0]

        def raises():
            try:
                return x + bad
            except RuntimeError:
                return None

        self.rejects("no dispatcher call", "binary", lambda: None, x, y)
        self.rejects("not a single dispatcher call", "binary", lambda: x + y + y, x, y)
        self.rejects("not a single dispatcher call", "binary", nested, x, y)
        self.rejects("the kernel raised", "binary", raises, x, bad)
        two_results = "operator signature is not supported"
        self.rejects(two_results, "unary", lambda: x.max(0), x)

    def test_rejects_operands_that_do_not_line_up(self):
        x, y = torch.randn(3), torch.randn(3)
        self.rejects("not a binary operator", "binary", lambda: -x, x)
        self.rejects("not a unary operator", "unary", lambda: x + y, x, y)
        self.rejects("unknown operation kind", "ternary", lambda: x + y, x, y)
        other_tensor = "tensor operand is not the kernel argument"
        self.rejects(other_tensor, "unary", lambda: -x, y)
        extra_tensor = "a tensor argument does not come from an operand"
        self.rejects(extra_tensor, "unary", lambda: x + y, x)
        not_a_tensor = "unsupported operand type tuple"
        self.rejects(not_a_tensor, "unary", lambda: torch.stack((x, y)), (x, y))
        self.rejects("the result is not the kernel output", "unary", lambda: [-x], x)

    def test_resolution_chain(self):
        x, y = torch.randn(2, 3), torch.randn(2, 3)
        chain = self.site("binary", lambda: x + y, x, y)[1].info()["chain"]
        self.assertEqual(chain, ["AutogradCPU", "CPU"])
        chain = self.site("binary", lambda: x.__iadd__(y), x, y)[1].info()["chain"]
        self.assertEqual(chain, ["AutogradCPU", "ADInplaceOrView", "CPU"])
        # A kernel that calls other operators ends the chain.
        chain = self.site("unary", lambda: x.T, x)[1].info()["chain"]
        self.assertEqual(chain, ["AutogradCPU"])

    def test_rejects_a_torch_function_mode(self):
        x, y = torch.randn(3), torch.randn(3)
        with torch.overrides.BaseTorchFunctionMode():
            self.rejects(
                "a torch function mode is active", "binary", lambda: x + y, x, y
            )


@unittest.skipUnless(UNDISTURBED, "the test configuration changes dispatch")
class TestDispatchCapture(TestCase):
    """The dispatcher hook that cfg_compile records kernels with."""

    hw_classification = HardwareClassification.GENERIC

    def trace(self, fn):
        return torch._C._cfg_dispatch_trace(fn)[1]

    def detail(self, fn):
        return torch._C._cfg_dispatch_trace(fn, detail=True)[1]

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

    def test_detail_links_kernels_to_their_callers(self):
        x, y = torch.randn(3), torch.randn(3)
        records = self.detail(lambda: x.add_(y))
        self.assertEqual(
            [(r["key"], r["parent"]) for r in records],
            [("AutogradCPU", -1), ("ADInplaceOrView", 0), ("CPU", 1)],
        )
        x, w = torch.randn(2, 3), torch.randn(4, 3)
        records = self.detail(lambda: torch.nn.functional.linear(x, w))
        self.assertEqual(records[0]["parent"], -1)
        for r in records[1:]:
            self.assertEqual(records[r["parent"]]["depth"], r["depth"] - 1)

    def test_detail_marks_kernels_that_raised(self):
        x, bad = torch.ones(3), torch.ones(4)

        def fn():
            with self.assertRaisesRegex(RuntimeError, "must match"):
                torch.add(x, bad)
            return torch.add(x, x)

        records = self.detail(fn)
        self.assertEqual([r["threw"] for r in records], [True, True, False, False])
        self.assertEqual(records[0]["outputs"], ())

    def test_detail_has_the_thread_local_key_sets(self):
        x, y = torch.randn(3), torch.randn(3)
        top, backend = self.detail(lambda: torch.add(x, y))
        excluded = str(torch._C._dispatch_tls_local_exclude_set())
        self.assertNotIn("AutogradOther", excluded)
        self.assertEqual(top["tls_excluded"], excluded)
        # The Autograd kernel excludes the autograd keys before it redispatches.
        self.assertIn("AutogradOther", backend["tls_excluded"])
        self.assertEqual(top["tls_included"], backend["tls_included"])

    def test_detail_keeps_inputs_and_outputs_of_top_level_calls(self):
        x, y = torch.randn(3), torch.randn(3)
        result, (top, backend) = torch._C._cfg_dispatch_trace(
            lambda: torch.add(x, y), detail=True
        )
        self.assertIs(top["inputs"][0], x)
        self.assertIs(top["inputs"][1], y)
        # alpha, a default the argument parser filled in
        self.assertEqual(top["inputs"][2:], (1,))
        self.assertEqual(len(top["outputs"]), 1)
        self.assertIs(top["outputs"][0], result)
        self.assertEqual((backend["inputs"], backend["outputs"]), (None, None))

    def test_exception_propagates(self):
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


if __name__ == "__main__":
    run_tests()
