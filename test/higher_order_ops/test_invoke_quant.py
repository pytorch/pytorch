# Owner(s): ["module: higher order operators"]
# flake8: noqa: B950

import contextlib
import logging
import unittest

import torch
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
from torch._higher_order_ops import InvokeQuant
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Ignored,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch.testing import FileCheck
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


invoke_quant_tracer = InvokeQuant()


@skipIfTorchDynamo("Not a torch._dynamo test")
class TestInvokeQuant(TestCase):
    backend = ""

    def test_simple(self):
        def gn(x, y):
            return (torch.mul(x, y) + y,)

        def fn(x, y):
            return invoke_quant_tracer(
                gn, (x, y), scheme="nf4", quant_options=invoke_quant_tracer
            )[0]

        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)
        ref = gn(x, y)[0]

        x_clone = x.clone().detach().requires_grad_(False)
        y_clone = y.clone().detach().requires_grad_(False)
        res = torch.compile(fn, backend=self.backend)(x_clone, y_clone)
        self.assertEqual(ref, res)

    def test_construct_inline(self):
        def gn(x, y):
            return (torch.mul(x, y) + y,)

        def fn(x, y):
            return InvokeQuant(codegen_low_precision=False)(gn, (x, y), scheme="nf4")[0]

        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)
        ref = gn(x, y)[0]

        x_clone = x.clone().detach().requires_grad_(False)
        y_clone = y.clone().detach().requires_grad_(False)
        res = torch.compile(fn, backend=self.backend)(x_clone, y_clone)
        self.assertEqual(ref, res)

    def test_inline(self):
        def gn(x, y):
            return (torch.mul(x, y) + y,)

        def fn(x, y):
            return InvokeQuant()(gn, (x, y), scheme="nf4")[0]

        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)
        ref = gn(x, y)[0]

        x_clone = x.clone().detach().requires_grad_(False)
        y_clone = y.clone().detach().requires_grad_(False)
        res = torch.compile(fn, backend=self.backend)(x_clone, y_clone)
        self.assertEqual(ref, res)

    def test_multiple(self):
        torch._logging.set_logs(post_grad_graphs=True)

        def gn(x, y):
            return torch.mul(x, y) + y

        def fn(x, y, z):
            o1 = invoke_quant_tracer(gn, (x, y), scheme="nf4")
            o2 = invoke_quant_tracer(gn, (y, z), scheme="nf4")
            return o1 + o2

        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)
        z = torch.randn(8, requires_grad=False)
        ref = fn(x, y, z)

        log_context = (
            contextlib.nullcontext()
            if self.backend != "inductor"
            else self.assertLogs(logger="torch._inductor", level=logging.DEBUG)
        )

        with log_context as log:
            res = torch.compile(fn, backend=self.backend)(x, y, z)

        self.assertEqual(ref, res)

        if self.backend == "inductor":
            logs = "\n".join(r.getMessage() for r in log.records)
            f = FileCheck()
            f.check("AFTER POST GRAD")
            f.check("subgraph0").check("subgraph1")
            for _ in range(2):
                f.check("torch.ops.higher_order.invoke_quant(").check_same("nf4")
            f.run(logs)


class TestInvokeQuantEager(TestInvokeQuant):
    backend = "eager"


class TestInvokeQuantAotEager(TestInvokeQuant):
    backend = "aot_eager"


class TestInvokeQuantInductor(TestInvokeQuant):
    backend = "inductor"

    def test_pattern_matching(self):
        counter = 0

        test_pass = PatternMatcherPass()

        def my_pass(g):
            return test_pass.apply(g)

        def gn(x, y):
            return torch.mul(x, y) + y

        def fn(x, y, z):
            return invoke_quant_tracer(gn, (x, y), scheme="nf4") @ z

        def fn_no_match(x, y, z):
            return invoke_quant_tracer(gn, (x, y)) @ z

        x = torch.randn(64, 64, requires_grad=False)
        y = torch.randn(64, 64, requires_grad=False)
        z = torch.randn(64, 64, requires_grad=False)

        @register_graph_pattern(
            CallFunction(
                torch.ops.aten.mm,
                CallFunction(
                    torch.ops.higher_order.invoke_quant,
                    Ignored(),
                    Ignored(),
                    Ignored(),
                    scheme="nf4",
                ),
                Arg(),
            ),
            pass_dict=test_pass,
        )
        def quant_matching(match: Match, *args, **kwargs):
            nonlocal counter
            counter += 1

        with unittest.mock.patch(
            "torch._inductor.config.post_grad_custom_pre_pass", my_pass
        ):
            torch.compile(fn)(x, y, z)
            self.assertTrue(counter == 1)

            torch.compile(fn_no_match)(x, y, z)
            self.assertTrue(counter == 1)


del TestInvokeQuant

if __name__ == "__main__":
    run_tests()
