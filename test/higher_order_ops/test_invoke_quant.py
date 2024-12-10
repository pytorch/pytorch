# Owner(s): ["module: higher order operators"]
# flake8: noqa: B950

import contextlib
import dataclasses
import logging

import torch
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
from torch.testing import FileCheck
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class InvokeQuantTest(torch._higher_order_ops.PrimHOPBase):
    def __init__(self):
        super().__init__("invoke_quant_test")

    def __call__(self, subgraph, operands, *, scheme=None, quant_options=None):
        return super().__call__(
            subgraph, operands, scheme=scheme, quant_options=quant_options
        )


invoke_quant_test = InvokeQuantTest()


@dataclasses.dataclass(frozen=True)
class InvokeQuant:
    """
    TODO - fill in next pr
    """

    codegen_low_precision: bool = True

    def __call__(
        self,
        *args,
        **kwargs,
    ):
        if not torch._utils.is_compiling():
            return args[0](*args[1])

        return invoke_quant_test(*args, **kwargs, quant_options=self)  # type: ignore[call-arg]


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
            f.check("repeated_subgraph0").check("repeated_subgraph1")
            for _ in range(2):
                f.check("torch.ops.higher_order.invoke_quant(").check_same("nf4")
            f.run(logs)


class TestInvokeQuantEager(TestInvokeQuant):
    backend = "eager"


class TestInvokeQuantAotEager(TestInvokeQuant):
    backend = "aot_eager"


del TestInvokeQuant

if __name__ == "__main__":
    run_tests()
