# Owner(s): ["module: higher order operators"]
import io
from unittest.mock import patch

import torch
from torch._dynamo.decorators import graph_break
from torch._dynamo.utils import counters
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing import FileCheck

from torch._functorch.aot_autograd import aot_export_module


class TestHopPrint(TestCase):
    def test_base_print(self):
        def f(x):
            x = x + x
            torch._higher_order_ops.print("moo")
            x = x * x
            torch._higher_order_ops.print("moo")
            return x

        counters.clear()
        x = torch.randn(3, 3)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            f(x)
            printed_output = mock_stdout.getvalue().strip()

        self.assertEqual(printed_output, "moo\nmoo")

    def test_para_print(self):
        def f(x):
            x = x + x
            torch._higher_order_ops.print("moo {x} {y}", x=1, y=2)
            x = x * x
            return x

        counters.clear()
        x = torch.randn(3, 3)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            f(x)
            printed_output = mock_stdout.getvalue().strip()

        self.assertEqual(printed_output, "moo 1 2")

    def test_tensor_print(self):
        def f(x):
            x = x + x
            torch._higher_order_ops.print("moo {x}", x=x)
            x = x * x
            torch._higher_order_ops.print("yeehop {x}", x=x.shape[0])
            return x

        counters.clear()
        x = torch.randn(3, 3)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            f(x)
            printed_output = mock_stdout.getvalue().strip()

        self.assertEqual(printed_output, f"moo {x * 2}\nyeehop 3")

    def test_print_with_proxy_graph(self):
        class M(torch.nn.Module):
            def forward(self, x):
                torch._higher_order_ops.print("moo {x} {y}", x=1, y=2)
                torch._higher_order_ops.print("moo {x}", x=x)
                res = x + x
                torch._higher_order_ops.print("moo {x} {y}", x=1, y=2)
                torch._higher_order_ops.print("yeehop {x}", x=x.shape[0])
                return (res,)

        inputs = (torch.randn(3),)

        # Without functionalization, print should just appear in the graph directly
        gm = make_fx(M())(*inputs)
        graph_break()
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1):
    print_1 = torch.ops.higher_order.print('moo {x} {y}', {'x': 1, 'y': 2});  print_1 = None
    print_2 = torch.ops.higher_order.print('moo {x}', {'x': arg0_1});  print_2 = None
    add = torch.ops.aten.add.Tensor(arg0_1, arg0_1);  arg0_1 = None
    print_3 = torch.ops.higher_order.print('moo {x} {y}', {'x': 1, 'y': 2});  print_3 = None
    print_4 = torch.ops.higher_order.print('yeehop {x}', {'x': 3});  print_4 = None
    return (add,)""",
        )

    def test_print_with_functionalization(self):
        class M(torch.nn.Module):
            def forward(self, x):
                torch._higher_order_ops.print("moo {x} {y}", x=1, y=2)
                res = x + x
                torch._higher_order_ops.print("moo {x} {y}", x=1, y=2)
                return (res,)

        inputs = (torch.randn(3),)

        # Without functionalization, print should just appear in the graph directly
        gm = make_fx(M())(*inputs)
        # FileCheck().check_count("torch.ops.aten._print.default", 2, exactly=True).run(
        #     gm.code
        # )

        # With functionalization, it should appear wrapped with with_effects()
        gm, gs = aot_export_module(M(), inputs, trace_joint=False)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1):
    print_1 = torch.ops.higher_order.print('moo {x} {y}', {'x': 1, 'y': 2});  print_1 = None
    add = torch.ops.aten.add.Tensor(arg0_1, arg0_1);  arg0_1 = None
    print_2 = torch.ops.higher_order.print('moo {x} {y}', {'x': 1, 'y': 2});  print_2 = None
    return (add,)""",
        )
        self.assertEqual(len(gs.input_tokens), 1)
        self.assertEqual(len(gs.output_tokens), 1)


if __name__ == "__main__":
    run_tests()
