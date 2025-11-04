# Owner(s): ["module: higher order operators"]
from torch._dynamo.decorators import graph_break
import io
from unittest.mock import patch

import torch
from torch._dynamo.utils import counters
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck

from torch.testing._internal.torchbind_impls import init_torchbind_implementations


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

    def setUp(self):
        init_torchbind_implementations()

    def test_print_with_proxy_graph(self):
        class M(torch.nn.Module):
            def forward(self, x):
                torch._higher_order_ops.print("moo {x} {y}", x=1, y=2)
                res = x + x
                torch._higher_order_ops.print("moo {x} {y}", x=1, y=2)
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
    add = torch.ops.aten.add.Tensor(arg0_1, arg0_1);  arg0_1 = None
    print_2 = torch.ops.higher_order.print('moo {x} {y}', {'x': 1, 'y': 2});  print_2 = None
    return (add,)""",
        )


if __name__ == "__main__":
    run_tests()
