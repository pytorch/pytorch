# Owner(s): ["module: higher order operators"]
import io
from unittest.mock import patch

import torch
from torch._dynamo.utils import counters
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests, TestCase


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

        fx_f = make_fx(f)(x)
        new_inp = torch.randn(3, 3)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            fx_f(new_inp)
            ori_printed_output = mock_stdout.getvalue().strip()

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            f(new_inp)
            fx_printed_output = mock_stdout.getvalue().strip()

        self.assertEqual(ori_printed_output, fx_printed_output)

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
        gm = make_fx(M(), tracing_mode="symbolic")(*inputs)

        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1):
    print_1 = torch.ops.higher_order.print('moo {x} {y}', x = 1, y = 2);  print_1 = None
    print_2 = torch.ops.higher_order.print('moo {x}', x = arg0_1);  print_2 = None
    add = torch.ops.aten.add.Tensor(arg0_1, arg0_1)
    print_3 = torch.ops.higher_order.print('moo {x} {y}', x = 1, y = 2);  print_3 = None
    sym_size_int = torch.ops.aten.sym_size.int(arg0_1, 0);  arg0_1 = None
    print_4 = torch.ops.higher_order.print('yeehop {x}', x = sym_size_int);  sym_size_int = print_4 = None
    return (add,)""",
        )

        new_inp = torch.randn(4)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            gm(
                new_inp,
            )
            printed_output = mock_stdout.getvalue().strip()

        self.assertEqual(printed_output, f"moo 1 2\nmoo {new_inp}\nmoo 1 2\nyeehop 4")


if __name__ == "__main__":
    run_tests()
