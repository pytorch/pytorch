# Owner(s): ["module: higher order operators"]
import io
from unittest.mock import patch

import torch
from torch._functorch.aot_autograd import aot_export_module
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestHopPrint(TestCase):
    def test_base_print(self):
        def f(x):
            x = x + x
            torch._higher_order_ops.print("moo")
            x = x * x
            torch._higher_order_ops.print("moo")
            return x

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

    def test_print_with_side_effect(self):
        class M(torch.nn.Module):
            def forward(self, x):
                torch._higher_order_ops.print("moo {x} {y}", x=1, y=2)
                res = x + x
                torch._higher_order_ops.print("moo {x} {y}", x=1, y=2)
                return (res,)

        inputs = (torch.randn(3),)

        # With functionalization, it should appear wrapped with with_effects()
        gm, gs = aot_export_module(M(), inputs, trace_joint=False)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    with_effects = torch.ops.higher_order.with_effects(arg0_1, torch.ops.higher_order.print, 'moo {x} {y}', x = 1, y = 2);  \
arg0_1 = None
    getitem = with_effects[0];  with_effects = None
    add = torch.ops.aten.add.Tensor(arg1_1, arg1_1);  arg1_1 = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops.higher_order.print, 'moo {x} {y}', x = 1, y = 2);  \
getitem = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    return (getitem_2, add)""",
        )
        self.assertEqual(len(gs.input_tokens), 1)
        self.assertEqual(len(gs.output_tokens), 1)

    def test_print_with_input_mutations(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                torch._higher_order_ops.print("moo {x} {y}", x=x, y=2)
                res = x + x
                x.add_(res)
                res = x + x
                torch._higher_order_ops.print("moo {x} {y}", x=x, y=res)
                return (res,)

        inputs = (torch.randn(3),)

        # With functionalization, it should appear wrapped with with_effects()
        gm, gs = aot_export_module(M(), inputs, trace_joint=False)
        self.assertEqual(len(gs.input_tokens), 1)
        self.assertEqual(len(gs.output_tokens), 1)
        self.assertEqual(len(gs.user_inputs_to_mutate), 1)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    with_effects = torch.ops.higher_order.with_effects(arg0_1, torch.ops.higher_order.print, 'moo {x} {y}', \
x = arg1_1, y = 2);  arg0_1 = None
    getitem = with_effects[0];  with_effects = None
    add = torch.ops.aten.add.Tensor(arg1_1, arg1_1)
    add_1 = torch.ops.aten.add.Tensor(arg1_1, add);  arg1_1 = add = None
    add_2 = torch.ops.aten.add.Tensor(add_1, add_1)
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops.higher_order.print, 'moo {x} {y}', \
x = add_1, y = add_2);  getitem = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    return (getitem_2, add_1, add_2)""",
        )

    def test_print_gen_schema(self):
        from torch._higher_order_ops.print import print as print_op

        # Test basic schema generation with simple kwargs int
        format_str = "Hello {x} {y}"
        schema = print_op.gen_schema(format_str, x=1, y=2)
        self.assertExpectedInline(
            str(schema),
            """print(str format_str, *, int x, int y) -> ()""",
        )
        # Test schema generation with different types of inputs

        # Tensor input
        tensor = torch.randn(2, 2)
        schema_tensor = print_op.gen_schema("Tensor: {x}", x=tensor)
        self.assertExpectedInline(
            str(schema_tensor),
            """print(str format_str, *, Tensor x) -> ()""",
        )

        # TODO: Add schema support with kwargs with value of list type

        # No kwargs
        schema_no_kwargs = print_op.gen_schema("Simple message")
        self.assertExpectedInline(
            str(schema_no_kwargs),
            """print(str format_str) -> ()""",
        )

    @parametrize("backend", ["eager", "aot_eager"])
    def test_reorder_print_no_graph_break(self, backend):
        def f(x):
            x1 = x + x
            torch._higher_order_ops.print("moo {x}", x=x1)
            x2 = x1 * x1
            torch._higher_order_ops.print("moo {x}", x=x2)
            x3 = x2 + x2
            return (x1, x3)

        # Eager and aot_eager backend for dynamo tracing testing
        x = torch.randn(3, 3)
        opt_f = torch.compile(backend=backend, fullgraph=True)(f)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_out = opt_f(x)
            printed_output = mock_stdout.getvalue().strip()
            orig_out = f(x)

        self.assertEqual(
            printed_output,
            f"moo {x * 2}\nmoo {x * 2 * x * 2}",
        )
        self.assertEqual(orig_out, opt_out)

        x_new = torch.randn(2, 2)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_out = opt_f(x_new)
            printed_output = mock_stdout.getvalue().strip()

        self.assertEqual(
            printed_output,
            f"moo {x_new * 2}\nmoo {x_new * 2 * x_new * 2}",
        )

    @parametrize("backend", ["eager", "aot_eager"])
    def test_constant_mutation(self, backend):
        def f(x):
            alist = [x]
            alist.append(x + 1)
            torch._higher_order_ops.print("moo {x}", x=alist[-1])
            alist[0].sum().item()  # graph break
            res = alist.pop()
            torch._higher_order_ops.print("moo {x}", x=alist[-1])
            res.sum().item()  # graph break
            return res

        inputs = (torch.tensor([1]),)
        opt_f = torch.compile(backend=backend, fullgraph=True)(f)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_out = opt_f(*inputs)
            printed_output = mock_stdout.getvalue().strip()
            orig_out = f(*inputs)

        self.assertEqual(printed_output, "moo tensor([2])\nmoo tensor([1])")
        self.assertEqual(orig_out, opt_out)


if __name__ == "__main__":
    run_tests()
