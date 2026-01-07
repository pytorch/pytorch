# Owner(s): ["module: higher order operators"]
import io
from unittest.mock import patch

import torch
from torch._dynamo.testing import (
    AotEagerAndRecordGraphs,
    EagerAndRecordGraphs,
    InductorAndRecordGraphs,
)
from torch._functorch.aot_autograd import aot_export_module
from torch._inductor.utils import run_and_get_code
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
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

    def test_args_kwargs_print(self):
        """Test print with kwargs, positional args, and mixed args/kwargs."""

        # Test positional args, kwargs, and mixed only
        def f(x):
            x = x + x
            torch._higher_order_ops.print("moo kwargs {x} {y}", x=1, y=2)
            torch._higher_order_ops.print("moo args {} {}", 1, 2)
            torch._higher_order_ops.print("moo mixed {} {y}", 1, y=2)
            x = x * x
            return x

        x = torch.randn(3, 3)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            f(x)
            printed_output = mock_stdout.getvalue().strip()
        self.assertEqual(printed_output, "moo kwargs 1 2\nmoo args 1 2\nmoo mixed 1 2")

        # Test with make_fx
        fx_f = make_fx(f)(x)
        new_inp = torch.randn(3, 3)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            fx_f(new_inp)
            fx_printed_output = mock_stdout.getvalue().strip()

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            f(new_inp)
            ori_printed_output = mock_stdout.getvalue().strip()

        self.assertEqual(ori_printed_output, fx_printed_output)

    def test_args_kwargs_with_tensor(self):
        """Test print with args/kwargs including tensors."""

        # Test with kwargs
        def f(x):
            x = x + x
            torch._higher_order_ops.print("tensor: {t} value: {v}", t=x, v=42)
            torch._higher_order_ops.print("tensor: {} value: {}", x, 42)
            return x

        x = torch.tensor([1.0, 2.0, 3.0])
        expected = f"tensor: {x + x} value: 42\ntensor: {x + x} value: 42"

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            f(x)
            printed_output = mock_stdout.getvalue().strip()
        self.assertEqual(printed_output, expected)

    def test_print_with_proxy_graph(self):
        """Test print with both kwargs and positional args in proxy graph."""

        class M(torch.nn.Module):
            def forward(self, x):
                # kwargs style
                torch._higher_order_ops.print("moo {x} {y}", x=1, y=2)
                torch._higher_order_ops.print("moo {x}", x=x)
                res = x + x
                # positional args style
                torch._higher_order_ops.print("values: {} {}", 3, 4)
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
    print_3 = torch.ops.higher_order.print('values: {} {}', 3, 4);  print_3 = None
    sym_size_int = torch.ops.aten.sym_size.int(arg0_1, 0);  arg0_1 = None
    print_4 = torch.ops.higher_order.print('yeehop {x}', x = sym_size_int);  sym_size_int = print_4 = None
    return (add,)""",
        )

        new_inp = torch.randn(4)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            gm(new_inp)
            printed_output = mock_stdout.getvalue().strip()

        self.assertEqual(
            printed_output, f"moo 1 2\nmoo {new_inp}\nvalues: 3 4\nyeehop 4"
        )

    def test_print_with_side_effect(self):
        """Test print with kwargs and positional args with side effects."""

        class M(torch.nn.Module):
            def forward(self, x):
                torch._higher_order_ops.print("moo {x} {y}", x=1, y=2)
                res = x + x
                torch._higher_order_ops.print("values {} {}", 3, res)
                return (res,)

        inputs = (torch.randn(3),)

        # With functionalization, it should appear wrapped with with_effects()
        gm, gs = aot_export_module(M(), inputs, trace_joint=False)
        self.assertEqual(len(gs.input_tokens), 1)
        self.assertEqual(len(gs.output_tokens), 1)

        # Check detailed output for kwargs version
        gm, gs = aot_export_module(M(), inputs, trace_joint=False)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    with_effects = torch.ops.higher_order.with_effects(arg0_1, torch.ops.higher_order.print, 'moo {x} {y}', x = 1, y = 2);  \
arg0_1 = None
    getitem = with_effects[0];  with_effects = None
    add = torch.ops.aten.add.Tensor(arg1_1, arg1_1);  arg1_1 = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops.higher_order.print, 'values {} {}', 3, add);\
  getitem = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    return (getitem_2, add)""",
        )

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
        """Test schema generation with both kwargs and positional args."""
        from torch._higher_order_ops.print import print as print_op

        # Test basic schema generation with kwargs
        format_str = "Hello {x} {y}"
        schema = print_op.gen_schema(format_str, x=1, y=2)
        self.assertExpectedInline(
            str(schema),
            """print(str format_str, *, int x, int y) -> ()""",
        )

        # Test with positional args only
        schema_args = print_op.gen_schema("Hello {} {}", 1, 2)
        self.assertExpectedInline(
            str(schema_args),
            """print(str format_str, int arg0, int arg1) -> ()""",
        )

        # Test with mixed args and kwargs
        schema_mixed = print_op.gen_schema("Hello {} {y}", 1, y=2)
        self.assertExpectedInline(
            str(schema_mixed),
            """print(str format_str, int arg0, *, int y) -> ()""",
        )

        # Test with tensor input (kwargs)
        tensor = torch.randn(2, 2)
        schema_tensor = print_op.gen_schema("Tensor: {x}", x=tensor)
        self.assertExpectedInline(
            str(schema_tensor),
            """print(str format_str, *, Tensor x) -> ()""",
        )

        # Test with tensor positional arg
        schema_tensor_arg = print_op.gen_schema("Tensor: {}", tensor)
        self.assertExpectedInline(
            str(schema_tensor_arg),
            """print(str format_str, Tensor arg0) -> ()""",
        )

        # No args or kwargs
        schema_no_args = print_op.gen_schema("Simple message")
        self.assertExpectedInline(
            str(schema_no_args),
            """print(str format_str) -> ()""",
        )

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_reorder_print_no_graph_break(self, backend):
        """Test print with kwargs and positional args across different backends."""

        # Test with kwargs, args, and mixed
        def f(x):
            x1 = x + x
            torch._higher_order_ops.print("moo kwargs {x}", x=x1)
            x2 = x1 * x1
            torch._higher_order_ops.print("moo args {}", x2)
            x3 = x2 + x2
            return (x1, x3)

        x = torch.randn(3, 3)

        # Test kwargs version
        opt_f = torch.compile(backend=backend, fullgraph=True)(f)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_out = opt_f(x)
            printed_output = mock_stdout.getvalue().strip()
            orig_out = f(x)

        self.assertEqual(
            printed_output,
            f"moo kwargs {x * 2}\nmoo args {x * 2 * x * 2}",
        )
        self.assertEqual(orig_out, opt_out)

        # Test recompilation with different input shape
        x_new = torch.randn(2, 2)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_f(x_new)
            printed_output = mock_stdout.getvalue().strip()

        self.assertEqual(
            printed_output,
            f"moo kwargs {x_new * 2}\nmoo args {x_new * 2 * x_new * 2}",
        )

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
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

    @skipIfTorchDynamo("Skipped under Dynamo")
    def test_inductor_python_wrapper_uses_python_print(self):
        """Test that the Python wrapper uses python print instead of HOP for print fallback.

        This verifies that when compiling with inductor (Python wrapper), the generated
        code uses builtins.print directly rather than calling torch.ops.higher_order.print,
        which is more efficient and avoids unnecessary overhead.
        """

        # Test with kwargs, args
        def f(x):
            torch._higher_order_ops.print("value: {val}", val=x)
            res = x + x
            torch._higher_order_ops.print("values: {} {}", x, 42)
            return res

        inputs = (torch.randn(2, 3),)

        # Compile and get the generated code
        compiled_f = torch.compile(f, backend="inductor")
        _, codes = run_and_get_code(compiled_f, *inputs)

        # Concatenate all generated code chunks to simplify assertions
        merged_code = "\n".join(codes)

        # Verify that the merged code uses python print
        self.assertIn(
            "print",
            merged_code,
            "Generated code should use python print for print HOP fallback",
        )
        # And does not call torch.ops.higher_order.print
        self.assertNotIn(
            "torch.ops.higher_order.print",
            merged_code,
            "Generated code should not call torch.ops.higher_order.print directly",
        )

    def test_print_dynamo_graph(self):
        """Test capturing the actual Dynamo graph for print HOP.

        This test captures the Dynamo graph using EagerAndRecordGraphs backend,
        which shows how Dynamo traces the print HOP through bytecode analysis.
        """

        class M(torch.nn.Module):
            def forward(self, x):
                torch._higher_order_ops.print("moo {x} {y}", x=1, y=2)
                res = x + x
                torch._higher_order_ops.print("values {} {}", 3, res)
                return (res,)

        inputs = (torch.randn(3),)

        # Capture actual Dynamo graph using EagerAndRecordGraphs
        backend = EagerAndRecordGraphs()
        compiled_m = torch.compile(M(), backend=backend, fullgraph=True)

        with patch("sys.stdout", new_callable=io.StringIO):
            compiled_m(*inputs)

        # Verify we captured a graph
        self.assertEqual(len(backend.graphs), 1)

        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    print_1 = torch.ops.higher_order.print('moo {x} {y}', x = 1, y = 2);  print_1 = None
    res = l_x_ + l_x_;  l_x_ = None
    print_2 = torch.ops.higher_order.print('values {} {}', 3, res);  print_2 = None
    return (res,)""",
        )

    def test_print_aot_autograd_graph(self):
        """Test capturing the AOT Autograd graph for print HOP.

        This test captures the AOT Autograd forward graph using AotEagerAndRecordGraphs,
        which shows how AOT Autograd functionalizes the print HOP with with_effects.
        """

        class M(torch.nn.Module):
            def forward(self, x):
                torch._higher_order_ops.print("moo {x} {y}", x=1, y=2)
                res = x + x
                torch._higher_order_ops.print("values {} {}", 3, res)
                return (res,)

        inputs = (torch.randn(3, requires_grad=True),)

        # Capture AOT Autograd graphs using AotEagerAndRecordGraphs
        backend = AotEagerAndRecordGraphs()
        compiled_m = torch.compile(M(), backend=backend, fullgraph=True)

        with patch("sys.stdout", new_callable=io.StringIO):
            res = compiled_m(*inputs)
            # Run backward to capture backward graph
            res[0].sum().backward()

        # Check forward graph - should have with_effects wrapping print
        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, primals_1, primals_2):
    with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.higher_order.print, \
'moo {x} {y}', x = 1, y = 2);  primals_1 = None
    getitem = with_effects[0];  with_effects = None
    add = torch.ops.aten.add.Tensor(primals_2, primals_2);  primals_2 = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops.higher_order.print, \
'values {} {}', 3, add);  getitem = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    return (getitem_2, add)""",  # noqa: B950
        )

        # Check backward graph - print HOP doesn't contribute to gradients
        self.assertExpectedInline(
            backend.bw_graphs[0].code.strip(),
            """\
def forward(self, tangents_1):
    add_1 = torch.ops.aten.add.Tensor(tangents_1, tangents_1);  tangents_1 = None
    return (add_1,)""",
        )

    @skipIfTorchDynamo("Skipped under Dynamo")
    def test_print_inductor_graph(self):
        """Test capturing the Inductor graph and generated code for print HOP.

        This test captures:
        1. The Inductor input FX graph using InductorAndRecordGraphs
        2. The Inductor output generated code using run_and_get_code

        This shows the full Inductor pipeline for print HOP.
        """

        class M(torch.nn.Module):
            def forward(self, x):
                torch._higher_order_ops.print("moo {x} {y}", x=1, y=2)
                res = x + x
                torch._higher_order_ops.print("values {} {}", 3, res)
                return (res,)

        inputs = (torch.randn(3, requires_grad=False),)

        # 1. Capture Inductor INPUT graph using InductorAndRecordGraphs
        backend = InductorAndRecordGraphs()
        compiled_m = torch.compile(M(), backend=backend, fullgraph=True)

        with patch("sys.stdout", new_callable=io.StringIO):
            compiled_m(*inputs)

        # Verify we captured a graph
        self.assertEqual(len(backend.graphs), 1)
        self.assertGreaterEqual(len(backend.inductor_graphs), 1)

        # Check inductor INPUT graph - print wrapped with with_effects
        # Inductor creates/sinks tokens internally rather than passing as args
        self.assertExpectedInline(
            backend.inductor_graphs[0].code.strip(),
            """\
def forward(self, arg1_1):
    _make_token_default = torch.ops.prims._make_token.default()
    with_effects = torch.ops.higher_order.with_effects(_make_token_default, torch.ops.higher_order.print, 'moo {x} {y}', x = 1, y = 2);  _make_token_default = None
    getitem = with_effects[0];  with_effects = None
    add = torch.ops.aten.add.Tensor(arg1_1, arg1_1);  arg1_1 = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops.higher_order.print, 'values {} {}', 3, add);  getitem = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    _sink_tokens_default = torch.ops.prims._sink_tokens.default([getitem_2]);  getitem_2 = _sink_tokens_default = None
    return (add,)""",  # noqa: B950
        )


if __name__ == "__main__":
    run_tests()
