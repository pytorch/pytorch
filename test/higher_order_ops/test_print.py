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

        # Test kwargs only
        def f_kwargs(x):
            x = x + x
            torch._higher_order_ops.print("moo {x} {y}", x=1, y=2)
            x = x * x
            return x

        # Test positional args only
        def f_args(x):
            x = x + x
            torch._higher_order_ops.print("moo {} {}", 1, 2)
            x = x * x
            return x

        # Test mixed args and kwargs
        def f_mixed(x):
            x = x + x
            torch._higher_order_ops.print("moo {} {y}", 1, y=2)
            x = x * x
            return x

        x = torch.randn(3, 3)

        # Test all three variants produce same output
        for f in [f_kwargs, f_args, f_mixed]:
            with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                f(x)
                printed_output = mock_stdout.getvalue().strip()
            self.assertEqual(printed_output, "moo 1 2")

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
        def f_kwargs(x):
            x = x + x
            torch._higher_order_ops.print("tensor: {t} value: {v}", t=x, v=42)
            return x

        # Test with positional args
        def f_args(x):
            x = x + x
            torch._higher_order_ops.print("tensor: {} value: {}", x, 42)
            return x

        x = torch.tensor([1.0, 2.0, 3.0])
        expected = f"tensor: {x + x} value: 42"

        for f in [f_kwargs, f_args]:
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

        class M_kwargs(torch.nn.Module):
            def forward(self, x):
                torch._higher_order_ops.print("moo {x} {y}", x=1, y=2)
                res = x + x
                torch._higher_order_ops.print("moo {x} {y}", x=1, y=2)
                return (res,)

        class M_args(torch.nn.Module):
            def forward(self, x):
                torch._higher_order_ops.print("values {} {}", 1, 2)
                res = x + x
                torch._higher_order_ops.print("values {} {}", 3, 4)
                return (res,)

        inputs = (torch.randn(3),)

        # With functionalization, it should appear wrapped with with_effects()
        for M in [M_kwargs, M_args]:
            gm, gs = aot_export_module(M(), inputs, trace_joint=False)
            self.assertEqual(len(gs.input_tokens), 1)
            self.assertEqual(len(gs.output_tokens), 1)

        # Check detailed output for kwargs version
        gm, gs = aot_export_module(M_kwargs(), inputs, trace_joint=False)
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

        # Test with kwargs
        def f_kwargs(x):
            x1 = x + x
            torch._higher_order_ops.print("moo {x}", x=x1)
            x2 = x1 * x1
            torch._higher_order_ops.print("moo {x}", x=x2)
            x3 = x2 + x2
            return (x1, x3)

        # Test with positional args
        def f_args(x):
            x1 = x + x
            torch._higher_order_ops.print("value: {}", x1)
            x2 = x1 * x1
            torch._higher_order_ops.print("values: {} {}", x1, x2)
            x3 = x2 + x2
            return (x1, x3)

        x = torch.randn(3, 3)

        # Test kwargs version
        opt_f_kwargs = torch.compile(backend=backend, fullgraph=True)(f_kwargs)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_out = opt_f_kwargs(x)
            printed_output = mock_stdout.getvalue().strip()
            orig_out = f_kwargs(x)

        self.assertEqual(
            printed_output,
            f"moo {x * 2}\nmoo {x * 2 * x * 2}",
        )
        self.assertEqual(orig_out, opt_out)

        # Test args version
        opt_f_args = torch.compile(backend=backend, fullgraph=True)(f_args)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_f_args(x)
            printed_output = mock_stdout.getvalue().strip()

        x1_expected = x * 2
        x2_expected = x1_expected * x1_expected
        expected = f"value: {x1_expected}\nvalues: {x1_expected} {x2_expected}"
        self.assertEqual(printed_output, expected)

        # Test recompilation with different input shape
        x_new = torch.randn(2, 2)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_f_kwargs(x_new)
            printed_output = mock_stdout.getvalue().strip()

        self.assertEqual(
            printed_output,
            f"moo {x_new * 2}\nmoo {x_new * 2 * x_new * 2}",
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
    def test_inductor_python_wrapper_uses_builtin_print(self):
        """Test that the Python wrapper uses builtins.print for both kwargs and positional args.

        This verifies that when compiling with inductor (Python wrapper), the generated
        code uses builtins.print directly rather than calling torch.ops.higher_order.print,
        which is more efficient and avoids unnecessary overhead.
        """
        from torch._inductor.utils import run_and_get_code

        # Test with kwargs
        def f_kwargs(x):
            torch._higher_order_ops.print("value: {val}", val=x)
            res = x + x
            return res

        # Test with positional args
        def f_args(x):
            torch._higher_order_ops.print("values: {} {}", x, 42)
            res = x + x
            return res

        inputs = (torch.randn(2, 3),)

        for f in [f_kwargs, f_args]:
            # Compile and get the generated code
            compiled_f = torch.compile(f, backend="inductor")
            _, codes = run_and_get_code(compiled_f, *inputs)

            # Concatenate all generated code chunks to simplify assertions
            merged_code = "\n".join(codes)

            # Verify that the merged code uses builtins.print
            self.assertIn(
                "builtins.print",
                merged_code,
                "Generated code should use builtins.print for print HOP fallback",
            )
            # And does not call torch.ops.higher_order.print
            self.assertNotIn(
                "torch.ops.higher_order.print",
                merged_code,
                "Generated code should not call torch.ops.higher_order.print directly",
            )

    def test_compile_inductor_cpp_wrapper_print(self):
        """Test print with C++ wrapper enabled for both kwargs and positional args.

        C++ wrapper uses std::cout which writes to file descriptor 1 (stdout).
        We need to redirect the actual file descriptor to capture the output.
        """
        import os
        import tempfile

        from torch._inductor import config

        # Test with kwargs
        def f_kwargs(x):
            torch._higher_order_ops.print("C++ print test: value={x}", x=x)
            res = x + x
            torch._higher_order_ops.print("Result={res}", res=res)
            return res

        # Test with positional args
        def f_args(x):
            torch._higher_order_ops.print("C++ print: {} {}", x, 42)
            res = x + x
            return res

        inputs = (torch.randn(2, 3),)

        # Test kwargs version
        with config.patch({"cpp_wrapper": True}):
            compiled_f = torch.compile(f_kwargs, backend="inductor")

            with tempfile.TemporaryFile(mode="w+") as tmp_stdout:
                original_stdout_fd = os.dup(1)
                try:
                    os.dup2(tmp_stdout.fileno(), 1)
                    res = compiled_f(*inputs)
                    os.fsync(1)
                finally:
                    os.dup2(original_stdout_fd, 1)
                    os.close(original_stdout_fd)

                tmp_stdout.seek(0)
                captured_output = tmp_stdout.read().strip()

            # C++ prints literal format strings with buffer names for tensors
            self.assertEqual(
                captured_output,
                "C++ print test: value=<Tensor:arg1_1>\nResult=<Tensor:buf1>",
            )

            expected = f_kwargs(*inputs)
            self.assertTrue(torch.allclose(res, expected))

        # Test positional args version
        with config.patch({"cpp_wrapper": True}):
            compiled_f = torch.compile(f_args, backend="inductor")

            with tempfile.TemporaryFile(mode="w+") as tmp_stdout:
                original_stdout_fd = os.dup(1)
                try:
                    os.dup2(tmp_stdout.fileno(), 1)
                    res = compiled_f(*inputs)
                    os.fsync(1)
                finally:
                    os.dup2(original_stdout_fd, 1)
                    os.close(original_stdout_fd)

                tmp_stdout.seek(0)
                captured_output = tmp_stdout.read().strip()

            # C++ prints with buffer names for tensors and actual values for scalars
            self.assertEqual(captured_output, "C++ print: <Tensor:arg1_1> 42")

            expected = f_args(*inputs)
            self.assertTrue(torch.allclose(res, expected))


if __name__ == "__main__":
    run_tests()
