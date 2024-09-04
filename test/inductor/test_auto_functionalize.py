# Owner(s): ["module: functionalization"]

import numpy as np

import torch
import torch._dynamo.testing
import torch._inductor.test_case
import torch.onnx.operators
import torch.utils._pytree as pytree
import torch.utils.cpp_extension
from torch import Tensor
from torch.testing._internal.logging_utils import logs_to_string


class AutoFunctionalizeTests(torch._inductor.test_case.TestCase):
    def test_auto_functionalize_can_with_default(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor a, int b, Tensor(d!)? c=None, Tensor? d=None, int e=-1) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            def foo_impl(a, b, c=None, d=None, e=-1):
                a + b
                return

            def f(a, mode):
                return torch.ops.mylib.foo(
                    a,
                    0,
                )

            a = torch.tensor([10, 10, 10], dtype=torch.int64)

            torch.compile(f)(a, 0)

    def test_auto_functionalize_can_with_none_return(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("foo(Tensor x, Tensor(a!) out) -> None")

            def foo_impl(x, out):
                out.copy_(x)

            lib.impl("foo", foo_impl, "CompositeExplicitAutograd")
            x = torch.randn(3)
            out = torch.zeros(3)

            @torch.compile
            def f(x, out):
                torch.ops.mylib.foo(x, out)

            f(x, out)

    def test_auto_functionalize_self_as_mutate_arg(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("foo(Tensor(a!) self) -> None")

            def foo_impl(self: torch.Tensor) -> None:
                self.sin_()

            x = torch.randn(3)
            lib.impl("foo", foo_impl, "CompositeExplicitAutograd")

            @torch.compile(backend="inductor", fullgraph=True)
            def f(x):
                torch.ops.mylib.foo(x)

            f(x)

    def test_auto_functionalize_tensorlist(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor all_gather_output, SymInt[] all_gather_input_split_sizes, int dim, Tensor(a!)[] out) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(all_gather_output, all_gather_input_split_sizes, dim, out):
                for o in out:
                    o.copy_(all_gather_output)

            def f(all_gather_output, all_gather_input_split_sizes, dim, out):
                torch.ops.mylib.foo(
                    all_gather_output, all_gather_input_split_sizes, dim, out
                )

            a = torch.ones(4)
            b = [2, 3]
            c = 0
            d = [torch.empty(4) for _ in range(2)]
            orig_args = (a, b, c, d)

            compiled_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            torch.compile(f, backend="inductor", fullgraph=True)(*compiled_args)

            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            f(*eager_args)
            self.assertEqual(compiled_args, eager_args)

    def test_can_auto_functionalize(self):
        from torch._higher_order_ops.auto_functionalize import can_auto_functionalize

        expected_true = [
            "(Tensor(a!) x) -> ()",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> ()",
            "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> ()",
            "(Tensor(a!) x, Tensor y, Tensor(b!)[] z, SymInt w) -> ()",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> Tensor",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> (Tensor, Tensor)",
        ]
        expected_false = [
            "(Tensor x) -> ()",
            "(Tensor(a) x) -> Tensor(a)",
            "(Tensor(a!) x) -> Tensor(a!)",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> Tensor(a)",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> (Tensor, Tensor(a))",
            "(Tensor(a) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> (Tensor, Tensor(a))",
            "(Tensor(a!) x, Tensor y, Tensor(b!) z, SymInt w, Tensor(c!)? n) -> (Tensor, Tensor[])",
        ]
        for schema in expected_true:
            with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
                torch.library.define("mylib::a", schema, lib=lib)
                self.assertTrue(
                    can_auto_functionalize(torch.ops.mylib.a.default), msg=schema
                )
                self.assertFalse(can_auto_functionalize(torch.ops.mylib.a))

        for schema in expected_false:
            with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
                torch.library.define("mylib::a", schema, lib=lib)
                self.assertFalse(
                    can_auto_functionalize(torch.ops.mylib.a.default), msg=schema
                )
                self.assertFalse(can_auto_functionalize(torch.ops.mylib.a))

    def test_auto_functionalize(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor n) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y, z, w, n):
                x.add_(y[0] + w)
                z.add_(y[1] + n)

            def f(x, y, z, n):
                torch.ops.mylib.foo(x, y, z, 2, n)

            x = torch.randn(3)
            y = (torch.randn(3), torch.randn(3))
            z = torch.randn(3)
            n = torch.randn(3)
            orig_args = (x, y, z, n)

            compiled_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)

            log_stream, ctx = logs_to_string(
                "torch._inductor.compile_fx", "post_grad_graphs"
            )
            with ctx():
                torch.compile(f, backend="inductor", fullgraph=True)(*compiled_args)

            post_grad_graphs = "\n".join(
                log_stream.getvalue().strip().split("\n")[3:]
            ).strip()

            # Check the graph under static shapes
            if torch._dynamo.config.assume_static_by_default:
                self.assertExpectedInline(
                    post_grad_graphs,
                    """\
def forward(self, arg0_1: "f32[3][1]cpu", arg1_1: "f32[3][1]cpu", arg2_1: "f32[3][1]cpu", arg3_1: \
"f32[3][1]cpu", arg4_1: "f32[3][1]cpu"):
        # No stacktrace found for following nodes
        foo_default = torch.ops.mylib.foo.default(arg4_1, [arg2_1, arg3_1], arg1_1, 2, arg0_1);  arg4_1 = arg2_1 = \
arg3_1 = arg1_1 = arg0_1 = foo_default = None
        return ()""",
                )

            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            f(*eager_args)
            self.assertEqual(compiled_args, eager_args)

    def test_auto_functionalize_with_returns(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor n) -> (Tensor, Tensor)",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y, z, w, n):
                x.add_(y[0] + w)
                z.add_(y[1] + n)
                return y[0] + w, y[1] + n

            @torch.library.impl_abstract("mylib::foo", lib=lib)
            def foo_abstract(x, y, z, w, n):
                return y[0] + w, y[1] + n

            def f(x, y, z, n):
                return torch.ops.mylib.foo(x, y, z, 2, n)

            x = torch.randn(3)
            y = (torch.randn(3), torch.randn(3))
            z = torch.randn(3)
            n = torch.randn(3)
            orig_args = (x, y, z, n)

            compiled_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            log_stream, ctx = logs_to_string(
                "torch._inductor.compile_fx", "post_grad_graphs"
            )
            with ctx():
                compiled_out = torch.compile(f, backend="inductor", fullgraph=True)(
                    *compiled_args
                )

            if torch._dynamo.config.assume_static_by_default:
                post_grad_graphs = "\n".join(
                    log_stream.getvalue().strip().split("\n")[3:]
                ).strip()
                self.assertExpectedInline(
                    post_grad_graphs,
                    """\
def forward(self, arg0_1: "f32[3][1]cpu", arg1_1: "f32[3][1]cpu", arg2_1: "f32[3][1]cpu", \
arg3_1: "f32[3][1]cpu", arg4_1: "f32[3][1]cpu"):
        # No stacktrace found for following nodes
        foo_default = torch.ops.mylib.foo.default(arg4_1, [arg2_1, arg3_1], arg1_1, 2, arg0_1);  \
arg4_1 = arg2_1 = arg3_1 = arg1_1 = arg0_1 = None
        getitem_4: "f32[3][1]cpu" = foo_default[0]
        getitem_5: "f32[3][1]cpu" = foo_default[1];  foo_default = None
        return (getitem_4, getitem_5)""",
                )

            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            eager_out = f(*eager_args)
            self.assertEqual(compiled_args, eager_args)
            self.assertEqual(compiled_out, eager_out)

    def test_auto_functionalize_on_view(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x):
                x_np = x.detach().numpy()  # view
                np.sin(x_np, out=x_np)
                return

            x = torch.randn(3)
            expected = x.sin()
            torch.ops.mylib.foo(x)
            assert torch.allclose(x, expected)

            @torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
            def f(x):
                x = x.clone()
                y = x[:]
                torch.ops.mylib.foo(y)
                return x

            y = f(x)
            self.assertEqual(y, x.sin())

    def test_auto_functionalize_optional(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!)? x, Tensor[] y, Tensor(b!)? z, SymInt w, Tensor n) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y, z, w, n):
                if x is not None:
                    x.add_(y[0] + w)
                if z is not None:
                    z.add_(y[1] + n)

            def f(x, y, z, n):
                torch.ops.mylib.foo(x, y, z, 2, n)

            x = None
            y = (torch.randn(3), torch.randn(3))
            z = torch.randn(3)
            n = torch.randn(3)
            orig_args = (x, y, z, n)

            compiled_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            log_stream, ctx = logs_to_string(
                "torch._inductor.compile_fx", "post_grad_graphs"
            )
            with ctx():
                torch.compile(f, backend="inductor", fullgraph=True)(*compiled_args)

            if torch._dynamo.config.assume_static_by_default:
                post_grad_graphs = "\n".join(
                    log_stream.getvalue().strip().split("\n")[3:]
                ).strip()
                self.assertExpectedInline(
                    post_grad_graphs,
                    """\
def forward(self, arg0_1: "f32[3][1]cpu", arg1_1: "f32[3][1]cpu", arg2_1: "f32[3][1]cpu", arg3_1: "f32[3][1]cpu"):
        # No stacktrace found for following nodes
        foo_default = torch.ops.mylib.foo.default(None, [arg2_1, arg3_1], arg1_1, 2, arg0_1);  \
arg2_1 = arg3_1 = arg1_1 = arg0_1 = foo_default = None
        return ()""",
                )

            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            f(*eager_args)
            self.assertEqual(compiled_args, eager_args)

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_unbacked_auto_functionalize_op(self):
        @torch.library.custom_op(
            "mylib::mk_image", mutates_args=("decoder",), device_types=["cpu"]
        )
        def mk_image(decoder: Tensor) -> Tensor:
            return torch.randn(2, 3, 4, 5)

        @torch.library.register_fake("mylib::mk_image")
        def _(decoder: Tensor) -> Tensor:
            image_size = [torch.library.get_ctx().new_dynamic_size() for _ in range(4)]
            return torch.empty(image_size)

        @torch.compile(fullgraph=True)
        def f(x):
            return torch.ops.mylib.mk_image.default(x)

        x = torch.zeros(100, dtype=torch.int64)
        f(x)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
