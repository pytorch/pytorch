# Owner(s): ["module: dynamo"]

import numpy as np

import torch
import torch._dynamo.testing
import torch._inductor.test_case
import torch.onnx.operators
import torch.utils._pytree as pytree
import torch.utils.cpp_extension
from torch.testing._internal.logging_utils import logs_to_string


class MiscTests(torch._inductor.test_case.TestCase):
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

    def test_inference_mode1(self):
        with torch.inference_mode():
            self.test_auto_functionalize_extra1()

    # This fail
    # def test_inference_mode2(self):
    #     with torch.inference_mode():
    #         self.test_auto_functionalize_extra2()

    def test_dynamic(self):
        self.test_auto_functionalize(_dynamic=True)

    def test_dynamic2(self):
        self.test_auto_functionalize_extra1(_dynamic=True)

    def test_dynamic3(self):
        self.test_auto_functionalize_extra2(_dynamic=True)

    def test_auto_functionalize(self, _dynamic=False):
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
                torch.compile(f, backend="inductor", dynamic=_dynamic, fullgraph=True)(
                    *compiled_args
                )

            post_grad_graphs = "\n".join(
                log_stream.getvalue().strip().split("\n")[3:]
            ).strip()

            if torch._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        post_grad_graphs,
                        """\
def forward(self, arg0_1: "Sym(s0)", arg1_1: "f32[s0][1]cpu", arg2_1: "f32[s0][1]cpu", arg3_1: "f32[s0][1]cpu", \
arg4_1: "f32[s0][1]cpu", arg5_1: "f32[s0][1]cpu"):
        foo_default = torch.ops.mylib.foo.default(arg5_1, [arg3_1, arg4_1], arg2_1, 2, arg1_1);  \
arg3_1 = arg4_1 = arg1_1 = foo_default = None

        copy_: "f32[s0][1]cpu" = torch.ops.aten.copy_.default(arg2_1, arg2_1);  arg2_1 = copy_ = None
        copy__1: "f32[s0][1]cpu" = torch.ops.aten.copy_.default(arg5_1, arg5_1);  arg5_1 = copy__1 = None
        return ()""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        post_grad_graphs,
                        """\
def forward(self, arg0_1: "f32[3][1]cpu", arg1_1: "f32[3][1]cpu", arg2_1: "f32[3][1]cpu", arg3_1: "f32[3][1]cpu", \
arg4_1: "f32[3][1]cpu"):
        foo_default = torch.ops.mylib.foo.default(arg4_1, [arg2_1, arg3_1], arg1_1, 2, arg0_1);  arg2_1 = \
arg3_1 = arg0_1 = foo_default = None

        copy_: "f32[3][1]cpu" = torch.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy_ = None
        copy__1: "f32[3][1]cpu" = torch.ops.aten.copy_.default(arg4_1, arg4_1);  arg4_1 = copy__1 = None
        return ()""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
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
def forward(self, arg0_1: "f32[3][1]cpu", arg1_1: "f32[3][1]cpu", arg2_1: "f32[3][1]cpu", arg3_1: "f32[3][1]cpu", arg4_1: "f32[3][1]cpu"):
        foo_default = torch.ops.mylib.foo.default(arg4_1, [arg2_1, arg3_1], arg1_1, 2, arg0_1);  arg2_1 = arg3_1 = arg0_1 = None
        getitem_4: "f32[3][1]cpu" = foo_default[0]
        getitem_5: "f32[3][1]cpu" = foo_default[1];  foo_default = None
        
        copy_: "f32[3][1]cpu" = torch.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy_ = None
        copy__1: "f32[3][1]cpu" = torch.ops.aten.copy_.default(arg4_1, arg4_1);  arg4_1 = copy__1 = None
        return (getitem_4, getitem_5)""",
                    ignore_comments=True,
                )

            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            eager_out = f(*eager_args)
            self.assertEqual(compiled_args, eager_args)
            self.assertEqual(compiled_out, eager_out)

    def run_aot_eager(self, f, orig_args, _dynamic=False):
        aot_eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)

        log_stream, ctx = logs_to_string(
            "torch._functorch._aot_autograd.dispatch_and_compile_graph", "aot_graphs"
        )

        result = None
        with ctx():
            result = torch.compile(
                f, backend="aot_eager", fullgraph=True, dynamic=_dynamic
            )(*aot_eager_args)

            graph = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        return [aot_eager_args, result, graph]

    def run_inductor(self, f, orig_args, _dynamic=False):
        compiled_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)

        log_stream, ctx = logs_to_string(
            "torch._inductor.compile_fx", "post_grad_graphs"
        )
        result = None
        with ctx():
            result = torch.compile(
                f, backend="inductor", fullgraph=True, dynamic=_dynamic
            )(*compiled_args)

            graph = "\n".join(log_stream.getvalue().strip().split("\n")[3:]).strip()

        return [compiled_args, result, graph]

    # foo takes two inputs that are not views.
    def test_auto_functionalize_extra1(self, _dynamic=False):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y):
                x.sin_()
                y.sin_()

            def f(x, y):
                torch.ops.mylib.foo(x, y)
                return x + y

            orig_args = (torch.randn(2), torch.randn(2))

            [aot_eager_args, result1, graph_aot] = self.run_aot_eager(
                f, orig_args, _dynamic
            )
            [inductor_args, result2, graph_inductor] = self.run_inductor(
                f, orig_args, _dynamic
            )
            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            result3 = f(*eager_args)

            self.assertEqual(inductor_args, eager_args)
            self.assertEqual(inductor_args, aot_eager_args)

            self.assertEqual(result3, result1)
            self.assertEqual(result3, result2)

            if torch._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "Sym(s0)", arg1_1: "f32[s0][1]cpu", arg2_1: "f32[s0][1]cpu"):
        auto_functionalized = torch.ops.higher_order.auto_functionalized(torch.ops.mylib.foo.default, \
_x_base_index = 0, _y_base_index = 1, _all_bases = [arg2_1, arg1_1])
        getitem_1: "f32[s0][1]cpu" = auto_functionalized[1]
        getitem_2: "f32[s0][1]cpu" = auto_functionalized[2];  auto_functionalized = None
        add: "f32[s0][1]cpu" = torch.ops.aten.add.Tensor(getitem_1, getitem_2)
        copy_: "f32[s0][1]cpu" = torch.ops.aten.copy_.default(arg1_1, getitem_2);  arg1_1 = getitem_2 = copy_ = None
        copy__1: "f32[s0][1]cpu" = torch.ops.aten.copy_.default(arg2_1, getitem_1);  arg2_1 = getitem_1 = copy__1 = None
        return (add,)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu", arg1_1: "f32[2][1]cpu"):
        auto_functionalized = torch.ops.higher_order.auto_functionalized(torch.ops.mylib.foo.default, \
_x_base_index = 0, _y_base_index = 1, _all_bases = [arg1_1, arg0_1])
        getitem_1: "f32[2][1]cpu" = auto_functionalized[1]
        getitem_2: "f32[2][1]cpu" = auto_functionalized[2];  auto_functionalized = None
        add: "f32[2][1]cpu" = torch.ops.aten.add.Tensor(getitem_1, getitem_2)
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_2);  arg0_1 = getitem_2 = copy_ = None
        copy__1: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg1_1, getitem_1);  arg1_1 = getitem_1 = copy__1 = None
        return (add,)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

            if torch._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "Sym(s0)", arg1_1: "f32[s0][1]cpu", arg2_1: "f32[s0][1]cpu"):
        foo_default = torch.ops.mylib.foo.default(arg2_1, arg1_1);  foo_default = None
        add: "f32[s0][1]cpu" = torch.ops.aten.add.Tensor(arg2_1, arg1_1)
        copy_: "f32[s0][1]cpu" = torch.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy_ = None
        copy__1: "f32[s0][1]cpu" = torch.ops.aten.copy_.default(arg2_1, arg2_1);  arg2_1 = copy__1 = None
        return (add,)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu", arg1_1: "f32[2][1]cpu"):
        foo_default = torch.ops.mylib.foo.default(arg1_1, arg0_1);  foo_default = None
        add: "f32[2][1]cpu" = torch.ops.aten.add.Tensor(arg1_1, arg0_1)
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  arg0_1 = copy_ = None
        copy__1: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy__1 = None
        return (add,)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

    # foo takes two views on the same input, function does not have return.
    def test_auto_functionalize_extra2(self, _dynamic=False):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y):
                x.sin_()
                y.sin_()

            def f(x):
                a = x[0]
                b = x[1]
                torch.ops.mylib.foo(a, b)
                return

            orig_args = [torch.randn(2)]

            [aot_eager_args, result1, graph_aot] = self.run_aot_eager(
                f, orig_args, _dynamic
            )
            [inductor_args, result2, graph_inductor] = self.run_inductor(
                f, orig_args, _dynamic
            )
            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            result3 = f(*eager_args)

            self.assertEqual(inductor_args, eager_args)
            self.assertEqual(inductor_args, aot_eager_args)

            self.assertEqual(result3, result1)
            self.assertEqual(result3, result2)

            if torch._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "Sym(s0)", arg1_1: "f32[s0][1]cpu"):
        auto_functionalized = torch.ops.higher_order.auto_functionalized(torch.ops.mylib.foo.default, \
_x_base_index = 0, _x_size = (), _x_stride = (), _x_storage_offset = 0, _y_base_index = 0, _y_size = (), \
_y_stride = (), _y_storage_offset = 1, _all_bases = [arg1_1])
        getitem_1: "f32[s0][1]cpu" = auto_functionalized[1];  auto_functionalized = None
        copy_: "f32[s0][1]cpu" = torch.ops.aten.copy_.default(arg1_1, getitem_1);  arg1_1 = getitem_1 = copy_ = None
        return ()""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        auto_functionalized = torch.ops.higher_order.auto_functionalized(torch.ops.mylib.foo.default, \
_x_base_index = 0, _x_size = (), _x_stride = (), _x_storage_offset = 0, _y_base_index = 0, _y_size = (), \
_y_stride = (), _y_storage_offset = 1, _all_bases = [arg0_1])
        getitem_1: "f32[2][1]cpu" = auto_functionalized[1];  auto_functionalized = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = getitem_1 = copy_ = None
        return ()""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

            # 2. Run with inductor backend

            if torch._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "Sym(s0)", arg1_1: "f32[s0][1]cpu"):
        as_strided_default: "f32[][]cpu" = torch.ops.aten.as_strided.default(arg1_1, [], [], 0)
        as_strided_default_1: "f32[][]cpu" = torch.ops.aten.as_strided.default(arg1_1, [], [], 1)
        foo_default = torch.ops.mylib.foo.default(as_strided_default, as_strided_default_1);  \
as_strided_default = as_strided_default_1 = foo_default = None
        copy_: "f32[s0][1]cpu" = torch.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy_ = None
        return ()""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        as_strided_default: "f32[][]cpu" = torch.ops.aten.as_strided.default(arg0_1, [], [], 0)
        as_strided_default_1: "f32[][]cpu" = torch.ops.aten.as_strided.default(arg0_1, [], [], 1)
        foo_default = torch.ops.mylib.foo.default(as_strided_default, as_strided_default_1);  \
as_strided_default = as_strided_default_1 = foo_default = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  arg0_1 = copy_ = None
        return ()""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

    # foo takes two views on the same input, function returns both views and the input
    def test_auto_functionalize_extra3(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y):
                x.sin_()
                y.sin_()

            def f(x):
                a = x[0]
                b = x[1]
                torch.ops.mylib.foo(a, b)
                return (a, b, x)

            orig_args = [torch.randn(2)]

            [aot_eager_args, result1, graph_aot] = self.run_aot_eager(f, orig_args)
            [inductor_args, result2, graph_inductor] = self.run_inductor(f, orig_args)
            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            result3 = f(*eager_args)

            self.assertEqual(inductor_args, eager_args)
            self.assertEqual(inductor_args, aot_eager_args)

            self.assertEqual(result3, result1)
            self.assertEqual(result3, result2)

            if torch._dynamo.config.assume_static_by_default:
                self.assertExpectedInline(
                    graph_aot,
                    """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        auto_functionalized = torch.ops.higher_order.auto_functionalized(torch.ops.mylib.foo.default, \
_x_base_index = 0, _x_size = (), _x_stride = (), _x_storage_offset = 0, _y_base_index = 0, _y_size = (), \
_y_stride = (), _y_storage_offset = 1, _all_bases = [arg0_1])
        getitem_1: "f32[2][1]cpu" = auto_functionalized[1];  auto_functionalized = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = copy_ = None
        select_2: "f32[][]cpu" = torch.ops.aten.select.int(getitem_1, 0, 0)
        select_3: "f32[][]cpu" = torch.ops.aten.select.int(getitem_1, 0, 1);  getitem_1 = None
        return (select_2, select_3)""",
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

            # 2. Run with inductor backend

            if torch._dynamo.config.assume_static_by_default:
                self.assertExpectedInline(
                    graph_inductor,
                    """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        as_strided_default: "f32[][]cpu" = torch.ops.aten.as_strided.default(arg0_1, [], [], 0)
        as_strided_default_1: "f32[][]cpu" = torch.ops.aten.as_strided.default(arg0_1, [], [], 1)
        foo_default = torch.ops.mylib.foo.default(as_strided_default, as_strided_default_1);  \
as_strided_default = as_strided_default_1 = foo_default = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  copy_ = None
        select_2: "f32[][]cpu" = torch.ops.aten.select.int(arg0_1, 0, 0)
        select_3: "f32[][]cpu" = torch.ops.aten.select.int(arg0_1, 0, 1);  arg0_1 = None
        return (select_2, select_3)""",
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

    # foo takes a mutable list with views in addition to other args.
    def test_auto_functionalize_extra4(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!)[] y) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo_impl(x, y):
                x.sin_()
                y[0].sin_()

            def f(x, y, z):
                a = x[0]
                b = z[0]
                torch.ops.mylib.foo(a, [b, y])

            orig_args = [torch.randn(2), torch.randn(2), torch.randn(2)]

            [aot_eager_args, result1, graph_aot] = self.run_aot_eager(f, orig_args)
            [inductor_args, result2, graph_inductor] = self.run_inductor(f, orig_args)
            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            result3 = f(*eager_args)

            self.assertEqual(inductor_args[2], eager_args[2])
            self.assertEqual(inductor_args, aot_eager_args)

            self.assertEqual(result3, result1)
            self.assertEqual(result3, result2)

            if torch._dynamo.config.assume_static_by_default:
                self.assertExpectedInline(
                    graph_aot,
                    """\
def forward(self, arg0_1: "f32[2][1]cpu", arg1_1: "f32[2][1]cpu", arg2_1: "f32[2][1]cpu"):
        auto_functionalized = torch.ops.higher_order.auto_functionalized(torch.ops.mylib.foo.default, \
_x_base_index = 0, _x_size = (), _x_stride = (), _x_storage_offset = 0, _y_length = 2, _y_0_base_index = 1, \
_y_0_size = (), _y_0_stride = (), _y_0_storage_offset = 0, _y_1_base_index = 2, _all_bases = [arg0_1, arg1_1, arg2_1])
        getitem_1: "f32[2][1]cpu" = auto_functionalized[1]
        getitem_2: "f32[2][1]cpu" = auto_functionalized[2]
        getitem_3: "f32[2][1]cpu" = auto_functionalized[3];  auto_functionalized = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = getitem_1 = copy_ = None
        copy__1: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg1_1, getitem_2);  arg1_1 = getitem_2 = copy__1 = None
        copy__2: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg2_1, getitem_3);  arg2_1 = getitem_3 = copy__2 = None
        return ()""",
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

            # 2. Run with inductor backend

            if torch._dynamo.config.assume_static_by_default:
                self.assertExpectedInline(
                    graph_inductor,
                    """\
def forward(self, arg0_1: "f32[2][1]cpu", arg1_1: "f32[2][1]cpu", arg2_1: "f32[2][1]cpu"):
        as_strided_default: "f32[][]cpu" = torch.ops.aten.as_strided.default(arg0_1, [], [], 0)
        as_strided_default_1: "f32[][]cpu" = torch.ops.aten.as_strided.default(arg1_1, [], [], 0)
        foo_default = torch.ops.mylib.foo.default(as_strided_default, [as_strided_default_1, arg2_1]);  \
as_strided_default = as_strided_default_1 = foo_default = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  arg0_1 = copy_ = None
        copy__1: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy__1 = None
        copy__2: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg2_1, arg2_1);  arg2_1 = copy__2 = None
        return ()""",
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

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
        foo_default = torch.ops.mylib.foo.default(None, [arg2_1, arg3_1], arg1_1, 2, arg0_1);  \
arg2_1 = arg3_1 = arg0_1 = foo_default = None

        copy_: "f32[3][1]cpu" = torch.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy_ = None
        return ()""",
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            f(*eager_args)
            self.assertEqual(compiled_args, eager_args)

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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
