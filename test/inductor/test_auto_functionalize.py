# Owner(s): ["module: functionalization"]

import unittest

import numpy as np

import torch
import torch._dynamo.testing
import torch._inductor.config as inductor_config
import torch._inductor.test_case
import torch.utils._pytree as pytree
import torch.utils.cpp_extension
from torch import Tensor
from torch._dynamo.testing import CompileCounterWithBackend
from torch._dynamo.utils import counters
from torch._higher_order_ops.auto_functionalize import try_use_slice
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    expectedIfCppFakeTensor,
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.logging_utils import logs_to_string


class AutoFunctionalizeTests(torch._inductor.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

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

    @parametrize("enable_auto_functionalized_v2", [False, True])
    @parametrize(
        "case",
        ["e8m0", "complex", "meta", "sparse", "fp8", "fp8_uint8", "cpu_output"],
    )
    def test_auto_functionalize_decomposition_before_type_fallback(
        self, enable_auto_functionalized_v2, case
    ):
        from contextlib import ExitStack
        from unittest import mock

        import torch._inductor.codegen.triton_utils as triton_utils
        import torch._inductor.lowering as lowering
        from torch._inductor.fx_passes.post_grad import decompose_auto_functionalized
        from torch._subclasses.functional_tensor import PythonFunctionalizeAPI
        from torch.fx.experimental.proxy_tensor import make_fx

        dtype = {
            "e8m0": torch.float8_e8m0fnu,
            "complex": torch.complex64,
            "fp8": torch.float8_e4m3fn,
            "fp8_uint8": torch.float8_e4m3fn,
        }.get(case, torch.float32)
        device = "meta" if case == "meta" else "cpu"
        src = torch.tensor([2.0, 4.0], device=device).to(dtype)
        out = torch.ones(2, device=device).to(dtype)
        if case == "sparse":
            src = src.to_sparse()

        with (
            inductor_config.patch(
                enable_auto_functionalized_v2=enable_auto_functionalized_v2,
                disable_cpp_codegen=case == "cpu_output",
            ),
            torch.library._scoped_library("mylib", "FRAGMENT") as lib,
            ExitStack() as stack,
        ):
            lib.define(
                "foo(Tensor src, Tensor(a!) out) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            def foo_impl(src, out):
                out.copy_(src)

            @torch.library.register_fake("mylib::foo", lib=lib)
            def foo_fake(src, out):
                return None

            if case in ("fp8", "fp8_uint8"):
                # Exercise both unsupported-FP8 paths without requiring a GPU.
                stack.enter_context(
                    mock.patch.object(
                        lowering,
                        "is_triton_fp8_dtype_supported",
                        return_value=False,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        triton_utils,
                        "use_uint8_triton_storage_for_cuda_float8_e4m3fn",
                        return_value=case == "fp8_uint8",
                    )
                )

            def f(src, out):
                torch.ops.mylib.foo(src, out)
                return out

            gm = make_fx(
                PythonFunctionalizeAPI().functionalize(f), tracing_mode="fake"
            )(src, out)
            hop = (
                torch.ops.higher_order.auto_functionalized_v2
                if enable_auto_functionalized_v2
                else torch.ops.higher_order.auto_functionalized
            )
            self.assertEqual(
                len(gm.graph.find_nodes(op="call_function", target=hop)), 1
            )
            decompose_auto_functionalized(gm.graph)
            mutable_nodes = gm.graph.find_nodes(
                op="call_function", target=torch.ops.mylib.foo.default
            )
            self.assertEqual(len(mutable_nodes), 1)

            if case in ("e8m0", "complex"):
                eager_out = out.clone()
                compiled_out = out.clone()
                f(src, eager_out)
                torch.compile(f, backend="inductor", fullgraph=True)(src, compiled_out)

                # Compare bytes because E8M0 does not support arithmetic comparisons.
                self.assertEqual(
                    compiled_out.view(torch.uint8), eager_out.view(torch.uint8)
                )
                self.assertEqual(compiled_out.view(torch.uint8), src.view(torch.uint8))

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

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("a(Tensor(a!) x) -> None")
            lib.impl("a", lambda x: None, "Functionalize")
            self.assertFalse(can_auto_functionalize(torch.ops.mylib.a.default))

    @torch._inductor.config.patch(enable_auto_functionalized_v2=False)
    def test_auto_functionalize_old(self):
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
def forward(self, arg0_1: "f32[3][1]cpu", arg1_1: "f32[3][1]cpu", arg2_1: "f32[3][1]cpu", arg3_1: "f32[3][1]cpu", arg4_1: "f32[3][1]cpu"):
        foo_default = torch.ops.mylib.foo.default(arg1_1, [arg2_1, arg3_1], arg4_1, 2, arg0_1);  arg1_1 = arg2_1 = arg3_1 = arg4_1 = arg0_1 = foo_default = None
        return ()""",
                    ignore_comments=True,
                )

                # stack trace should be in post_grad_graph
                self.assertTrue(
                    "code: torch.ops.mylib.foo(x, y, z, 2, n)" in post_grad_graphs,
                )

            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            f(*eager_args)
            self.assertEqual(compiled_args, eager_args)

    @torch._inductor.config.patch(enable_auto_functionalized_v2=False)
    def test_auto_functionalize_with_returns_old(self):
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

            @torch.library.register_fake("mylib::foo", lib=lib)
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
        foo_default = torch.ops.mylib.foo.default(arg1_1, [arg2_1, arg3_1], arg4_1, 2, arg0_1);  arg1_1 = arg2_1 = arg3_1 = arg4_1 = arg0_1 = None
        getitem_4: "f32[3][1]cpu" = foo_default[0]
        getitem_5: "f32[3][1]cpu" = foo_default[1];  foo_default = None
        return (getitem_4, getitem_5)""",
                    ignore_comments=True,
                )

            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            eager_out = f(*eager_args)
            self.assertEqual(compiled_args, eager_args)
            self.assertEqual(compiled_out, eager_out)

    def test_auto_functionalize_on_view(self):
        for value in [True, False]:
            with (
                torch.library._scoped_library("mylib", "FRAGMENT") as lib,
                inductor_config.patch({"enable_auto_functionalized_v2": value}),
            ):
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
                if not torch.allclose(x, expected):
                    raise AssertionError

                @torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
                def f(x):
                    x = x.clone()
                    y = x[:]
                    torch.ops.mylib.foo(y)
                    return x

                y = f(x)
                self.assertEqual(y, x.sin())

    @torch._inductor.config.patch(enable_auto_functionalized_v2=False)
    def test_auto_functionalize_optional_old(self):
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
        foo_default = torch.ops.mylib.foo.default(None, [arg1_1, arg2_1], arg3_1, 2, arg0_1);  arg1_1 = arg2_1 = arg3_1 = arg0_1 = foo_default = None
        return ()""",
                    ignore_comments=True,
                )

                # stack trace should be in post_grad_graph
                self.assertTrue(
                    "code: torch.ops.mylib.foo(x, y, z, 2, n)" in post_grad_graphs,
                )

            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            f(*eager_args)
            self.assertEqual(compiled_args, eager_args)

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    @parametrize("increase_gb", (None, 0.0))
    def test_unbacked_auto_functionalize_op(self, increase_gb):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("mk_image(Tensor(a!) decoder) -> Tensor")

            @torch.library.impl("mylib::mk_image", "CPU", lib=lib)
            def mk_image(decoder: Tensor) -> Tensor:
                return torch.randn(2, 3, 4, 5)

            @torch.library.register_fake("mylib::mk_image", lib=lib)
            def _(decoder: Tensor) -> Tensor:
                image_size = [
                    torch.library.get_ctx().new_dynamic_size() for _ in range(4)
                ]
                return torch.empty(image_size)

            @torch.compile(fullgraph=True)
            def f(x):
                return torch.ops.mylib.mk_image.default(x)

            x = torch.zeros(100, dtype=torch.int64)
            with inductor_config.patch(
                fusion_memory_timeline_peak_memory_increase_gb=increase_gb
            ):
                f(x)

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_auto_functionalize_v2(self, _dynamic=False):
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
def forward(self, arg0_1: "f32[s77][1]cpu", arg1_1: "f32[s77][1]cpu", arg2_1: "Sym(s77)", arg3_1: "f32[s77][1]cpu", arg4_1: "f32[s77][1]cpu", arg5_1: "f32[s77][1]cpu"):
        foo_default = torch.ops.mylib.foo.default(arg1_1, [arg3_1, arg4_1], arg5_1, 2, arg0_1);  arg3_1 = arg4_1 = arg0_1 = foo_default = None
        copy_: "f32[s77][1]cpu" = torch.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy_ = None
        copy__1: "f32[s77][1]cpu" = torch.ops.aten.copy_.default(arg5_1, arg5_1);  arg5_1 = copy__1 = None
        return ()""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        post_grad_graphs,
                        """\
def forward(self, arg0_1: "f32[3][1]cpu", arg1_1: "f32[3][1]cpu", arg2_1: "f32[3][1]cpu", arg3_1: "f32[3][1]cpu", arg4_1: "f32[3][1]cpu"):
        foo_default = torch.ops.mylib.foo.default(arg1_1, [arg2_1, arg3_1], arg4_1, 2, arg0_1);  arg2_1 = arg3_1 = arg0_1 = foo_default = None
        copy_: "f32[3][1]cpu" = torch.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy_ = None
        copy__1: "f32[3][1]cpu" = torch.ops.aten.copy_.default(arg4_1, arg4_1);  arg4_1 = copy__1 = None
        return ()""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            f(*eager_args)
            self.assertEqual(compiled_args, eager_args)

    def run_aot_eager(self, f, orig_args, _dynamic=False):
        aot_eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)

        log_stream, ctx = logs_to_string(
            "torch._functorch._aot_autograd.graph_capture", "aot_graphs"
        )

        result = None
        with ctx():
            result = torch.compile(
                f, backend="aot_eager", fullgraph=True, dynamic=_dynamic
            )(*aot_eager_args)

            graph = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        return [aot_eager_args, result, graph]

    def run_inductor(
        self,
        f,
        orig_args,
        _dynamic=False,
        log_module="torch._inductor.compile_fx",
        log_function="post_grad_graphs",
    ):
        compiled_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)

        log_stream, ctx = logs_to_string(log_module, log_function)
        result = None
        with ctx():
            result = torch.compile(
                f, backend="inductor", fullgraph=True, dynamic=_dynamic
            )(*compiled_args)

            graph = "\n".join(log_stream.getvalue().strip().split("\n")[3:]).strip()

        return [compiled_args, result, graph]

    def test_cumulative_out_preserves_out_dtype_under_compile(self):
        test_cases = [
            (torch.cumsum, torch.tensor([0.9201, 0.1166], dtype=torch.float32)),
            (torch.cumprod, torch.tensor([2.0, 0.6], dtype=torch.float32)),
        ]

        for op, x in test_cases:
            with self.subTest(op=op.__name__):

                def f(out, x):
                    return op(x, 0, out=out)

                eager_out = torch.tensor([0, 0], dtype=torch.int32)
                aot_eager_out = eager_out.clone()
                inductor_out = eager_out.clone()

                eager_ret = f(eager_out, x)
                aot_eager_ret = torch.compile(f, backend="aot_eager", fullgraph=True)(
                    aot_eager_out, x
                )
                inductor_ret = torch.compile(f, backend="inductor", fullgraph=True)(
                    inductor_out, x
                )

                self.assertEqual(aot_eager_out, eager_out)
                self.assertEqual(inductor_out, eager_out)
                self.assertEqual(aot_eager_ret, eager_ret)
                self.assertEqual(inductor_ret, eager_ret)

    def test_out_return_preserves_view_after_graph_break(self):
        sz = 9

        def f(data, other1, other2):
            inp = data[:sz]
            output = data[:sz]
            torch._dynamo.graph_break()
            expected = torch.empty_like(output)
            torch.addcmul(
                inp,
                other1.view(inp.shape),
                other2.view(inp.shape),
                out=expected,
            )
            result = torch.addcmul(
                inp,
                other1.view(inp.shape),
                other2.view(inp.shape),
                out=output,
            )
            return output, result, expected

        data = torch.randn(2 * sz, dtype=torch.float64)
        other1 = torch.randn(sz, dtype=torch.float64)
        other2 = torch.randn(sz, dtype=torch.float64)

        output, result, expected = torch.compile(f, backend="aot_eager")(
            data, other1, other2
        )

        self.assertEqual(output.shape, (sz,))
        self.assertEqual(result.shape, (sz,))
        self.assertEqual(expected.shape, (sz,))
        self.assertEqual(result.data_ptr(), output.data_ptr())
        self.assertEqual(output, expected)
        self.assertEqual(result, expected)

        def same_metadata_view(data, other):
            inp = data.view(data.shape)
            output = data.view(data.shape)
            torch._dynamo.graph_break()
            expected = torch.empty_like(output)
            torch.add(inp, other, out=expected)
            result = torch.add(inp, other, out=output)
            return data, output, result, expected

        base, output, result, expected = torch.compile(
            same_metadata_view, backend="aot_eager"
        )(torch.randn(sz), torch.randn(sz))

        self.assertIs(output._base, base)
        self.assertIs(result._base, base)
        self.assertEqual(result.data_ptr(), output.data_ptr())
        self.assertEqual(output, expected)
        self.assertEqual(result, expected)

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_auto_functionalize_with_returns_v2(self):
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

            @torch.library.register_fake("mylib::foo", lib=lib)
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
        foo_default = torch.ops.mylib.foo.default(arg1_1, [arg2_1, arg3_1], arg4_1, 2, arg0_1);  arg2_1 = arg3_1 = arg0_1 = None
        getitem_4: "f32[3][1]cpu" = foo_default[0]
        getitem_5: "f32[3][1]cpu" = foo_default[1];  foo_default = None
        copy_: "f32[3][1]cpu" = torch.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy_ = None
        copy__1: "f32[3][1]cpu" = torch.ops.aten.copy_.default(arg4_1, arg4_1);  arg4_1 = copy__1 = None
        return (getitem_4, getitem_5)""",
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            eager_out = f(*eager_args)
            self.assertEqual(compiled_args, eager_args)
            self.assertEqual(compiled_out, eager_out)

    # foo takes two inputs that are not views.
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
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
def forward(self, arg0_1: "f32[s77][1]cpu", arg1_1: "Sym(s77)", arg2_1: "f32[s77][1]cpu"):
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _y_base_index = 1, _all_bases = [arg0_1, arg2_1])
        getitem_1: "f32[s77][1]cpu" = auto_functionalized_v2[1]
        getitem_2: "f32[s77][1]cpu" = auto_functionalized_v2[2];  auto_functionalized_v2 = None
        add: "f32[s77][1]cpu" = torch.ops.aten.add.Tensor(getitem_1, getitem_2)
        copy_: "f32[s77][1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = getitem_1 = copy_ = None
        copy__1: "f32[s77][1]cpu" = torch.ops.aten.copy_.default(arg2_1, getitem_2);  arg2_1 = getitem_2 = copy__1 = None
        return (add,)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu", arg1_1: "f32[2][1]cpu"):
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _y_base_index = 1, _all_bases = [arg0_1, arg1_1])
        getitem_1: "f32[2][1]cpu" = auto_functionalized_v2[1]
        getitem_2: "f32[2][1]cpu" = auto_functionalized_v2[2];  auto_functionalized_v2 = None
        add: "f32[2][1]cpu" = torch.ops.aten.add.Tensor(getitem_1, getitem_2)
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = getitem_1 = copy_ = None
        copy__1: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg1_1, getitem_2);  arg1_1 = getitem_2 = copy__1 = None
        return (add,)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

            if torch._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[s77][1]cpu", arg1_1: "Sym(s77)", arg2_1: "f32[s77][1]cpu"):
        foo_default = torch.ops.mylib.foo.default(arg0_1, arg2_1);  foo_default = None
        add: "f32[s77][1]cpu" = torch.ops.aten.add.Tensor(arg0_1, arg2_1)
        copy_: "f32[s77][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  arg0_1 = copy_ = None
        copy__1: "f32[s77][1]cpu" = torch.ops.aten.copy_.default(arg2_1, arg2_1);  arg2_1 = copy__1 = None
        return (add,)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu", arg1_1: "f32[2][1]cpu"):
        foo_default = torch.ops.mylib.foo.default(arg0_1, arg1_1);  foo_default = None
        add: "f32[2][1]cpu" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  arg0_1 = copy_ = None
        copy__1: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy__1 = None
        return (add,)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

    # foo takes two views on the same input, function does not have return.
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
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
def forward(self, arg0_1: "f32[s77][1]cpu", arg1_1: "Sym(s77)"):
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_size = (), _x_stride = (), _x_storage_offset = 0, _y_base_index = 0, _y_size = (), _y_stride = (), _y_storage_offset = 1, _all_bases = [arg0_1])
        getitem_1: "f32[s77][1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[s77][1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = getitem_1 = copy_ = None
        return ()""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_size = (), _x_stride = (), _x_storage_offset = 0, _y_base_index = 0, _y_size = (), _y_stride = (), _y_storage_offset = 1, _all_bases = [arg0_1])
        getitem_1: "f32[2][1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
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
def forward(self, arg0_1: "f32[s77][1]cpu", arg1_1: "Sym(s77)"):
        as_strided_default: "f32[][]cpu" = torch.ops.aten.as_strided.default(arg0_1, [], [], 0)
        as_strided_default_1: "f32[][]cpu" = torch.ops.aten.as_strided.default(arg0_1, [], [], 1)
        foo_default = torch.ops.mylib.foo.default(as_strided_default, as_strided_default_1);  as_strided_default = as_strided_default_1 = foo_default = None
        copy_: "f32[s77][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  arg0_1 = copy_ = None
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
        foo_default = torch.ops.mylib.foo.default(as_strided_default, as_strided_default_1);  as_strided_default = as_strided_default_1 = foo_default = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  arg0_1 = copy_ = None
        return ()""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

    # foo takes two views on the same input, function returns both views and the input
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
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
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_size = (), _x_stride = (), _x_storage_offset = 0, _y_base_index = 0, _y_size = (), _y_stride = (), _y_storage_offset = 1, _all_bases = [arg0_1])
        getitem_1: "f32[2][1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
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
        foo_default = torch.ops.mylib.foo.default(as_strided_default, as_strided_default_1);  as_strided_default = as_strided_default_1 = foo_default = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  copy_ = None
        select_2: "f32[][]cpu" = torch.ops.aten.select.int(arg0_1, 0, 0)
        select_3: "f32[][]cpu" = torch.ops.aten.select.int(arg0_1, 0, 1);  arg0_1 = None
        return (select_2, select_3)""",
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

    # foo takes x, y both being graph inputs and views of the same shared base but do not overlap.
    # In this special case functionlization will have none as base for x and y. so they will be assumed
    # to have unique bases during functionalizations. During inplace, we notice that they both share storage
    # but because their memory does not overlap we can inplace both. see github issue #139628
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_auto_functionalize_extra5(self):
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
                return torch.ops.mylib.foo(x, y)

            base = torch.randn(2, 2)
            orig_args = [base[0], base[1]]

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
def forward(self, arg0_1: "f32[2][1]cpu", arg1_1: "f32[2][1]cpu"):
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _y_base_index = 1, _all_bases = [arg0_1, arg1_1])
        getitem_1: "f32[2][1]cpu" = auto_functionalized_v2[1]
        getitem_2: "f32[2][1]cpu" = auto_functionalized_v2[2];  auto_functionalized_v2 = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = getitem_1 = copy_ = None
        copy__1: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg1_1, getitem_2);  arg1_1 = getitem_2 = copy__1 = None
        return ()""",
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

            # 2. Run with inductor backend
            if torch._dynamo.config.assume_static_by_default:
                self.assertExpectedInline(
                    graph_inductor,
                    """\
def forward(self, arg0_1: "f32[2][1]cpu", arg1_1: "f32[2][1]cpu"):
        foo_default = torch.ops.mylib.foo.default(arg0_1, arg1_1);  foo_default = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  arg0_1 = copy_ = None
        copy__1: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy__1 = None
        return ()""",
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

    # foo takes a mutable list with views in addition to other args.
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
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
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_size = (), _x_stride = (), _x_storage_offset = 0, _y_length = 2, _y_0_base_index = 1, _y_0_size = (), _y_0_stride = (), _y_0_storage_offset = 0, _y_1_base_index = 2, _all_bases = [arg0_1, arg2_1, arg1_1])
        getitem_1: "f32[2][1]cpu" = auto_functionalized_v2[1]
        getitem_2: "f32[2][1]cpu" = auto_functionalized_v2[2]
        getitem_3: "f32[2][1]cpu" = auto_functionalized_v2[3];  auto_functionalized_v2 = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = getitem_1 = copy_ = None
        copy__1: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg1_1, getitem_3);  arg1_1 = getitem_3 = copy__1 = None
        copy__2: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg2_1, getitem_2);  arg2_1 = getitem_2 = copy__2 = None
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
        as_strided_default_1: "f32[][]cpu" = torch.ops.aten.as_strided.default(arg2_1, [], [], 0)
        foo_default = torch.ops.mylib.foo.default(as_strided_default, [as_strided_default_1, arg1_1]);  as_strided_default = as_strided_default_1 = foo_default = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  arg0_1 = copy_ = None
        copy__1: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg1_1, arg1_1);  arg1_1 = copy__1 = None
        copy__2: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg2_1, arg2_1);  arg2_1 = copy__2 = None
        return ()""",
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_auto_functionalize_optional_v2(self):
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
        foo_default = torch.ops.mylib.foo.default(None, [arg1_1, arg2_1], arg3_1, 2, arg0_1);  arg1_1 = arg2_1 = arg0_1 = foo_default = None
        copy_: "f32[3][1]cpu" = torch.ops.aten.copy_.default(arg3_1, arg3_1);  arg3_1 = copy_ = None
        return ()""",
                    ignore_comments=True,
                    ignore_empty_lines=True,
                )

            eager_args = pytree.tree_map_only(torch.Tensor, torch.clone, orig_args)
            f(*eager_args)
            self.assertEqual(compiled_args, eager_args)

    @torch._inductor.config.patch(enable_auto_functionalized_v2=False)
    def test_inference_mode1_v2(self):
        with torch.inference_mode():
            self.test_auto_functionalize_extra1()

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_inference_mode2_v2(self):
        with torch.inference_mode():
            self.test_auto_functionalize_extra2()

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_inference_mode3_v2(self):
        with torch.inference_mode():
            self.test_auto_functionalize_extra3()

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_inference_mode4_v2(self):
        with torch.inference_mode():
            self.test_auto_functionalize_extra4()

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_dynamic_v2(self):
        self.test_auto_functionalize_v2(_dynamic=True)

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_dynamic2_v2(self):
        self.test_auto_functionalize_extra1(_dynamic=True)

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_dynamic3_v2(self):
        self.test_auto_functionalize_extra2(_dynamic=True)

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_graph_input_is_view(self):
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
                pass

            @torch.compile(fullgraph=True, dynamic=False, backend="aot_eager")
            def f(x):
                a = x[0]
                torch.ops.mylib.foo(a)
                return

            x = torch.tensor([[1, 2], [3, 4]])
            # This would fail if auto_functionalized_v2 uses clone and not clone_preserve_strides
            # to clone not-inplaced args.
            f(x[1])

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_graph_input_view_storage_offset(self):
        def f_as_strided(x):
            return torch.as_strided(x, (2,), (1,), 0).clone()

        opt_f_as_strided = torch.compile(
            f_as_strided, fullgraph=True, dynamic=False, backend="inductor"
        )
        base = torch.arange(8, dtype=torch.float32)
        self.assertEqual(opt_f_as_strided(base[2:4]), f_as_strided(base[2:4]))

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
                x.add_(10)

            def f(x):
                a = x[0]
                torch.ops.mylib.foo(a)
                return x

            opt_f = torch.compile(f, fullgraph=True, dynamic=False, backend="inductor")

            eager_base = torch.arange(8, dtype=torch.float32).reshape(4, 2)
            result_eager = f(eager_base[1])

            inductor_base = torch.arange(8, dtype=torch.float32).reshape(4, 2)
            result_inductor = opt_f(inductor_base[1])

            self.assertEqual(inductor_base, eager_base)
            self.assertEqual(result_inductor, result_eager)

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_alias(self, _dynamic=False):
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
                a = torch.ops.aten.alias.default(x)
                b = torch.ops.aten.alias.default(x)
                torch.ops.mylib.foo(a, b)
                return (a, b, x)

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
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_alias = True, _y_base_index = 0, _y_alias = True, _all_bases = [arg1_1])
        getitem_1: "f32[s0][1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[s0][1]cpu" = torch.ops.aten.copy_.default(arg1_1, getitem_1);  arg1_1 = copy_ = None
        alias_2: "f32[s0][1]cpu" = torch.ops.aten.alias.default(getitem_1)
        alias_3: "f32[s0][1]cpu" = torch.ops.aten.alias.default(getitem_1);  getitem_1 = None
        return (alias_2, alias_3)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_alias = True, _y_base_index = 0, _y_alias = True, _all_bases = [arg0_1])
        getitem_1: "f32[2][1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = copy_ = None
        alias_2: "f32[2][1]cpu" = torch.ops.aten.alias.default(getitem_1)
        alias_3: "f32[2][1]cpu" = torch.ops.aten.alias.default(getitem_1);  getitem_1 = None
        return (alias_2, alias_3)""",
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
        alias_default: "f32[s0][1]cpu" = torch.ops.aten.alias.default(arg1_1)
        alias_default_1: "f32[s0][1]cpu" = torch.ops.aten.alias.default(arg1_1)
        foo_default = torch.ops.mylib.foo.default(alias_default, alias_default_1);  \
alias_default = alias_default_1 = foo_default = None
        copy_: "f32[s0][1]cpu" = torch.ops.aten.copy_.default(arg1_1, arg1_1);  copy_ = None
        return (arg1_1, arg1_1)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        alias_default: "f32[2][1]cpu" = torch.ops.aten.alias.default(arg0_1)
        alias_default_1: "f32[2][1]cpu" = torch.ops.aten.alias.default(arg0_1)
        foo_default = torch.ops.mylib.foo.default(alias_default, alias_default_1);  \
alias_default = alias_default_1 = foo_default = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  copy_ = None
        return (arg0_1, arg0_1)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

    # Test that slice view is generated instead of as_strided when split is used.
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_split(self, _dynamic=False):
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
                splits = x.split([4, 6], dim=1)
                a = splits[0]
                b = splits[1]
                torch.ops.mylib.foo(a, b)
                return (a, b, x)

            orig_args = [torch.randn(10, 10)]
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
                    # split forces a specialization on size so we dont see arg0_1 dynamic anymore.
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[10, 10][10, 1]cpu"):
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_slice_dim = 1, _x_slice_start = 0, _x_slice_end = 4, _y_base_index = 0, _y_slice_dim = 1, _y_slice_start = 4, _y_slice_end = 10, _all_bases = [arg0_1])
        getitem_3: "f32[10, 10][10, 1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[10, 10][10, 1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_3);  arg0_1 = copy_ = None
        slice_1: "f32[10, 4][10, 1]cpu" = torch.ops.aten.slice.Tensor(getitem_3, 1, 0, 4)
        slice_2: "f32[10, 6][10, 1]cpu" = torch.ops.aten.slice.Tensor(getitem_3, 1, 4, 10);  getitem_3 = None
        return (slice_1, slice_2)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[10, 10][10, 1]cpu"):
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_slice_dim = 1, _x_slice_start = 0, _x_slice_end = 4, _y_base_index = 0, _y_slice_dim = 1, _y_slice_start = 4, _y_slice_end = 10, _all_bases = [arg0_1])
        getitem_3: "f32[10, 10][10, 1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[10, 10][10, 1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_3);  arg0_1 = copy_ = None
        slice_1: "f32[10, 4][10, 1]cpu" = torch.ops.aten.slice.Tensor(getitem_3, 1, 0, 4)
        slice_2: "f32[10, 6][10, 1]cpu" = torch.ops.aten.slice.Tensor(getitem_3, 1, 4, 10);  getitem_3 = None
        return (slice_1, slice_2)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

            # 2. Run with inductor backend
            if torch._dynamo.config.assume_static_by_default:
                if _dynamic:
                    # split forces a specialization on size so we dont see arg0_1 dynamic anymore.
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[10, 10][10, 1]cpu"):
        slice_tensor: "f32[10, 4][10, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, 4)
        slice_tensor_1: "f32[10, 6][10, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 1, 4, 10)
        foo_default = torch.ops.mylib.foo.default(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = foo_default = None
        copy_: "f32[10, 10][10, 1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  copy_ = None
        slice_1: "f32[10, 4][10, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, 4)
        slice_2: "f32[10, 6][10, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 1, 4, 10);  arg0_1 = None
        return (slice_1, slice_2)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[10, 10][10, 1]cpu"):
        slice_tensor: "f32[10, 4][10, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, 4)
        slice_tensor_1: "f32[10, 6][10, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 1, 4, 10)
        foo_default = torch.ops.mylib.foo.default(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = foo_default = None
        copy_: "f32[10, 10][10, 1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  copy_ = None
        slice_1: "f32[10, 4][10, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, 4)
        slice_2: "f32[10, 6][10, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 1, 4, 10);  arg0_1 = None
        return (slice_1, slice_2)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

    # Note that split force the input tensor to get specialized. So we do not see SymInts when _dynamic=True.
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_split_dynamic(self):
        self.test_split(_dynamic=True)

    # Test that slice view is generated instead of as_strided when slice is used.
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_slice(self, _dynamic=False):
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
                a = torch.ops.aten.slice.Tensor(x, 0, 0, 2)
                b = torch.ops.aten.slice.Tensor(x, 1, 3, 4)
                torch.ops.mylib.foo(a, b)
                return (a, b, x)

            orig_args = [torch.randn(10, 10)]
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
def forward(self, arg0_1: "f32[s77, s77][s77, 1]cpu", arg1_1: "Sym(s77)"):
        floordiv: "Sym(0)" = 0 // arg1_1;  arg1_1 = None
        add: "Sym(2)" = floordiv + 2
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_slice_dim = 0, _x_slice_start = floordiv, _x_slice_end = add, _y_base_index = 0, _y_slice_dim = 1, _y_slice_start = 3, _y_slice_end = 4, _all_bases = [arg0_1]);  floordiv = add = None
        getitem_1: "f32[s77, s77][s77, 1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[s77, s77][s77, 1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = copy_ = None
        slice_3: "f32[2, s77][s77, 1]cpu" = torch.ops.aten.slice.Tensor(getitem_1, 0, 0, 2)
        slice_4: "f32[s77, 1][s77, 1]cpu" = torch.ops.aten.slice.Tensor(getitem_1, 1, 3, 4);  getitem_1 = None
        return (slice_3, slice_4)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[10, 10][10, 1]cpu"):
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_slice_dim = 0, _x_slice_start = 0, _x_slice_end = 2, _y_base_index = 0, _y_slice_dim = 1, _y_slice_start = 3, _y_slice_end = 4, _all_bases = [arg0_1])
        getitem_1: "f32[10, 10][10, 1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[10, 10][10, 1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = copy_ = None
        slice_3: "f32[2, 10][10, 1]cpu" = torch.ops.aten.slice.Tensor(getitem_1, 0, 0, 2)
        slice_4: "f32[10, 1][10, 1]cpu" = torch.ops.aten.slice.Tensor(getitem_1, 1, 3, 4);  getitem_1 = None
        return (slice_3, slice_4)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

            # 2. Run with inductor backend
            if torch._dynamo.config.assume_static_by_default:
                if _dynamic:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[s77, s77][s77, 1]cpu", arg1_1: "Sym(s77)"):
        floordiv: "Sym(0)" = 0 // arg1_1;  arg1_1 = None
        add: "Sym(2)" = floordiv + 2;  floordiv = add = None
        slice_tensor: "f32[2, s77][s77, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, 2)
        slice_tensor_1: "f32[s77, 1][s77, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 1, 3, 4)
        foo_default = torch.ops.mylib.foo.default(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = foo_default = None
        copy_: "f32[s77, s77][s77, 1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  copy_ = None
        slice_3: "f32[2, s77][s77, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, 2)
        slice_4: "f32[s77, 1][s77, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 1, 3, 4);  arg0_1 = None
        return (slice_3, slice_4)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[10, 10][10, 1]cpu"):
        slice_tensor: "f32[2, 10][10, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, 2)
        slice_tensor_1: "f32[10, 1][10, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 1, 3, 4)
        foo_default = torch.ops.mylib.foo.default(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = foo_default = None
        copy_: "f32[10, 10][10, 1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  copy_ = None
        slice_3: "f32[2, 10][10, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, 2)
        slice_4: "f32[10, 1][10, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 1, 3, 4);  arg0_1 = None
        return (slice_3, slice_4)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

    # Note that split force the input tensor to get specialized. So we do not see SymInts when _dynamic=True.
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_slice_dynamic(self):
        self.test_slice(_dynamic=True)

    def test_try_use_slice(self):
        def test_round_trip(base, tensor):
            (dim, start, end) = try_use_slice(base, tensor)
            sliced = torch.ops.aten.slice.Tensor(base, dim, start, end)
            self.assertEqual(sliced, tensor)

        t = torch.tensor([[2, 2], [3, 4]])
        test_round_trip(t, t)

        for dim in range(-1, 1):
            f = t.split(2, dim)
            test_round_trip(t, f[0])

        for dim in range(-1, 1):
            f = t.split(1, dim)
            test_round_trip(t, f[0])
            test_round_trip(t, f[1])

        t = torch.randint(1, 10, (3, 3, 3))
        test_round_trip(t, t)

        for dim in range(-3, 3):
            f = t.split([1, 2], dim)
            test_round_trip(t, f[0])
            test_round_trip(t, f[1])

        for dim in range(-3, 3):
            f = t.split(1, dim)
            test_round_trip(t, f[0])
            test_round_trip(t, f[1])
            test_round_trip(t, f[2])

        t = torch.rand(10, 10, 10)
        test_round_trip(t, t)
        for dim in range(-3, 3):
            f = t.split([2, 2, 6], dim)
            test_round_trip(t, f[0])
            test_round_trip(t, f[1])
            test_round_trip(t, f[2])

        # example where slice won't work

        # selection
        t = torch.ones(10)
        b = t[0]
        self.assertEqual(try_use_slice(t, b), None)

        t = torch.tensor([[1, 2], [3, 4]])
        self.assertEqual(try_use_slice(t, t[0]), None)
        self.assertEqual(try_use_slice(t, t[1]), None)

        t = torch.tensor(
            [
                [[1, 2, 3, 4, 5, 6, 7, 8], [10, 11, 12, 13, 14, 15, 16, 17]],
                [[71, 72, 73, 74, 75, 76, 77, 78], [81, 82, 83, 84, 85, 86, 87, 88]],
            ]
        )

        self.assertEqual(try_use_slice(t, t[0:1, 0:1, :7]), None)
        self.assertEqual(try_use_slice(t, t[0:1, 0:2, :3]), None)
        self.assertEqual(try_use_slice(t, t[0:2, 1, 0:8]), None)

        # simple slice operations are supported
        test_round_trip(t, t[0:2])
        test_round_trip(t, t[3:4])

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_alias2(self, _dynamic=False):
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
                a = torch.ops.aten.alias.default(x)
                b = x.clone()
                c = b.nonzero().float()
                d = torch.ops.aten.slice(
                    c
                )  # d is a Tensor with unbacked Symint in the shape
                torch.ops.mylib.foo(a, d)
                return a, d

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
                    cpp_expected = """\
def forward(self, arg0_1: "f32[s77][1]cpu", arg1_1: "Sym(s77)"):
        clone: "f32[s77][1]cpu" = torch.ops.aten.clone.default(arg0_1)
        nonzero: "i64[u0, 1][1, u0]cpu" = torch.ops.aten.nonzero.default(clone);  clone = None
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(nonzero, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar = None
        _to_copy: "f32[u0, 1][1, u0]cpu" = torch.ops.aten._to_copy.default(nonzero, dtype = torch.float32);  nonzero = None
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_alias = True, _y_base_index = 1, _y_alias = True, _all_bases = [arg0_1, _to_copy]);  _to_copy = None
        getitem_1: "f32[s77][1]cpu" = auto_functionalized_v2[1]
        getitem_2: "f32[u0, 1][1, u0]cpu" = auto_functionalized_v2[2];  auto_functionalized_v2 = None
        copy_: "f32[s77][1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = copy_ = None
        alias_1: "f32[s77][1]cpu" = torch.ops.aten.alias.default(getitem_1);  getitem_1 = None
        slice_2: "f32[u0, 1][1, u0]cpu" = torch.ops.aten.slice.Tensor(getitem_2);  getitem_2 = None
        return (alias_1, slice_2)"""
                    python_expected = cpp_expected.replace(
                        "sym_size_int_1", "sym_size_int"
                    )
                    self.assertExpectedInline(
                        graph_aot,
                        expectedIfCppFakeTensor(cpp_expected, python_expected),
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_aot,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        clone: "f32[2][1]cpu" = torch.ops.aten.clone.default(arg0_1)
        nonzero: "i64[u0, 1][1, u0]cpu" = torch.ops.aten.nonzero.default(clone);  clone = None
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(nonzero, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar = None
        le: "Sym(u0 <= 2)" = sym_size_int <= 2;  sym_size_int = None
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(le, "Runtime assertion failed for expression u0 <= 2 on node 'le'");  le = _assert_scalar_1 = None
        _to_copy: "f32[u0, 1][1, u0]cpu" = torch.ops.aten._to_copy.default(nonzero, dtype = torch.float32);  nonzero = None
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.foo.default, _x_base_index = 0, _x_alias = True, _y_base_index = 1, _y_alias = True, _all_bases = [arg0_1, _to_copy]);  _to_copy = None
        getitem_1: "f32[2][1]cpu" = auto_functionalized_v2[1]
        getitem_2: "f32[u0, 1][1, u0]cpu" = auto_functionalized_v2[2];  auto_functionalized_v2 = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = copy_ = None
        alias_1: "f32[2][1]cpu" = torch.ops.aten.alias.default(getitem_1);  getitem_1 = None
        slice_2: "f32[u0, 1][1, u0]cpu" = torch.ops.aten.slice.Tensor(getitem_2);  getitem_2 = None
        return (alias_1, slice_2)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

            # 2. Run with inductor backend
            if torch._dynamo.config.assume_static_by_default:
                if _dynamic:
                    cpp_expected = """\
def forward(self, arg0_1: "f32[s77][1]cpu", arg1_1: "Sym(s77)"):
        clone: "f32[s77][1]cpu" = torch.ops.aten.clone.default(arg0_1)
        nonzero: "i64[u0, 1][1, u0]cpu" = torch.ops.aten.nonzero.default(clone);  clone = None
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(nonzero, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar = None
        convert_element_type: "f32[u0, 1][1, u0]cpu" = torch.ops.prims.convert_element_type.default(nonzero, torch.float32);  nonzero = None
        alias_default: "f32[s77][1]cpu" = torch.ops.aten.alias.default(arg0_1)
        alias_default_1: "f32[u0, 1][1, u0]cpu" = torch.ops.aten.alias.default(convert_element_type)
        foo_default = torch.ops.mylib.foo.default(alias_default, alias_default_1);  alias_default = alias_default_1 = foo_default = None
        copy_: "f32[s77][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  copy_ = None
        slice_2: "f32[u0, 1][1, u0]cpu" = torch.ops.aten.slice.Tensor(convert_element_type);  convert_element_type = None
        return (arg0_1, slice_2)"""
                    python_expected = cpp_expected.replace(
                        "sym_size_int_1", "sym_size_int"
                    )
                    self.assertExpectedInline(
                        graph_inductor,
                        expectedIfCppFakeTensor(cpp_expected, python_expected),
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )
                else:
                    self.assertExpectedInline(
                        graph_inductor,
                        """\
def forward(self, arg0_1: "f32[2][1]cpu"):
        clone: "f32[2][1]cpu" = torch.ops.aten.clone.default(arg0_1)
        nonzero: "i64[u0, 1][1, u0]cpu" = torch.ops.aten.nonzero.default(clone);  clone = None
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(nonzero, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar = None
        le: "Sym(u0 <= 2)" = sym_size_int <= 2;  sym_size_int = None
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(le, "Runtime assertion failed for expression u0 <= 2 on node 'le'");  le = _assert_scalar_1 = None
        convert_element_type: "f32[u0, 1][1, u0]cpu" = torch.ops.prims.convert_element_type.default(nonzero, torch.float32);  nonzero = None
        alias_default: "f32[2][1]cpu" = torch.ops.aten.alias.default(arg0_1)
        alias_default_1: "f32[u0, 1][1, u0]cpu" = torch.ops.aten.alias.default(convert_element_type)
        foo_default = torch.ops.mylib.foo.default(alias_default, alias_default_1);  alias_default = alias_default_1 = foo_default = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  copy_ = None
        slice_2: "f32[u0, 1][1, u0]cpu" = torch.ops.aten.slice.Tensor(convert_element_type);  convert_element_type = None
        return (arg0_1, slice_2)""",
                        ignore_comments=True,
                        ignore_empty_lines=True,
                    )

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_alias2_dynamic(self):
        self.test_alias2(_dynamic=True)

    # Test that the view regeneration optimizations do not result in recompilations. By comparing re-compilation in eager backend
    # with recompilation in inductor backend.
    @torch.fx.experimental._config.patch(use_duck_shape=False)
    def test_recompile(self):
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
                pass

            def run_and_compare(func, expected=1):
                counter_v2 = CompileCounterWithBackend("inductor")
                counter_v1 = CompileCounterWithBackend("inductor")
                v1 = torch.compile(
                    func, backend=counter_v1, fullgraph=True, dynamic=True
                )

                v2 = torch.compile(
                    func, backend=counter_v2, fullgraph=True, dynamic=True
                )
                inputs = [
                    torch.rand(10, 10),
                    torch.rand(100, 100),
                    torch.rand(10, 2),
                    torch.rand(1000, 1000),
                ]

                with torch._inductor.config.patch(enable_auto_functionalized_v2=True):
                    for input in inputs:
                        v2(input)

                torch._dynamo.reset()

                with torch._inductor.config.patch(enable_auto_functionalized_v2=False):
                    for input in inputs:
                        v1(input)

                self.assertEqual(counter_v2.frame_count, counter_v1.frame_count)

                self.assertEqual(counter_v1.frame_count, expected)

            def func(x):
                a = x[0]
                b = x[1]
                torch.ops.mylib.foo(a, b)

            run_and_compare(func)

            def func(x):
                a = torch.ops.aten.alias.default(x)
                b = torch.ops.aten.alias.default(x)
                torch.ops.mylib.foo(a, b)

            run_and_compare(func)

            def func(x):
                # last row
                a = x[x.size()[0] - 1]

                # first row
                b = x[0]
                torch.ops.mylib.foo(a, b)

            run_and_compare(func)

            def func(x):
                a = torch.ops.aten.slice.Tensor(x, 1, 3, 4)
                b = torch.ops.aten.slice.Tensor(x, 0, 1, 4)
                torch.ops.mylib.foo(a, b)

            # recompile here is not triggered by auto_functionalize
            # [__recompiles]     - 0/0: 4 <= L['x'].size()[1]  # a = torch.ops.aten.slice.Tensor(x, 1, 3, 4)
            # test/inductor/test_auto_functionalize.py:1160 in func (_decomp/decompositions.py:781 in slice_forward)
            run_and_compare(func, 2)

            def func(x):
                a = torch.ops.aten.alias.default(x)
                b = x.clone()
                c = b.nonzero().float()
                d = torch.ops.aten.slice(
                    c
                )  # d is a Tensor with unbacked Symint in the shape
                torch.ops.mylib.foo(a, d)
                return a, d

            with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
                run_and_compare(func, 1)

    # Test that the alias optimization, were alias is called instead of as_strided, preserve the fact
    # that id(x) != id(base)
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    @unittest.skip(
        reason="This test fails because something else in inductor optimize out the alias. issue #137434"
    )
    def test_alias_id_input_to_custom_op(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::not_eq",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::not_eq", "cpu", lib=lib)
            @torch._dynamo.disable
            def not_eq_impl(x, y):
                self.assertNotEqual(id(x), id(y))

            def func(x):
                a = torch.ops.aten.alias.default(x)
                torch.ops.mylib.not_eq(a, x)

            compiled = torch.compile(func, backend="inductor", fullgraph=True)
            compiled(torch.rand(2, 2))

    # Test that the alias optimization, were alias is called instead of as_strided, preserve the fact
    # that id(x) != id(base)
    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_alias_id_output(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor(a!) x, Tensor(b!) y) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo(x, y):
                pass

            def func(x):
                a = torch.ops.aten.alias.default(x)
                torch.ops.mylib.foo(a, x)
                return a

            compiled = torch.compile(func, backend="inductor", fullgraph=True)
            input = torch.rand(2, 2)
            output = compiled(torch.rand(2, 2))
            self.assertNotEqual(id(output), id(input))

    def test_inference_mode_view(self):
        @torch.library.custom_op(
            "test_inference_mode_view::foo", mutates_args={"workspace"}
        )
        def foo(x: torch.Tensor, workspace: torch.Tensor) -> torch.Tensor:
            return x.clone()

        @foo.register_fake
        def _(x, workspace):
            return x.clone()

        @torch.compile(fullgraph=True, backend="aot_eager")
        def f(x, w):
            y = foo(x, w)
            z = y.view(-1)
            return z.sin()

        x = torch.randn(2)
        w = torch.randn(2)
        with torch.inference_mode():
            y = f(x, w)
        self.assertEqual(y, x.sin())

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_scheduling_with_multiple_mutates(self):
        with torch.library._scoped_library("mylib_scheduling", "FRAGMENT") as lib:
            torch.library.define(
                "mylib_scheduling::foo",
                "(Tensor! x, Tensor! y, Tensor z) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib_scheduling::foo", "cpu", lib=lib)
            @torch._dynamo.disable
            def foo(x, y, z):
                pass

            def func(x, w):
                a = torch.empty_like(x)  # buf0
                b = torch.empty_like(x)  # buf1
                torch.ops.mylib_scheduling.foo(a, b, x)  # buf2, buf3, buf4
                c = torch.mm(a, w)  # buf5
                torch.ops.mylib_scheduling.foo(c, b, x)  # buf6, buf7, buf8
                return c

            input = torch.rand(2, 2)
            weight = torch.rand(2, 2)
            [inductor_args, output, graph_inductor] = self.run_inductor(
                func,
                [input, weight],
                False,
                "torch._inductor.scheduler",
                "compute_dependencies",
            )
            name_to_users = eval(graph_inductor)
            self.assertNotEqual(name_to_users["buf1"], name_to_users["buf5"])

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_dtype_view_clone_elimination(self):
        """
        This test verifies that dtype views (tensors sharing storage but with
        different dtypes, e.g., int32 -> float32) have their clones eliminated
        when passed to mutating custom ops. The fix_auto_functionalized_dtype_views
        pass detects dtype views that alias graph inputs and removes unnecessary
        clone operations.

        Also tests negative cases where clones SHOULD be retained.
        """
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::mutate_inplace",
                "(Tensor(a!) x, Tensor y) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::mutate_inplace", "cpu", lib=lib)
            @torch._dynamo.disable
            def mutate_inplace_cpu(x, y):
                x.copy_(y)

            @torch.library.impl("mylib::mutate_inplace", "Meta", lib=lib)
            def mutate_inplace_meta(x, y):
                pass

            # Dtype view (int32 -> float32) - clone should be eliminated
            def f_dtype_view(cache_int32, data_float32):
                cache_float = cache_int32.view(torch.float32)
                torch.ops.mylib.mutate_inplace(cache_float, data_float32)
                return cache_int32

            cache = torch.zeros((10, 10), dtype=torch.int32)
            data = torch.randn((10, 10), dtype=torch.float32)
            result_eager = f_dtype_view(cache.clone(), data)

            for mode_name, mode_ctx in [
                ("inference", torch.inference_mode()),
                ("no_grad", torch.no_grad()),
            ]:
                with mode_ctx:
                    counters.clear()
                    torch._dynamo.reset()
                    compiled_f = torch.compile(
                        f_dtype_view, fullgraph=True, backend="inductor"
                    )
                    result = compiled_f(cache.clone(), data)

                    self.assertEqual(result.dtype, torch.int32)
                    self.assertEqual(result, result_eager)
                    self.assertEqual(
                        counters["inductor"]["fix_auto_functionalized_dtype_views"], 1
                    )

            # Negative cases (slice view, intermediate tensor) - clone should be retained
            def f_slice(cache, data):
                cache_slice = cache[0:10, 0:10]
                torch.ops.mylib.mutate_inplace(cache_slice, data)
                return cache

            def f_intermediate(data):
                temp = torch.zeros((10, 10), dtype=torch.int32)
                torch.ops.mylib.mutate_inplace(temp.view(torch.float32), data)
                return temp

            for func, args in [
                (
                    f_slice,
                    [
                        torch.zeros((20, 20), dtype=torch.float32),
                        torch.randn((10, 10), dtype=torch.float32),
                    ],
                ),
                (f_intermediate, [torch.randn((10, 10), dtype=torch.float32)]),
            ]:
                result_eager_neg = func(*[a.clone() for a in args])

                with torch.no_grad():
                    counters.clear()
                    torch._dynamo.reset()
                    compiled_f = torch.compile(func, fullgraph=True, backend="inductor")
                    result = compiled_f(*[a.clone() for a in args])

                    self.assertEqual(result, result_eager_neg)
                    self.assertEqual(
                        counters["inductor"]["fix_auto_functionalized_dtype_views"], 0
                    )

            # Two mutations through the same dtype view: counter is 1, not 2 -
            # reinplace_inplaceable_ops already inplaces the second.
            def f_twice(cache, data1, data2):
                v = cache.view(torch.float32)
                torch.ops.mylib.mutate_inplace(v, data1)
                torch.ops.mylib.mutate_inplace(v, data2)
                return cache

            data1 = torch.randn((10, 10), dtype=torch.float32)
            data2 = torch.randn((10, 10), dtype=torch.float32)
            cache_eager = cache.clone()
            f_twice(cache_eager, data1, data2)

            with torch.no_grad():
                counters.clear()
                torch._dynamo.reset()
                compiled_f = torch.compile(f_twice, fullgraph=True, backend="inductor")
                cache_compiled = cache.clone()
                compiled_f(cache_compiled, data1, data2)

                self.assertEqual(cache_compiled, cache_eager)
                self.assertEqual(
                    counters["inductor"]["fix_auto_functionalized_dtype_views"], 1
                )

    @torch._inductor.config.patch(enable_auto_functionalized_v2=True)
    def test_dtype_view_clone_elimination_shared_storage(self):
        """
        Two graph-input placeholders that share the same underlying storage
        (e.g. per-layer slices of a single KV pool) collide in the pass's
        ``storage_to_input`` map, which is keyed by
        ``untyped_storage()._cdata`` alone. Verify the pass still produces
        correct output for both mutations.

        This case is realistic for paged-KV deployments where each layer holds
        a slice of one big buffer with positive ``storage_offset``.
        """
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::mutate_inplace_pool",
                "(Tensor(a!) x, Tensor y) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::mutate_inplace_pool", "cpu", lib=lib)
            @torch._dynamo.disable
            def mutate_inplace_pool_cpu(x, y):
                x.copy_(y)

            @torch.library.impl("mylib::mutate_inplace_pool", "Meta", lib=lib)
            def mutate_inplace_pool_meta(x, y):
                pass

            def f_pool_slices(slice_a, slice_b, data_a, data_b):
                a_view = slice_a.view(torch.float32)
                b_view = slice_b.view(torch.float32)
                torch.ops.mylib.mutate_inplace_pool(a_view, data_a)
                torch.ops.mylib.mutate_inplace_pool(b_view, data_b)
                return slice_a, slice_b

            pool = torch.zeros((4, 10, 10), dtype=torch.int32)
            # Two slices share `_cdata` but have different `storage_offset`.
            slice_a = pool[1]
            slice_b = pool[3]
            self.assertEqual(
                slice_a.untyped_storage()._cdata,
                slice_b.untyped_storage()._cdata,
            )
            self.assertNotEqual(slice_a.storage_offset(), slice_b.storage_offset())

            data_a = torch.randn((10, 10), dtype=torch.float32)
            data_b = torch.randn((10, 10), dtype=torch.float32)

            pool_eager = pool.clone()
            sa_eager, sb_eager = f_pool_slices(
                pool_eager[1], pool_eager[3], data_a, data_b
            )

            with torch.no_grad():
                counters.clear()
                torch._dynamo.reset()
                compiled_f = torch.compile(
                    f_pool_slices, fullgraph=True, backend="inductor"
                )
                pool_compiled = pool.clone()
                sa, sb = compiled_f(pool_compiled[1], pool_compiled[3], data_a, data_b)

                # Outputs must match eager for BOTH slices, regardless of the
                # pass's storage-collision identification order.
                self.assertEqual(sa.dtype, torch.int32)
                self.assertEqual(sb.dtype, torch.int32)
                self.assertEqual(sa, sa_eager)
                self.assertEqual(sb, sb_eager)
                # The underlying pool must reflect both mutations through the
                # shared storage.
                self.assertEqual(pool_compiled, pool_eager)
                # The pass must fire once per auto_functionalized_v2 node;
                # we have two dtype-view mutations so the counter is 2.
                self.assertEqual(
                    counters["inductor"]["fix_auto_functionalized_dtype_views"], 2
                )

    def test_reinplace_mutated_empty_no_self_edge(self):
        # Regression test for a Scheduler.compute_ancestors KeyError.
        #
        # An auto_functionalized custom op that mutates freshly allocated (empty)
        # output buffers whose results are then reshaped forces the reinplace pass
        # to fold a realization back onto the op's own buffer. That produced a
        # scheduler node with a non-weak MemoryDep on the buffer it also writes --
        # a self-edge -- and compute_ancestors looked up name_to_ancestors[op]
        # before it was populated, raising KeyError: 'opN'.
        #
        # Mirrors flash_attn._flash_attn_backward writing dq/dk/dv into empties.
        D = 512
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::attn_bwd",
                "(Tensor dout, Tensor(a!) dq, Tensor(b!) dk, Tensor(c!) dv) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::attn_bwd", "cpu", lib=lib)
            def attn_bwd(dout, dq, dk, dv):
                src = dout.reshape(dq.shape)
                dq.copy_(src)
                dk.copy_(src)
                dv.copy_(src)
                return torch.zeros_like(dq)  # workspace, discarded by caller

            @torch.library.register_fake("mylib::attn_bwd", lib=lib)
            def _(dout, dq, dk, dv):
                return torch.empty_like(dq)

            def f(dout, weight, num_heads):
                head_dim = D // num_heads
                seqlen = dout.shape[0]
                dq = torch.empty(1, seqlen, num_heads, head_dim, dtype=dout.dtype)
                dk = torch.empty(1, seqlen, num_heads, head_dim, dtype=dout.dtype)
                dv = torch.empty(1, seqlen, num_heads, head_dim, dtype=dout.dtype)
                torch.ops.mylib.attn_bwd(dout, dq, dk, dv)  # return discarded
                a = dq.reshape(1, seqlen, D)
                b = dk.reshape(1, seqlen, D)
                c = dv.reshape(1, seqlen, D)
                return (a + b + c) @ weight

            dout = torch.randn(128, D)
            weight = torch.randn(D, D)
            torch._dynamo.mark_dynamic(dout, 0)

            expected = f(dout, weight, 8)
            got = torch.compile(f, fullgraph=True, dynamic=True)(dout, weight, 8)
            self.assertEqual(got, expected)


class AutoFunctionalizeViewTests(torch._inductor.test_case.TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @inductor_config.patch(enable_auto_functionalized_v2=False)
    @parametrize("metadata", [False, True])
    @parametrize("dynamic", [False, True])
    def test_pattern_replacement_joint_provenance(self, device, metadata, dynamic):
        from torch._higher_order_ops.auto_functionalize import auto_functionalized
        from torch._inductor.pattern_matcher import (
            joint_fwd_bwd,
            PatternMatcherPass,
            register_replacement,
        )

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            for name in ("partial", "replacement"):
                lib.define(f"{name}(Tensor(a!) out) -> ()")

                def partial(out):
                    out[:-2].add_(100)
                    out[-1:].add_(200)

                lib.impl(name, partial, "CompositeExplicitAutograd")
                torch.library.register_fake("mylib::" + name, lambda out: None, lib=lib)

            original = torch.ops.mylib.partial.default
            replacement_op = torch.ops.mylib.replacement.default

            def pattern(out, weight):
                return auto_functionalized(original, out=out)[1] * weight

            def replacement(out, weight):
                view = out.view(out.shape[0], 2, 4)
                result = auto_functionalized(replacement_op, out=view)[1]
                return result.view(out.shape) * weight

            patterns = PatternMatcherPass()
            register_replacement(
                pattern,
                replacement,
                [
                    torch.empty(6, 8, device=device),
                    torch.empty(6, 8, device=device, requires_grad=True),
                ],
                joint_fwd_bwd,
                patterns,
            )
            counts = []

            def custom_pass(graph):
                before = next(
                    iter(
                        graph.find_nodes(op="call_function", target=auto_functionalized)
                    )
                )
                self.assertEqual(len(before.args), 2 if metadata else 1)
                base = before.args[1]["out"][0] if metadata else None
                counts.append(patterns.apply(graph))
                after = list(
                    graph.find_nodes(op="call_function", target=auto_functionalized)
                )
                self.assertEqual(len(after), 1)
                self.assertIs(after[0].args[0], replacement_op)
                self.assertEqual(len(after[0].args), 2 if metadata else 1)
                if metadata:
                    entry = after[0].args[1]["out"]
                    self.assertIs(entry[0], base)
                    size, stride, offset = torch.fx.map_arg(
                        entry[1:], lambda node: node.meta["val"]
                    )
                    value = after[0].kwargs["out"].meta["val"]
                    self.assertEqual(tuple(size), tuple(value.shape))
                    self.assertEqual(tuple(stride), (16, 8, 2))
                    self.assertEqual(offset, 32)
                graph.lint()

            def fn(buf, weight):
                base = buf + 10
                if metadata:
                    view = base[2:, ::2]
                    result = auto_functionalized(
                        original,
                        {"out": (base, view.size(), view.stride(), 32)},
                        out=view,
                    )[1]
                else:
                    result = auto_functionalized(original, out=base)[1]
                return result * weight

            with inductor_config.patch(joint_custom_post_pass=custom_pass):
                compiled = torch.compile(fn, fullgraph=True, dynamic=dynamic)
                for rows in [6, 8] if dynamic else [6]:
                    shape = (rows + 2, 16) if metadata else (rows, 8)
                    for offset in (1, 100):
                        buf = (
                            torch.arange(
                                shape[0] * shape[1], device=device, dtype=torch.float32
                            ).reshape(shape)
                            + offset
                        )
                        saved = buf.clone()
                        weight = torch.randn(rows, 8, device=device, requires_grad=True)
                        expected = fn(buf, weight)
                        actual = compiled(buf, weight)
                        tangent = (
                            torch.arange(
                                rows * 8, device=device, dtype=torch.float32
                            ).reshape(rows, 8)
                            % 3
                            + 1
                        )
                        expected_grad = torch.autograd.grad(expected, weight, tangent)[
                            0
                        ]
                        actual_grad = torch.autograd.grad(actual, weight, tangent)[0]
                        self.assertEqual(actual, expected, atol=0, rtol=0)
                        self.assertEqual(actual_grad, expected_grad, atol=0, rtol=0)
                        self.assertEqual(buf, saved, atol=0, rtol=0)
            self.assertTrue(counts)
            self.assertEqual(counts, [1] * len(counts))

    @inductor_config.patch(enable_auto_functionalized_v2=False)
    @parametrize(
        "kind",
        [
            "view",
            "strided",
            "dynamic",
            "plain",
            "explicit",
            "unmatched",
            "derived",
            "layout",
            "layout_dynamic",
            "manual",
        ],
    )
    def test_pattern_replacement_preserves_view_metadata(self, device, kind):
        import operator

        from torch._higher_order_ops.auto_functionalize import auto_functionalized
        from torch._inductor.pattern_matcher import (
            CallFunction,
            FailedMatch,
            fwd_only,
            KeywordArg,
            PatternMatcherPass,
            register_replacement,
        )

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("partial(Tensor(a!) out, int amount) -> ()")
            lib.define("replacement(Tensor(a!) destination, int amount) -> ()")

            def partial(out, amount):
                out[:-2].add_(amount)
                out[-1:].add_(2 * amount)

            def partial_fake(out, amount):
                return None

            for name in ("partial", "replacement"):
                lib.impl(name, partial, "CompositeExplicitAutograd")
                torch.library.register_fake("mylib::" + name, partial_fake, lib=lib)

            original = torch.ops.mylib.partial.default
            replacement_op = torch.ops.mylib.replacement.default

            def pattern(out):
                return auto_functionalized(original, out=out, amount=100)[1]

            def replacement(out):
                if kind == "derived":
                    out = out.flip((0,)).flip((0,))
                result = auto_functionalized(
                    replacement_op, destination=out, amount=100
                )[1]
                if kind in ("layout", "layout_dynamic", "manual"):
                    return result.as_strided(
                        result.size(), result.stride(), result.storage_offset()
                    )
                return result

            patterns = PatternMatcherPass()
            register_replacement(
                pattern,
                replacement,
                [torch.empty(6, 8, device=device)],
                fwd_only,
                patterns,
            )
            counts = []

            def custom_pass(graph):
                before = list(
                    graph.find_nodes(op="call_function", target=auto_functionalized)
                )
                self.assertEqual(len(before), 1)
                metadata = before[0].args[1] if len(before[0].args) == 2 else None
                output_layout = (
                    before[0].meta["val"][1].stride(),
                    before[0].meta["val"][1].storage_offset(),
                )
                if kind in (
                    "view",
                    "strided",
                    "dynamic",
                    "unmatched",
                    "derived",
                    "layout",
                    "layout_dynamic",
                    "manual",
                ):
                    self.assertIsNotNone(metadata)
                else:
                    self.assertIsNone(metadata)
                if kind == "manual":
                    legacy_pattern = CallFunction(
                        operator.getitem,
                        CallFunction(
                            auto_functionalized,
                            original,
                            out=KeywordArg("out"),
                            amount=100,
                        ),
                        1,
                    )
                    output = next(n for n in before[0].users if n.args[1] == 1)
                    match = legacy_pattern.match(output)
                    self.assertTrue(match)
                    with self.assertRaisesRegex(FailedMatch, "register_replacement"):
                        match.replace_by_example(replacement, [before[0].kwargs["out"]])
                    counts.append(0)
                else:
                    counts.append(patterns.apply(graph))
                after = list(
                    graph.find_nodes(op="call_function", target=auto_functionalized)
                )
                self.assertEqual(len(after), 1)
                if kind != "derived":
                    self.assertEqual(
                        (
                            after[0].meta["val"][1].stride(),
                            after[0].meta["val"][1].storage_offset(),
                        ),
                        output_layout,
                    )
                if kind in ("unmatched", "manual"):
                    self.assertIs(after[0].args[0], original)
                    self.assertEqual(after[0].args[1], metadata)
                else:
                    self.assertIs(after[0].args[0], replacement_op)
                    if metadata is None or kind == "derived":
                        self.assertEqual(len(after[0].args), 1)
                    else:
                        self.assertEqual(
                            after[0].args[1], {"destination": metadata["out"]}
                        )
                graph.lint()

            def f(x):
                base = x + 10
                if kind == "explicit":
                    return pattern(base)
                view = (
                    base
                    if kind == "plain"
                    else base[2:, :: 2 if kind == "strided" else 1]
                )
                original(view, 101 if kind == "unmatched" else 100)
                return base, view

            with inductor_config.patch(post_grad_custom_post_pass=custom_pass):
                compiled = torch.compile(
                    f,
                    fullgraph=True,
                    dynamic=kind in ("dynamic", "layout_dynamic", "manual"),
                )
                for rows in (
                    [8, 10] if kind in ("dynamic", "layout_dynamic", "manual") else [8]
                ):
                    for offset in (1, 100):
                        x = torch.arange(
                            offset,
                            offset + rows * 8,
                            device=device,
                            dtype=torch.float32,
                        ).reshape(rows, 8)
                        self.assertEqual(compiled(x), f(x), atol=0, rtol=0)
            self.assertTrue(counts)
            self.assertEqual(
                counts,
                [int(kind not in ("unmatched", "manual"))] * len(counts),
            )

    @inductor_config.patch(enable_auto_functionalized_v2=False)
    @parametrize("kind", ["split", "rank_view"])
    @parametrize("dynamic", [False, True])
    @parametrize("inference", [False, True])
    def test_pattern_replacement_created_views(self, device, kind, dynamic, inference):
        from torch._higher_order_ops.auto_functionalize import auto_functionalized
        from torch._inductor.pattern_matcher import (
            fwd_only,
            PatternMatcherPass,
            register_replacement,
        )

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("partial(Tensor(a!) out) -> ()")
            lib.define("replacement(Tensor(a!)[] destinations) -> ()")

            def partial(out):
                out[:-2].add_(100)
                out[-1:].add_(200)

            lib.impl("partial", partial, "CompositeExplicitAutograd")

            def replace_impl(destinations):
                for out in destinations:
                    partial(out)

            lib.impl("replacement", replace_impl, "CompositeExplicitAutograd")
            torch.library.register_fake("mylib::partial", lambda out: None, lib=lib)
            torch.library.register_fake(
                "mylib::replacement", lambda destinations: None, lib=lib
            )
            original = torch.ops.mylib.partial.default
            replacement_op = torch.ops.mylib.replacement.default

            def pattern(out):
                return auto_functionalized(original, out=out)[1]

            def replacement(out):
                destinations = (
                    list(out.split([3, 5], dim=1))
                    if kind == "split"
                    else [out.view(out.shape[0], 2, 4)]
                )
                result = auto_functionalized(replacement_op, destinations=destinations)[
                    1
                ]
                return (
                    torch.cat(result, dim=1)
                    if kind == "split"
                    else result[0].view(out.shape)
                )

            patterns = PatternMatcherPass()
            register_replacement(
                pattern,
                replacement,
                [torch.empty(6, 8, device=device)],
                fwd_only,
                patterns,
            )
            counts = []

            def custom_pass(graph):
                before = next(
                    iter(
                        graph.find_nodes(op="call_function", target=auto_functionalized)
                    )
                )
                original_metadata = before.args[1]["out"]
                counts.append(patterns.apply(graph))
                hops = list(
                    graph.find_nodes(op="call_function", target=auto_functionalized)
                )
                self.assertEqual(len(hops), 1)
                hop = hops[0]
                self.assertIs(hop.args[0], replacement_op)
                self.assertEqual(len(hop.args), 2)
                entries = hop.args[1]["destinations"]
                self.assertEqual(len(entries), 2 if kind == "split" else 1)
                if kind == "split":
                    self.assertTrue(
                        list(
                            graph.find_nodes(
                                op="call_function",
                                target=torch.ops.aten.split_with_sizes.default,
                            )
                        )
                    )
                for destination, entry in zip(hop.kwargs["destinations"], entries):
                    self.assertIs(entry[0], original_metadata[0])
                    value = destination.meta["val"]
                    size, stride, offset = torch.fx.map_arg(
                        entry[1:], lambda node: node.meta["val"]
                    )
                    self.assertEqual(tuple(size), tuple(value.shape))
                    self.assertEqual(tuple(stride), value.stride())
                    self.assertEqual(offset, value.storage_offset())
                graph.lint()

            def f(x):
                base = x + 10
                view = base[2:, ::2]
                original(view)
                return base, view

            with (
                torch.inference_mode(inference),
                inductor_config.patch(post_grad_custom_post_pass=custom_pass),
            ):
                compiled = torch.compile(f, fullgraph=True, dynamic=dynamic)
                for rows in [8, 10] if dynamic else [8]:
                    x = torch.arange(rows * 16, device=device, dtype=torch.float32)
                    x = x.reshape(rows, 16)
                    actual = compiled(x)
                    expected = f(x)
                    self.assertEqual(actual, expected, atol=0, rtol=0)
                    self.assertEqual(
                        actual[0].untyped_storage()._cdata,
                        actual[1].untyped_storage()._cdata,
                    )
                    self.assertEqual(actual[1].stride(), expected[1].stride())
                    self.assertEqual(actual[1].storage_offset(), 32)
                    self.assertEqual(
                        x,
                        torch.arange(
                            rows * 16, device=device, dtype=torch.float32
                        ).reshape(rows, 16),
                    )
            self.assertTrue(counts)
            self.assertEqual(counts, [1] * len(counts))

    @parametrize("kind", ["fresh", "mixed"])
    def test_pattern_replacement_fresh_buffers(self, device, kind):
        from torch._higher_order_ops.auto_functionalize import auto_functionalized
        from torch._inductor.pattern_matcher import (
            fwd_only,
            PatternMatcherPass,
            register_replacement,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("partial(Tensor(a!) out) -> ()")
            lib.define("replacement(Tensor(a!)[] destinations) -> ()")

            def partial(out):
                out[:-2].add_(100)
                out[-1:].add_(200)

            def replace_impl(destinations):
                for out in destinations:
                    partial(out)

            lib.impl("partial", partial, "CompositeExplicitAutograd")
            lib.impl("replacement", replace_impl, "CompositeExplicitAutograd")
            torch.library.register_fake("mylib::partial", lambda out: None, lib=lib)
            torch.library.register_fake(
                "mylib::replacement", lambda destinations: None, lib=lib
            )
            original = torch.ops.mylib.partial.default
            replacement_op = torch.ops.mylib.replacement.default

            def pattern(out):
                return auto_functionalized(original, out=out)[1]

            def replacement(out):
                fresh = out.clone()
                destinations = (
                    [fresh] if kind == "fresh" else [out.view(6, 2, 4), fresh]
                )
                result = auto_functionalized(replacement_op, destinations=destinations)[
                    1
                ]
                if kind == "fresh":
                    return result[0]
                # Consume both buffers so the fresh output stays live.
                return (result[0].view(6, 8) + result[1]) / 2

            def f(base):
                return auto_functionalized(
                    original,
                    {"out": (base, (6, 8), (16, 2), 32)},
                    out=base[2:, ::2],
                )[1]

            x = torch.arange(128, device=device, dtype=torch.float32).reshape(8, 16)
            patterns = PatternMatcherPass()
            register_replacement(pattern, replacement, [x[2:, ::2]], fwd_only, patterns)
            gm = make_fx(f, tracing_mode="fake")(x)
            gm.graph.eliminate_dead_code()
            base = next(n for n in gm.graph.nodes if n.op == "placeholder")
            self.assertEqual(patterns.apply(gm.graph), 1)
            hop = next(
                iter(
                    gm.graph.find_nodes(op="call_function", target=auto_functionalized)
                )
            )
            self.assertIs(hop.args[0], replacement_op)
            if kind == "fresh":
                self.assertEqual(len(hop.args), 1)
            else:
                self.assertEqual(len(hop.args), 2)
                self.assertEqual(
                    hop.args[1],
                    {"destinations": [(base, (6, 2, 4), (16, 8, 2), 32), None]},
                )
            gm.graph.lint()
            gm.recompile()
            for offset in (0, 100):
                inp = x + offset
                self.assertEqual(gm(inp), f(inp), atol=0, rtol=0)
                self.assertEqual(inp, x + offset)

    @inductor_config.patch(enable_auto_functionalized_v2=False)
    @parametrize("metadata", [False, True])
    @parametrize("replacement_hop", [False, True])
    def test_pattern_replacement_with_internal_base(
        self, device, metadata, replacement_hop
    ):
        from torch._higher_order_ops.auto_functionalize import auto_functionalized
        from torch._inductor.pattern_matcher import (
            fwd_only,
            PatternMatcherPass,
            register_replacement,
        )

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("partial(Tensor(a!) out) -> ()")
            lib.define("replacement(Tensor(a!) out) -> ()")

            @torch.library.impl(lib, "partial", "CompositeExplicitAutograd")
            def partial(out):
                out[:-2].add_(100)
                out[-1:].add_(200)

            @torch.library.register_fake("mylib::partial", lib=lib)
            def partial_fake(out):
                return None

            lib.impl("replacement", partial, "CompositeExplicitAutograd")
            torch.library.register_fake("mylib::replacement", partial_fake, lib=lib)

            def pattern(x):
                base = x + 10
                return auto_functionalized(
                    torch.ops.mylib.partial.default, out=base[2:]
                )[1]

            def replacement(x):
                view = (x + 10)[2:]
                if replacement_hop:
                    return auto_functionalized(
                        torch.ops.mylib.replacement.default, out=view
                    )[1]
                return torch.cat((view[:-2] + 100, view[-2:-1], view[-1:] + 200))

            patterns = PatternMatcherPass()
            register_replacement(
                pattern,
                replacement,
                [torch.empty(8, device=device)],
                fwd_only,
                patterns,
            )
            counts = []

            def custom_pass(graph):
                counts.append(patterns.apply(graph))
                hops = list(
                    graph.find_nodes(op="call_function", target=auto_functionalized)
                )
                self.assertEqual(len(hops), int(replacement_hop))
                if replacement_hop:
                    hop = hops[0]
                    self.assertIs(hop.args[0], torch.ops.mylib.replacement.default)
                    self.assertEqual(len(hop.args), 2 if metadata else 1)
                    if metadata:
                        base, size, stride, offset = hop.args[1]["out"]
                        self.assertEqual(base.target, torch.ops.aten.add.Tensor)
                        self.assertEqual(
                            (tuple(size), tuple(stride), offset), ((6,), (1,), 2)
                        )
                graph.lint()

            def f(x):
                if not metadata:
                    return pattern(x)
                base = x + 10
                return auto_functionalized(
                    torch.ops.mylib.partial.default,
                    {"out": (base, (6,), (1,), 2)},
                    out=base[2:],
                )[1]

            with inductor_config.patch(post_grad_custom_post_pass=custom_pass):
                compiled = torch.compile(f, fullgraph=True)
                for offset in (1, 100):
                    x = torch.arange(
                        offset, offset + 8, device=device, dtype=torch.float32
                    )
                    self.assertEqual(compiled(x), f(x), atol=0, rtol=0)
            self.assertEqual(counts, [1])

    def test_pattern_replacement_with_explicit_view_metadata(self, device):
        from torch._higher_order_ops.auto_functionalize import auto_functionalized
        from torch._inductor.pattern_matcher import (
            fwd_only,
            PatternMatcherPass,
            register_replacement,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            for name in ("partial", "replacement"):
                lib.define(f"{name}(Tensor(a!) out) -> ()")

                def partial(out):
                    out[:2].add_(100)

                def partial_fake(out):
                    return None

                lib.impl(name, partial, "CompositeExplicitAutograd")
                torch.library.register_fake("mylib::" + name, partial_fake, lib=lib)

            original = torch.ops.mylib.partial.default
            replacement_op = torch.ops.mylib.replacement.default

            def pattern(base):
                return auto_functionalized(
                    original, {"out": (base, (6,), (1,), 2)}, out=base[2:]
                )[1]

            def replacement(base):
                return auto_functionalized(
                    replacement_op, {"out": (base, (6,), (1,), 2)}, out=base[2:]
                )[1]

            x = torch.arange(8, device=device, dtype=torch.float32)
            patterns = PatternMatcherPass()
            register_replacement(pattern, replacement, [x], fwd_only, patterns)
            gm = make_fx(pattern)(x)
            gm.graph.eliminate_dead_code()
            gm.recompile()
            before = next(
                iter(
                    gm.graph.find_nodes(op="call_function", target=auto_functionalized)
                )
            )
            metadata_base = before.args[1]["out"][0]
            self.assertEqual(patterns.apply(gm.graph), 1)
            gm.graph.lint()
            gm.recompile()
            hops = list(
                gm.graph.find_nodes(op="call_function", target=auto_functionalized)
            )
            self.assertEqual(len(hops), 1)
            self.assertIs(hops[0].args[0], replacement_op)
            self.assertEqual(hops[0].args[1], {"out": (metadata_base, (6,), (1,), 2)})
            for offset in (0, 20):
                inp = x + offset
                self.assertEqual(gm(inp), pattern(inp), atol=0, rtol=0)
                self.assertEqual(inp, x + offset)

    @parametrize(
        "kind",
        [
            "inplace",
            "out",
            "mutable_alias",
            "ambiguous_alias",
            "hop_getitem",
            "unsafe_view",
            "reshape_copy",
            "contiguous_copy",
            "reshape_alias",
            "contiguous_alias",
            "view_alias",
        ],
    )
    def test_pattern_replacement_checks_provenance(self, device, kind):
        from torch._higher_order_ops.auto_functionalize import auto_functionalized
        from torch._inductor.pattern_matcher import (
            fwd_only,
            PatternMatcherPass,
            register_replacement,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            for name in ("partial", "replacement", "identity"):
                lib.define(f"{name}(Tensor(a!) out) -> ()")

                def mutate(out):
                    out[:-2].add_(100)
                    out[-1:].add_(200)

                lib.impl(
                    name,
                    (lambda out: None) if name == "identity" else mutate,
                    "CompositeExplicitAutograd",
                )
                torch.library.register_fake("mylib::" + name, lambda out: None, lib=lib)

            lib.define("mutable_alias(Tensor(a!) out) -> Tensor(a!)")
            lib.define(
                "ambiguous_alias(Tensor(a) left, Tensor(b) right) -> Tensor(a|b)"
            )
            lib.impl(
                "mutable_alias", lambda out: out.add_(0), "CompositeExplicitAutograd"
            )
            lib.impl(
                "ambiguous_alias", lambda left, right: left, "CompositeExplicitAutograd"
            )
            torch.library.register_fake(
                "mylib::mutable_alias", lambda out: out, lib=lib
            )
            torch.library.register_fake(
                "mylib::ambiguous_alias", lambda left, right: left, lib=lib
            )
            original = torch.ops.mylib.partial.default
            replacement_op = torch.ops.mylib.replacement.default

            def pattern(out):
                return auto_functionalized(original, out=out)[1]

            def replacement(out):
                if kind == "view_alias":
                    derived = out.view(6, 2, 4).view(6, 8)
                elif kind == "inplace":
                    derived = out.add_(0)
                elif kind == "out":
                    derived = torch.ops.aten.add.out(out, 0, out=out)
                elif kind == "mutable_alias":
                    derived = torch.ops.mylib.mutable_alias.default(out)
                elif kind == "ambiguous_alias":
                    derived = torch.ops.mylib.ambiguous_alias.default(out, out.clone())
                elif kind == "hop_getitem":
                    derived = auto_functionalized(
                        torch.ops.mylib.identity.default, out=out
                    )[1]
                elif kind == "unsafe_view":
                    derived = torch.ops.aten._unsafe_view.default(out, out.shape)
                elif kind == "reshape_alias":
                    derived = out.reshape(6, 2, 4).view(6, 8)
                elif kind == "reshape_copy":
                    derived = out.t().reshape(-1).view(8, 6).t()
                else:
                    derived = torch.ops.aten.contiguous.default(out)
                return auto_functionalized(replacement_op, out=derived)[1]

            aliases = kind in ("reshape_alias", "contiguous_alias")
            accepted = aliases or kind == "view_alias"
            columns = 8 if aliases else 16
            step = 1 if aliases else 2
            geometry = ((6, 8), (columns, step), 2 * columns)

            def fn(base):
                return auto_functionalized(
                    original, {"out": (base, *geometry)}, out=base[2:, ::step]
                )[1]

            x = torch.arange(8 * columns, device=device, dtype=torch.float32).reshape(
                8, columns
            )
            patterns = PatternMatcherPass()

            # Keep conditional-copy schemas, unsafe views, and mutable operators.
            def trace_fn(fn, args, **kwargs):
                if kind in (
                    "reshape_copy",
                    "contiguous_copy",
                    "reshape_alias",
                    "contiguous_alias",
                ):
                    traced = make_fx(fn, pre_dispatch=True, tracing_mode="fake")(*args)
                    if fn is replacement:
                        target = (
                            torch.ops.aten.reshape.default
                            if kind.startswith("reshape_")
                            else torch.ops.aten.contiguous.default
                        )
                        self.assertTrue(
                            list(
                                traced.graph.find_nodes(
                                    op="call_function", target=target
                                )
                            )
                        )
                else:
                    kwargs.update(get_decomp_fn=dict, run_functional_passes=False)
                    traced = fwd_only(fn, args, **kwargs)
                # Match the target graph's user counts, including unused HOP returns.
                traced.graph.eliminate_dead_code()
                traced.recompile()
                return traced

            register_replacement(
                pattern, replacement, [x[2:, ::step]], trace_fn, patterns
            )
            gm = make_fx(fn, tracing_mode="fake")(x)
            gm.graph.eliminate_dead_code()
            gm.recompile()
            original_hop = next(
                iter(
                    gm.graph.find_nodes(op="call_function", target=auto_functionalized)
                )
            )
            original_value = original_hop.kwargs["out"].meta["val"]
            self.assertEqual(
                (
                    tuple(original_value.shape),
                    original_value.stride(),
                    original_value.storage_offset(),
                ),
                geometry,
            )
            before = gm.code
            self.assertEqual(patterns.apply(gm.graph), int(accepted))
            gm.graph.lint()
            gm.recompile()
            if accepted:
                hop = next(
                    iter(
                        gm.graph.find_nodes(
                            op="call_function", target=auto_functionalized
                        )
                    )
                )
                self.assertIs(hop.args[0], replacement_op)
                self.assertEqual(len(hop.args), 2)
                base = next(n for n in gm.graph.nodes if n.op == "placeholder")
                self.assertEqual(hop.args[1], {"out": (base, *geometry)})
            else:
                self.assertEqual(gm.code, before)
            for offset in (0, 100):
                inp = x + offset
                self.assertEqual(gm(inp), fn(inp), atol=0, rtol=0)
                self.assertEqual(inp, x + offset)

    @inductor_config.patch(enable_auto_functionalized_v2=False)
    @parametrize("kind", ["legacy", "explicit_metadata", "varargs"])
    @parametrize("with_metadata", [False, True])
    def test_graph_handler_view_metadata(self, device, kind, with_metadata):
        from torch._higher_order_ops.auto_functionalize import auto_functionalized
        from torch._inductor.pattern_matcher import (
            CallFunction,
            CallFunctionVarArgs,
            KeywordArg,
            PatternMatcherPass,
            register_graph_pattern,
        )
        from torch._subclasses.functional_tensor import (
            dispatch_functionalize,
            FunctionalTensorMode,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:

            def partial(out):
                out[:2].add_(100)

            for name in ("partial", "replacement"):
                lib.define(f"{name}(Tensor(a!) out) -> ()")
                lib.impl(name, partial, "CompositeExplicitAutograd")
                torch.library.register_fake(f"mylib::{name}", lambda out: None, lib=lib)
            op = torch.ops.mylib.partial.default
            replacement_op = torch.ops.mylib.replacement.default

            def fn(x):
                base = x + 10
                out = base[1:, ::2] if with_metadata else base
                op(out)
                return base, out, base[:, 1::2]

            x = torch.arange(48, device=device, dtype=torch.float32).reshape(6, 8)
            gm = make_fx(
                dispatch_functionalize(fn, mode=FunctionalTensorMode()),
                tracing_mode="fake",
            )(x)
            original_hop = next(
                iter(
                    gm.graph.find_nodes(op="call_function", target=auto_functionalized)
                )
            )
            original_args = original_hop.args
            self.assertEqual(len(original_args), 2 if with_metadata else 1)
            if kind == "varargs":
                pattern = CallFunctionVarArgs(auto_functionalized)
            else:
                pattern_args = (
                    (op, KeywordArg("view_metadata"))
                    if kind == "explicit_metadata"
                    else (op,)
                )
                pattern = CallFunction(
                    auto_functionalized, *pattern_args, out=KeywordArg("out")
                )
            patterns = PatternMatcherPass()
            calls = []

            @register_graph_pattern(pattern, pass_dict=patterns)
            def handler(match, *args, **kwargs):
                calls.append(True)
                if kind == "varargs":
                    self.assertIs(args[0], op)
                    hop_args = (replacement_op, *args[1:])
                else:
                    metadata = kwargs.pop("view_metadata", None)
                    hop_args = (
                        (replacement_op, metadata) if metadata else (replacement_op,)
                    )
                old = match.output_node()
                new = match.graph.call_function(auto_functionalized, hop_args, kwargs)
                new.meta = old.meta.copy()
                old.replace_all_uses_with(new)
                match.erase_nodes()

            expected_count = int(
                kind == "varargs"
                or (kind == "explicit_metadata" and with_metadata)
                or (kind == "legacy" and not with_metadata)
            )
            self.assertEqual(patterns.apply(gm.graph), expected_count)
            self.assertEqual(len(calls), expected_count)
            gm.graph.lint()
            gm.recompile()
            hop = next(
                iter(
                    gm.graph.find_nodes(op="call_function", target=auto_functionalized)
                )
            )
            self.assertIs(hop.args[0], replacement_op if expected_count else op)
            self.assertEqual(hop.args[1:], original_args[1:])
            for offset in (0, 100):
                inp = x + offset
                self.assertEqual(gm(inp), fn(inp), atol=0, rtol=0)
                self.assertEqual(inp, x + offset)

    @parametrize("kind", ["graph_handler", "external_metadata_user"])
    def test_pattern_replacement_rejects_unowned_view_metadata(self, device, kind):
        from torch._higher_order_ops.auto_functionalize import auto_functionalized
        from torch._inductor.pattern_matcher import (
            CallFunction,
            fwd_only,
            KeywordArg,
            PatternMatcherPass,
            register_graph_pattern,
            register_replacement,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("partial(Tensor(a!) out) -> ()")

            @torch.library.impl(lib, "partial", "CompositeExplicitAutograd")
            def partial(out):
                out[:2].add_(100)

            @torch.library.register_fake("mylib::partial", lib=lib)
            def partial_fake(out):
                return None

            op = torch.ops.mylib.partial.default

            def pattern(x):
                base = x + 10
                return auto_functionalized(op, out=base[2:])[1]

            def replacement(x):
                view = (x + 10)[2:]
                return torch.cat((view[:2] + 100, view[2:]))

            def fn(x, other):
                base = x + 10
                metadata = {"out": (base, (6,), (1,), 2)}
                result = auto_functionalized(op, metadata, out=base[2:])[1]
                if kind == "external_metadata_user":
                    return result, auto_functionalized(op, metadata, out=other)[1]
                return result

            x = torch.arange(8, device=device, dtype=torch.float32)
            other = (x + 10)[2:]
            patterns = PatternMatcherPass()
            if kind == "graph_handler":

                @register_graph_pattern(
                    CallFunction(auto_functionalized, op, out=KeywordArg("out")),
                    pass_dict=patterns,
                )
                def handler(match, out):
                    self.fail("A legacy graph handler must not discard view metadata")

            else:
                register_replacement(pattern, replacement, [x], fwd_only, patterns)
            gm = make_fx(fn)(x, other)
            gm.graph.eliminate_dead_code()
            gm.recompile()
            before = gm.code
            self.assertEqual(patterns.apply(gm.graph), 0)
            gm.graph.lint()
            gm.recompile()
            self.assertEqual(gm.code, before)
            self.assertEqual(gm(x, other), fn(x, other), atol=0, rtol=0)

    @inductor_config.patch(enable_auto_functionalized_v2=False)
    @parametrize("gap", [1, 2])
    @parametrize("inference", [False, True])
    @parametrize("dynamic", [False, True])
    def test_mutated_intermediate_view(self, device, gap, inference, dynamic):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("partial(Tensor(a!) out) -> ()")

            @torch.library.impl(lib, "partial", "CompositeExplicitAutograd")
            def partial(out):
                # Leave a late row untouched so an incorrect prefill is observable.
                out[:-2].add_(1000)
                out[-1:].add_(2000)

            @torch.library.register_fake("mylib::partial", lib=lib)
            def partial_fake(out):
                return None

            def f(x):
                base = x + 10
                view = base[2:, ::gap]
                torch.ops.mylib.partial(view)
                # The base must remain live to prevent eliminating the view clone.
                return base, view

            with torch.inference_mode(inference):
                compiled = torch.compile(f, fullgraph=True, dynamic=dynamic)
                for rows in [8, 10] if dynamic else [8]:
                    for offset in (1, 100):
                        x = torch.arange(
                            offset,
                            offset + rows * 8,
                            device=device,
                            dtype=torch.float32,
                        ).reshape(rows, 8)
                        self.assertEqual(compiled(x), f(x), atol=0, rtol=0)

    @inductor_config.patch(enable_auto_functionalized_v2=False)
    @parametrize("kind", ["slice", "set"])
    @parametrize("gap", [1, 2])
    @parametrize("inference", [False, True])
    def test_mutated_input_view(self, device, kind, gap, inference):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("partial(Tensor(a!) out) -> ()")

            @torch.library.impl(lib, "partial", "CompositeExplicitAutograd")
            def partial(out):
                out[:-2].add_(1000)
                out[-1:].add_(2000)

            @torch.library.register_fake("mylib::partial", lib=lib)
            def partial_fake(out):
                return None

            def f(view):
                before = view + 1
                torch.ops.mylib.partial(view)
                return before, view

            def make_input():
                base = torch.arange(1, 65, device=device, dtype=torch.float32)
                view = base.reshape(8, 8)[2:, ::gap]
                if kind == "set":
                    view = torch.empty(0, device=device).set_(
                        base.untyped_storage(),
                        view.storage_offset(),
                        view.size(),
                        view.stride(),
                    )
                return base, view

            with torch.inference_mode(inference):
                expected_base, expected_view = make_input()
                actual_base, actual_view = make_input()
                expected = f(expected_view)
                actual = torch.compile(f, fullgraph=True)(actual_view)
                self.assertEqual(actual, expected, atol=0, rtol=0)
                self.assertEqual(actual_base, expected_base, atol=0, rtol=0)

    @inductor_config.patch(enable_auto_functionalized_v2=False)
    @parametrize("kind", ["slice", "set"])
    @parametrize("inference", [False, True])
    @parametrize("dynamic", [False, True])
    @parametrize("fixed_extent", [False, True])
    def test_mutated_view_past_input_extent(
        self, device, kind, inference, dynamic, fixed_extent
    ):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("partial(Tensor(a!) out) -> ()")

            @torch.library.impl(lib, "partial", "CompositeExplicitAutograd")
            def partial(out):
                # Mutate within the input; the expanded view reads the rest.
                out[:2].add_(100)

            @torch.library.register_fake("mylib::partial", lib=lib)
            def partial_fake(out):
                return None

            def f(x):
                size = 8 if fixed_extent else 2 * x.size(0)
                view = x.as_strided((size,), (1,), 0)
                torch.ops.mylib.partial(view)
                return x, view

            def make_input(size, offset):
                owner = torch.arange(
                    offset, offset + 2 * size, device=device, dtype=torch.float32
                )
                if kind == "slice":
                    x = owner[:size]
                else:
                    x = torch.empty(0, device=device).set_(
                        owner.untyped_storage(), 0, (size,), (1,)
                    )
                return owner, x

            with torch.inference_mode(inference):
                compiled = torch.compile(f, fullgraph=True, dynamic=dynamic)
                for size in [4, 10, 6] if dynamic else [4]:
                    for offset in [0, 20]:
                        expected_owner, expected_x = make_input(size, offset)
                        actual_owner, actual_x = make_input(size, offset)
                        self.assertEqual(
                            compiled(actual_x), f(expected_x), atol=0, rtol=0
                        )
                        self.assertEqual(actual_owner, expected_owner, atol=0, rtol=0)

    @inductor_config.patch(enable_auto_functionalized_v2=False)
    @parametrize("inference", [False, True])
    def test_mutated_disjoint_views_shared_base(self, device, inference):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("partial(Tensor(a!) left, Tensor(b!) right) -> ()")

            @torch.library.impl(lib, "partial", "CompositeExplicitAutograd")
            def partial(left, right):
                left[:1].add_(100)
                right[-1:].add_(200)

            @torch.library.register_fake("mylib::partial", lib=lib)
            def partial_fake(left, right):
                return None

            def f(x):
                base = x + 10
                left, right = base[1:4], base[5:8]
                torch.ops.mylib.partial(left, right)
                return base, left, right

            with torch.inference_mode(inference):
                compiled = torch.compile(f, fullgraph=True)
                for offset in [0, 20]:
                    x = torch.arange(
                        offset, offset + 10, device=device, dtype=torch.float32
                    )
                    self.assertEqual(compiled(x), f(x), atol=0, rtol=0)

    @parametrize("fake", [False, True])
    def test_view_metadata_rejects_short_base(self, device, fake):
        from contextlib import nullcontext

        from torch._higher_order_ops.auto_functionalize import auto_functionalized
        from torch._subclasses.fake_tensor import FakeTensorMode

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("partial(Tensor(a!) out) -> ()")

            @torch.library.impl(lib, "partial", "CompositeExplicitAutograd")
            def partial(out):
                out[:2].add_(100)

            @torch.library.register_fake("mylib::partial", lib=lib)
            def partial_fake(out):
                return None

            with FakeTensorMode() if fake else nullcontext():
                view = torch.empty(8, device=device)
                metadata = {"out": (view[:4], view.size(), view.stride(), 0)}
                with self.assertRaisesRegex(
                    RuntimeError, "base spans 4 elements but x addresses 8"
                ):
                    auto_functionalized(
                        torch.ops.mylib.partial.default, metadata, out=view
                    )

    @inductor_config.patch(enable_auto_functionalized_v2=False)
    @parametrize("kind", ["real", "complex", "reinterpret", "conjugate", "negative"])
    @parametrize("inference", [False, True])
    def test_mutated_view_dtype_and_bits(self, device, kind, inference):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("partial(Tensor(a!) out) -> ()")

            @torch.library.impl(lib, "partial", "CompositeExplicitAutograd")
            def partial(out):
                out[:-2].add_(1000)
                out[-1:].add_(2000)

            # The implementation uses ATen operations that support view bits.
            lib.impl("partial", torch.library.fallthrough_kernel, "Conjugate")
            lib.impl("partial", torch.library.fallthrough_kernel, "Negative")

            @torch.library.register_fake("mylib::partial", lib=lib)
            def partial_fake(out):
                return None

            def f(x):
                base = x + 10
                if kind == "real":
                    view = torch.view_as_real(base)
                elif kind == "complex":
                    view = torch.view_as_complex(base)
                elif kind == "reinterpret":
                    view = base.view(torch.int32)
                elif kind == "conjugate":
                    view = base.conj()
                else:
                    view = torch._neg_view(base)
                view = view[2:, ::2]
                torch.ops.mylib.partial(view)
                return base, view

            with torch.inference_mode(inference):
                x = torch.arange(1, 129, device=device, dtype=torch.float32)
                if kind == "complex":
                    x = x.reshape(8, 8, 2)
                else:
                    x = x.reshape(8, 16)
                    if kind in ("real", "conjugate"):
                        x = x.to(torch.complex64) * (1 + 1j)
                self.assertEqual(
                    torch.compile(f, fullgraph=True)(x), f(x), atol=0, rtol=0
                )

    @inductor_config.patch(enable_auto_functionalized_v2=False)
    @parametrize("inference", [False, True])
    def test_mutated_dtype_view_with_unaligned_base(self, device, inference):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("partial(Tensor(a!) out) -> ()")

            @torch.library.impl(lib, "partial", "CompositeExplicitAutograd")
            def partial(out):
                out[:-2].add_(1000)
                out[-1:].add_(2000)

            @torch.library.register_fake("mylib::partial", lib=lib)
            def partial_fake(out):
                return None

            def f(x):
                base = x + 10
                view = torch.view_as_complex(base[:-1].reshape(8, 4, 2))[2:, ::2]
                torch.ops.mylib.partial(view)
                return base, view

            with torch.inference_mode(inference):
                x = torch.arange(1, 66, device=device, dtype=torch.float32)
                self.assertEqual(
                    torch.compile(f, fullgraph=True)(x), f(x), atol=0, rtol=0
                )

    def test_view_metadata_functionalize(self, device):
        from torch._higher_order_ops.auto_functionalize import auto_functionalized
        from torch._subclasses.functional_tensor import PythonFunctionalizeAPI
        from torch.fx.experimental.proxy_tensor import make_fx

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("partial(Tensor(a!) out) -> ()")

            @torch.library.impl(lib, "partial", "CompositeExplicitAutograd")
            def partial(out):
                out[:-2].add_(1000)
                out[-1:].add_(2000)

            @torch.library.register_fake("mylib::partial", lib=lib)
            def partial_fake(out):
                return None

            def f(x):
                base = x + 10
                view = base[2:, ::2]
                metadata = {
                    "out": (base, view.size(), view.stride(), view.storage_offset())
                }
                return auto_functionalized(
                    torch.ops.mylib.partial.default, metadata, out=view
                )

            x = torch.arange(1, 65, device=device, dtype=torch.float32).reshape(8, 8)
            graph = make_fx(PythonFunctionalizeAPI().functionalize(f))(x)
            self.assertEqual(graph(x), f(x), atol=0, rtol=0)

    @inductor_config.patch(enable_auto_functionalized_v2=False)
    @parametrize("optional", [False, True])
    @parametrize("inference", [False, True])
    def test_mutated_view_list_optional_and_defaults(self, device, optional, inference):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define(
                "partial(Tensor(a!) _tensor_bases, Tensor(b!)[] others, "
                "Tensor(c!)? optional=None, int increment=1000) -> ()"
            )

            @torch.library.impl(lib, "partial", "CompositeExplicitAutograd")
            def partial(_tensor_bases, others, optional=None, increment=1000):
                for out in [_tensor_bases, *others, optional]:
                    if out is not None:
                        out[:-2].add_(increment)
                        out[-1:].add_(2 * increment)

            @torch.library.register_fake("mylib::partial", lib=lib)
            def partial_fake(_tensor_bases, others, optional=None, increment=1000):
                return None

            def f(x):
                a, b, c, d = x + 10, x + 20, x + 30, x + 40
                views = [a[2:, ::2], b, c[2:]]
                if optional:
                    torch.ops.mylib.partial(views[0], views[1:], d[2:, ::2])
                else:
                    torch.ops.mylib.partial(views[0], views[1:])
                return a, b, c, d, views

            with torch.inference_mode(inference):
                x = torch.arange(1, 65, device=device, dtype=torch.float32).reshape(
                    8, 8
                )
                self.assertEqual(
                    torch.compile(f, fullgraph=True)(x), f(x), atol=0, rtol=0
                )


instantiate_device_type_tests(AutoFunctionalizeViewTests, globals())


instantiate_parametrized_tests(AutoFunctionalizeTests)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
