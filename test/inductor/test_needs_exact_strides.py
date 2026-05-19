# Owner(s): ["module: inductor"]

import torch
import torch.utils._pytree as pytree
from torch._inductor.pattern_matcher import (
    CallFunctionVarArgs,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_GPU_AND_TRITON


if HAS_GPU_AND_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _fill_kernel(
        buf_ptr,
        num_tokens,
        stride_t,
        stride_i,
        INNER: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= num_tokens:
            return
        for i in range(INNER):
            tl.store(buf_ptr + pid * stride_t + i * stride_i, 1.0)


class TestNeedsExactStrides(InductorTestCase):
    @parametrize("fallback_by_default", [False, True])
    def test_custom_op_inserted_without_eager_input_vals(self, fallback_by_default):
        class InsertCustomOp:
            def __init__(self, op):
                self.op = op
                self.called = False

            def __call__(self, graph):
                self.called = True
                output = next(n for n in graph.nodes if n.op == "output")
                (orig,) = output.args[0]
                with graph.inserting_before(output):
                    new = graph.call_function(self.op, (orig,), {})
                    new.meta = {
                        key: value
                        for key, value in orig.meta.items()
                        if key != "eager_input_vals"
                    }
                    output.args = ((new,),)
                graph.lint()

        with torch.library._scoped_library("mylib_exact_strides", "DEF") as lib:
            lib.define("check_last_dim(Tensor x) -> Tensor")

            seen_strides = []

            @torch.library.impl(lib, "check_last_dim", "CompositeExplicitAutograd")
            def _(x):
                seen_strides.append(tuple(x.stride()))
                if x.stride(-1) != 1 and x.size(-1) != 1:
                    raise AssertionError(
                        f"expected last dim contiguous, got {x.stride()}"
                    )
                return x.clone()

            @torch.library.impl(lib, "check_last_dim", "Meta")
            def _(x):
                return torch.empty_like(x)

            post_pass = InsertCustomOp(
                torch.ops.mylib_exact_strides.check_last_dim.default
            )

            def f(x):
                u = x.transpose(1, 2).contiguous()
                if u.stride(-1) != 1:
                    u = u.contiguous()
                return u + 1

            x = torch.randn(1, 8, 4)
            expected = f(x)

            from torch._inductor import config

            with config.patch(
                implicit_fallbacks=not fallback_by_default,
                fallback_by_default=fallback_by_default,
                post_grad_custom_post_pass=post_pass,
            ):
                actual = torch.compile(f, fullgraph=True)(x)

            self.assertTrue(post_pass.called)
            self.assertEqual(actual, expected)
            self.assertEqual(seen_strides, [(32, 8, 1)])

    def test_registered_lowering_without_tag_has_no_default_constraint(self):
        with torch.library._scoped_library("mylib_lowering_layout", "DEF") as lib:
            lib.define("lowered_identity(Tensor x) -> Tensor")

            @torch.library.impl(lib, "lowered_identity", "Meta")
            def _(x):
                return torch.empty_like(x)

            from torch._inductor.graph import GraphLowering
            from torch._inductor.lowering import (
                constrain_to_fake_tensors,
                register_lowering,
            )

            target = torch.ops.mylib_lowering_layout.lowered_identity.default

            @register_lowering(target)
            def _(x):
                return x

            self.assertIsNone(GraphLowering._layout_constraints_for_target(target))
            self.assertIs(
                GraphLowering._layout_constraints_for_target(target, with_default=True),
                constrain_to_fake_tensors,
            )

    def test_registered_lowering_selective_fallback_uses_default_constraint(self):
        class InsertFallbackCustomOp:
            def __init__(self, op):
                self.op = op
                self.called = False

            def __call__(self, graph):
                self.called = True
                output = next(n for n in graph.nodes if n.op == "output")
                (orig,) = output.args[0]
                with graph.inserting_before(output):
                    new = graph.call_function(self.op, (orig,), {})
                    new.meta = {
                        key: value
                        for key, value in orig.meta.items()
                        if key != "eager_input_vals"
                    }
                    custom_meta = dict(new.meta.get("custom", {}))
                    custom_meta["fallback_to_eager"] = True
                    new.meta["custom"] = custom_meta
                    output.args = ((new,),)
                graph.lint()

        with torch.library._scoped_library(
            "mylib_lowering_fallback_layout", "DEF"
        ) as lib:
            lib.define("check_last_dim(Tensor x) -> Tensor")

            seen_strides = []

            @torch.library.impl(lib, "check_last_dim", "CompositeExplicitAutograd")
            def _(x):
                seen_strides.append(tuple(x.stride()))
                if x.stride(-1) != 1 and x.size(-1) != 1:
                    raise AssertionError(
                        f"expected last dim contiguous, got {x.stride()}"
                    )
                return x.clone()

            @torch.library.impl(lib, "check_last_dim", "Meta")
            def _(x):
                return torch.empty_like(x)

            from torch._inductor import config
            from torch._inductor.lowering import lowerings, register_lowering

            target = torch.ops.mylib_lowering_fallback_layout.check_last_dim.default

            @register_lowering(target)
            def _(x):
                return x

            try:
                post_pass = InsertFallbackCustomOp(target)

                def f(x):
                    u = x.transpose(1, 2).contiguous()
                    if u.stride(-1) != 1:
                        u = u.contiguous()
                    return u + 1

                x = torch.randn(1, 8, 4)
                expected = f(x)

                with config.patch(post_grad_custom_post_pass=post_pass):
                    actual = torch.compile(f, fullgraph=True)(x)

                self.assertTrue(post_pass.called)
                self.assertEqual(actual, expected)
                self.assertEqual(seen_strides, [(32, 8, 1)])
            finally:
                lowerings.pop(target, None)

    @parametrize("dtype", [torch.float, torch.float8_e8m0fnu])
    def test_custom_op(self, dtype):
        device = (
            torch.accelerator.current_accelerator()
        )  # float8_e8m0fnu errors on "cpu"
        x = torch.ones(4, 4, 2, 2, device=device, dtype=torch.float8_e8m0fnu)
        other = torch.ones(4, 4, 2, 2, device=device, dtype=torch.float8_e8m0fnu)

        class _CustomPass(PatternMatcherPass):
            def __init__(self) -> None:
                super().__init__()

            def __call__(self, g: torch.fx.Graph):
                self.apply(g)

        g = _CustomPass()
        called = False

        @register_graph_pattern(
            CallFunctionVarArgs(torch.ops.aten.permute),
            pass_dict=g,
        )
        def _(match, *args, **kwargs):
            flat_args, spec = pytree.tree_flatten((args, kwargs))

            def decomp(*flat_args):
                args, kwargs = pytree.tree_unflatten(flat_args, spec)
                return torch.ops.mylib.force_channels_last(
                    torch.ops.aten.permute(*args, **kwargs)
                )

            nonlocal called
            called = True
            match.replace_by_example(decomp, flat_args)

        from torch._inductor import config

        class TestPassed(RuntimeError):
            pass

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define(
                "force_channels_last(Tensor x) -> Tensor",
                tags=[torch._C.Tag.flexible_layout],
            )

            def impl2(x):
                return x.clone(memory_format=torch.channels_last)

            lib.impl("force_channels_last", impl2, "CompositeExplicitAutograd")

            lib.define(
                "add_op(Tensor x, Tensor y) -> Tensor",
            )

            def impl(x, y):
                if not x.transpose(2, 3).is_contiguous():
                    raise AssertionError
                if not y.is_contiguous():
                    raise AssertionError
                return x.float() + y.float()

            def meta(x, y):
                return x.float() + y.float()

            lib.impl("add_op", impl, "CompositeExplicitAutograd")
            lib.impl("add_op", meta, "Meta")

            def f(x, other):
                return torch.ops.mylib.add_op.default(x.transpose(2, 3), other)

            with config.patch(
                post_grad_custom_post_pass=g,
            ):
                try:
                    f_compile = torch.compile(f, fullgraph=True)
                    f_compile(x, other)
                except TestPassed:
                    pass
                if not called:
                    raise AssertionError

    @parametrize("n", [64, 128, 256])
    def test_dynamic_size_one_leading_dim_view_triton_kernel_wrapper_functional(
        self, n
    ):
        device = torch.accelerator.current_accelerator()
        inner = 8

        def fn(x):
            num_tokens = x.shape[0]
            aligned_num_tokens = ((num_tokens + 3) // 4) * 4
            buf = torch.empty(
                inner * aligned_num_tokens,
                dtype=torch.float32,
                device=x.device,
            ).as_strided(
                (1, num_tokens, inner),
                (inner * aligned_num_tokens, 1, aligned_num_tokens),
            )

            _fill_kernel[(num_tokens,)](
                buf,
                num_tokens,
                stride_t=buf.stride(1),
                stride_i=buf.stride(2),
                INNER=inner,
            )

            return buf

        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
        x = torch.randn(n, inner, device=device)
        self.assertEqual(compiled(x.clone()), fn(x.clone()))


instantiate_parametrized_tests(TestNeedsExactStrides)

if __name__ == "__main__":
    if IS_LINUX and HAS_GPU_AND_TRITON:
        run_tests(needs="filelock")
