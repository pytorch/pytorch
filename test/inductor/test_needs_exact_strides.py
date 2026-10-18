# Owner(s): ["module: inductor"]

import unittest

import torch
import torch.utils._pytree as pytree
from torch._inductor import config
from torch._inductor.pattern_matcher import (
    CallFunctionVarArgs,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyOn,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU_AND_TRITON


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


@unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
class TestNeedsExactStrides(InductorTestCase):
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


class TestCustomOpLayoutReuse(InductorTestCase):
    @parametrize("same_input", [False, True])
    @parametrize("reuse", [False, True])
    def test_input_storage_and_gradients(self, device, same_input, reuse):
        observed = []
        with (
            torch.library._scoped_library("layout_reuse", "FRAGMENT") as lib,
            config.patch(cache_layout_constrained_inputs=reuse),
        ):
            lib.define("pair(Tensor k, Tensor v) -> Tensor")

            def impl(k, v):
                observed.append((k.data_ptr() == v.data_ptr(), k.stride(), v.stride()))
                return k + v

            def backward(ctx, grad):
                return grad, grad

            lib.impl("pair", impl, "CompositeExplicitAutograd")
            lib.impl("pair", lambda k, v: k + v, "Meta")
            torch.library.register_autograd("layout_reuse::pair", backward, lib=lib)
            op = torch.ops.layout_reuse.pair.default

            def f(x, y):
                h = x + 1
                return op(h, v=h if same_input else y + 1)

            compiled = torch.compile(f, fullgraph=True, dynamic=True)
            for n in (17, 19):
                x = torch.arange(n * 8, device=device, dtype=torch.float).reshape(8, n)
                x.requires_grad_()
                y = x.detach().clone().requires_grad_()
                expected = f(x.detach(), y.detach())
                observed.clear()
                actual = compiled(x, y)
                actual.sum().backward()
                self.assertEqual(actual, expected, atol=0, rtol=0)
                self.assertEqual(
                    x.grad, torch.full_like(x, 2 if same_input else 1), atol=0, rtol=0
                )
                self.assertEqual(
                    y.grad, None if same_input else torch.ones_like(y), atol=0, rtol=0
                )
                self.assertEqual(actual.stride(), (n, 1))
                self.assertTrue(observed)
                for shared, k_stride, v_stride in observed:
                    self.assertEqual(shared, same_input and reuse)
                    self.assertEqual(k_stride, (n, 1))
                    self.assertEqual(v_stride, (n, 1))

    @config.patch(cache_layout_constrained_inputs=True)
    def test_cache_key_and_containers(self, device):
        from torch._inductor import lowering
        from torch._inductor.graph import GraphLowering
        from torch._inductor.virtualized import V
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode() as mode:
            x = torch.empty(3, 4, device=device)
            other = torch.empty_strided((3, 4), (1, 3), device=device)
            gm = torch.fx.symbolic_trace(lambda x: x)
            graph = GraphLowering(gm, example_inputs=[x])
            with V.set_graph_handler(graph), V.set_fake_mode(mode):
                graph.run(x)
                source = next(iter(graph.graph_inputs.values()))
                args, kwargs = lowering.constrain_to_fake_tensors(
                    (source, [source]),
                    {"nested": {"same": source}},
                    (x, [other]),
                    {"nested": {"same": other}},
                )
                self.assertEqual(args[0].get_stride(), list(x.stride()))
                self.assertEqual(args[1][0].get_stride(), list(other.stride()))
                self.assertIsNot(args[0], args[1][0])
                self.assertIs(args[1][0], kwargs["nested"]["same"])

    @parametrize("both_write", [False, True])
    @config.patch(cache_layout_constrained_inputs=True)
    def test_mutation_after_layout_copy(self, device, both_write):
        from torch._inductor.compile_fx import compile_fx_inner
        from torch.fx.experimental.proxy_tensor import make_fx

        observed = []
        with torch.library._scoped_library("layout_reuse", "FRAGMENT") as lib:
            name = "mutate_both" if both_write else "mutate_k"
            v_schema = "Tensor(b!) v" if both_write else "Tensor v"
            lib.define(f"{name}(Tensor(a!) k, {v_schema}) -> Tensor")

            def impl(k, v):
                observed.append((k.data_ptr() == v.data_ptr(), k.stride(), v.stride()))
                k.add_(3)
                if both_write:
                    v.mul_(2)
                return v.clone()

            lib.impl(name, impl, "CompositeExplicitAutograd")
            lib.impl(name, lambda k, v: torch.empty_like(v), "Meta")
            op = getattr(torch.ops.layout_reuse, name).default

            def f(x):
                return op(x, x), op(x, x), x

            x = torch.arange(136, device=device, dtype=torch.float32).reshape(8, 17).t()
            eager_x = x.clone()
            expected = f(eager_x)
            observed.clear()
            # Bypass AOT and force layout copies by overriding the traced strides.
            gm = make_fx(f, tracing_mode="fake")(x)
            for node in gm.graph.nodes:
                if node.target == op:
                    fake_x = torch.empty_like(
                        node.args[0].meta["val"], memory_format=torch.contiguous_format
                    )
                    node.meta["eager_input_vals"] = ((fake_x, fake_x), {})
            compiled = compile_fx_inner(gm, [x])
            self.assertEqual(compiled([x]), expected, atol=0, rtol=0)
            self.assertEqual(x, eager_x, atol=0, rtol=0)
            self.assertEqual(observed, [(True, (8, 1), (8, 1))] * 2)

    @onlyOn(GPU_TYPE)
    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    @config.patch(
        cache_layout_constrained_inputs=True,
        triton_kernel_default_layout_constraint="needs_fixed_stride_order",
    )
    def test_triton_mutation_after_layout_copy(self, device):
        from torch._inductor.compile_fx import compile_fx_inner
        from torch._library import capture_triton
        from torch.fx.experimental.proxy_tensor import make_fx

        @triton.jit
        def kernel(k, v, out):
            tl.store(k, tl.load(k) + 3)
            tl.debug_barrier()
            tl.store(out, tl.load(v, volatile=True))

        def f(x, out):
            capture_triton(kernel)[(1,)](x, x, out)
            return out, x

        x = torch.arange(136, device=device, dtype=torch.float32).reshape(8, 17).t()
        out = torch.empty((), device=device)
        eager_x = x.clone()
        expected = f(eager_x, torch.empty_like(out))
        gm = make_fx(f, tracing_mode="fake")(x, out)
        node = next(
            n
            for n in gm.graph.nodes
            if n.target is torch.ops.higher_order.triton_kernel_wrapper_mutation
        )
        # Force layout copies for both pointers to the transposed input.
        fake_x = torch.empty_like(
            node.kwargs["kwargs"]["k"].meta["val"],
            memory_format=torch.contiguous_format,
        )
        fake_out = node.kwargs["kwargs"]["out"].meta["val"]
        node.meta["eager_input_vals"] = (
            (),
            {**node.kwargs, "kwargs": {"k": fake_x, "v": fake_x, "out": fake_out}},
        )
        compiled = compile_fx_inner(gm, [x, out])
        self.assertEqual(compiled([x, out]), expected, atol=0, rtol=0)
        self.assertEqual(x, eager_x, atol=0, rtol=0)


instantiate_parametrized_tests(TestNeedsExactStrides)
instantiate_device_type_tests(
    TestCustomOpLayoutReuse, globals(), only_for=("cpu", GPU_TYPE), allow_xpu=True
)

if __name__ == "__main__":
    if IS_LINUX:
        run_tests(needs="filelock")
