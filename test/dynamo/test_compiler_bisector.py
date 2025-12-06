# Owner(s): ["module: dynamo"]

from contextlib import contextmanager
from importlib import import_module

import torch
import torch._prims_common as utils
from torch._dynamo.utils import preserve_rng_state
from torch._inductor import config
from torch._inductor.compiler_bisector import CompilerBisector
from torch._inductor.test_case import TestCase
from torch.library import _scoped_library, Library
from torch.testing._internal.triton_utils import requires_cuda_and_triton


aten = torch.ops.aten


f32 = torch.float32
i64 = torch.int64
i32 = torch.int32


@requires_cuda_and_triton
class TestCompilerBisector(TestCase):
    test_ns = "_test_bisector"

    def tearDown(self):
        if hasattr(torch.ops, self.test_ns):
            delattr(torch.ops, self.test_ns)
        if hasattr(self, "lib"):
            del self.lib.m
            del self.lib

    def get_op(self, name):
        return getattr(getattr(torch.ops, self.test_ns), name).default

    def get_lib(self):
        lib = Library(self.test_ns, "FRAGMENT")  # noqa: TOR901
        self.lib = lib
        return lib

    def test_bad_decomp(self):
        import_module("torch._inductor.compile_fx")

        def bad_exp_decomp(self, rate=1, generator=None):
            assert generator is None
            torch._check(
                not utils.is_complex_dtype(self.dtype)
                and not utils.is_integer_dtype(self.dtype)
                and not utils.is_boolean_dtype(self.dtype),
                lambda: f"Exponential distribution is a continuous probability distribution. \
                dtype must be a floating point but you specified {self.dtype}",
            )
            torch._check(
                rate > 0.0,
                lambda: f"exponential_ expects lambda > 0.0, but found lambda={rate}",
            )
            return torch.rand_like(self) * float("nan")

        @contextmanager
        def patch_exp_decomp():
            from torch._inductor.compile_fx import select_decomp_table as old_decomp

            def get_decomp():
                out = old_decomp()
                out = out.copy()
                out[aten.exponential.default] = bad_exp_decomp
                return out

            torch._inductor.compile_fx.select_decomp_table = get_decomp
            try:
                yield

            finally:
                torch._inductor.compile_fx.select_decomp_table = old_decomp

        def vq(x):
            return (x + 3).exponential_() * 10.5

        def test_fn():
            torch._dynamo.reset()
            with patch_exp_decomp():
                vq_compiled = torch.compile(vq)
                x = torch.randn(4, 400, 256).cuda()
                with torch._dynamo.utils.preserve_rng_state():
                    vq(x)
                out_compiled = vq_compiled(x)

            return not out_compiled.isnan().any()

        out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "aot_eager_decomp_partition")
        self.assertEqual(out.subsystem, "decomposition")
        self.assertEqual(out.bisect_number, 1)
        self.assertTrue("aten.exponential" in out.debug_info)

    def test_pre_grad(self):
        import operator

        from torch._inductor import config

        # similar setup to test_joint_graph (see below)
        def pass_fn(graph: torch.fx.Graph):
            nodes = graph.find_nodes(op="call_function", target=operator.add)
            assert len(nodes) == 1
            args = list(nodes[0].args)
            args[1] = 2
            nodes[0].args = tuple(args)

        config.pre_grad_custom_pass = pass_fn

        def foo(x):
            return x + 1

        def test_fn():
            torch._dynamo.reset()

            inp = torch.rand([10])

            out = foo(inp)
            out_c = torch.compile(foo)(inp)

            return torch.allclose(out, out_c)

        out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "inductor")
        self.assertEqual(out.subsystem, "pre_grad_passes")
        self.assertEqual(out.bisect_number, 0)
        self.assertTrue("pre_grad_custom_pass" in out.debug_info)

    def test_joint_graph(self):
        from torch._inductor import config

        def pass_fn(graph: torch.fx.Graph):
            nodes = graph.find_nodes(
                op="call_function", target=torch.ops.aten.add.Tensor
            )
            assert len(nodes) == 1
            args = list(nodes[0].args)
            args[1] = 2
            nodes[0].args = tuple(args)

        config.joint_custom_post_pass = pass_fn

        def foo(x):
            return x + 1

        def test_fn():
            torch._dynamo.reset()

            inp = torch.rand([10], device="cuda")

            out = foo(inp)
            out_c = torch.compile(foo)(inp)

            return torch.allclose(out, out_c)

        out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "inductor")
        self.assertEqual(out.subsystem, "joint_graph_passes")
        self.assertEqual(out.bisect_number, 4)
        self.assertTrue("joint_custom_post_pass" in out.debug_info)

    def test_rng(self):
        def foo():
            return torch.rand([10], device="cuda") + 1

        def test_fn():
            torch._dynamo.reset()

            with preserve_rng_state():
                out = foo()
            with preserve_rng_state():
                out_c = torch.compile(foo)()

            return torch.allclose(out, out_c)

        out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "inductor")
        self.assertEqual(out.subsystem, "inductor_fallback_random")
        self.assertTrue("inductor_fallback_random" in out.debug_info)

    def test_crossref(self):
        with _scoped_library(self.test_ns, "FRAGMENT") as lib:
            lib.define("foo(Tensor x) -> Tensor")
            op = self.get_op("foo")

            class Foo(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    # Emulate AutoDispatchBelowADInplaceOrView, which is not bound into python
                    with torch._C._AutoDispatchBelowAutograd():
                        with torch._C._ExcludeDispatchKeyGuard(
                            torch._C.DispatchKeySet(
                                torch._C.DispatchKey.ADInplaceOrView
                            )
                        ):
                            return op(x)

                @staticmethod
                def backward(ctx, gx):
                    return gx

            def foo_impl(x):
                return x.view_as(x).clone()

            def foo_meta(x):
                return x.view_as(x)

            lib.impl("foo", Foo.apply, "Autograd")
            lib.impl("foo", foo_impl, "CPU")
            lib.impl("foo", foo_meta, "Meta")

            x = torch.tensor(3.14159 / 3, requires_grad=True)

            def test_fn():
                torch._dynamo.reset()

                try:
                    torch.testing.assert_close(torch.compile(op)(x), op(x))
                except Exception:
                    return False
                return True

            out = CompilerBisector.do_bisect(test_fn)
            self.assertEqual(out.backend, "aot_eager_decomp_partition_crossref")

    def test_emulate_precision_casts(self):
        def test_fn():
            torch._dynamo.reset()

            def calculate_scale(inp):
                amax = torch.abs(torch.max(inp))
                scale = 448.0 / torch.clamp(amax, min=1e-12)
                scale = scale.to(torch.float32)
                return scale

            dtype = torch.bfloat16
            torch.manual_seed(0)
            inp = torch.randn(16, 16, 768, dtype=dtype, device="cuda")
            eager_scale = calculate_scale(inp)
            compile_scale = torch.compile(calculate_scale)(inp)

            return torch.equal(eager_scale, compile_scale)

        out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "inductor")
        self.assertEqual(out.subsystem, "inductor_emulate_precision_casts")

    def test_bad_lowering(self):
        def test_fn():
            torch._dynamo.reset()
            with config.patch("triton.inject_relu_bug_TESTING_ONLY", "accuracy"):

                def my_func(x):
                    return ((x * -1) - 0.01).relu()

                inp = torch.rand([100], device="cuda")

                return torch.allclose(torch.compile(my_func)(inp), my_func(inp))

        out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "inductor")
        self.assertEqual(out.subsystem, "lowerings")
        self.assertEqual(out.bisect_number, 2)
        self.assertTrue("relu" in out.debug_info)

    def test_eager_backend(self):
        # should indicate problem with first backend
        def test_fn():
            return False

        out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "eager")
        self.assertEqual(out.subsystem, None)

    @config.patch(
        {
            "test_configs.bisect_pre_grad_graph": True,
            "test_configs.bisect_keep_custom_backend_for_inductor": True,
        }
    )
    def test_bisect_pre_grad_graph(self):
        def f(x):
            for _ in range(5):
                x = x + 1
            return x.relu()

        class MyBackend:
            def __call__(self, gm, example_inputs):
                node_idx = 0

                def node_to_graph_id(node):
                    nonlocal node_idx
                    out = 0 if node_idx < 3 else 1
                    node_idx += 1
                    return out

                split_gm = torch.fx.passes.split_module.split_module(
                    gm, None, node_to_graph_id, keep_original_order=True
                )

                for name, submod in split_gm.named_modules():
                    if "submod_" in name:
                        # the test case is simple enough that using
                        # the original example_inputs works for sub
                        # moule
                        submod.forward = torch._inductor.standalone_compile(
                            submod,
                            example_inputs,
                            dynamic_shapes="from_example_inputs",
                            options={},
                        )

                return split_gm

        def test_fn():
            torch._dynamo.reset()

            x = torch.randn(1024, device="cuda")
            with config.patch("triton.inject_relu_bug_TESTING_ONLY", "accuracy"):
                opt_f = torch.compile(f, backend=MyBackend())
                return torch.allclose(opt_f(x), f(x))

        out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "inductor")
        self.assertEqual(out.subsystem, "pre_grad_graph")
        self.assertEqual(out.bisect_number, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
