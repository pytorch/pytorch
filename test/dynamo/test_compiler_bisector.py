# Owner(s): ["module: dynamo"]

import unittest
from contextlib import contextmanager
from importlib import import_module

import torch
import torch._prims_common as utils
from torch._dynamo.utils import preserve_rng_state
from torch._inductor import config
from torch._inductor.compiler_bisector import CompilerBisector
from torch._inductor.custom_graph_pass import CustomGraphPass
from torch._inductor.test_case import TestCase
from torch.library import _scoped_library
from torch.testing._internal.common_utils import HardwareClassification


aten = torch.ops.aten


f32 = torch.float32
i64 = torch.int64
i32 = torch.int32


class TestCompilerBisector(TestCase):
    hw_classification = HardwareClassification.GENERIC
    bisector_ns = "_test_bisector"

    def tearDown(self):
        if hasattr(torch.ops, self.bisector_ns):
            delattr(torch.ops, self.bisector_ns)

    def get_op(self, name):
        return getattr(getattr(torch.ops, self.bisector_ns), name).default

    def test_pre_grad(self):
        import operator

        from torch._inductor import config

        class CustomPrePass(CustomGraphPass):
            def __call__(self, graph: torch.fx.Graph):
                nodes = graph.find_nodes(op="call_function", target=operator.add)
                if len(nodes) != 1:
                    raise AssertionError(f"Expected 1 node, got {len(nodes)}")
                args = list(nodes[0].args)
                args[1] = 2
                nodes[0].args = tuple(args)

            def uuid(self):
                return hash("TestCompilerBisector.test_pre_grad.pass_class")

        def foo(x):
            return x + 1

        def test_fn():
            torch._dynamo.reset()

            inp = torch.rand([10])

            out = foo(inp)
            out_c = torch.compile(foo)(inp)  # noqa: UNSPECIFIED_BACKEND

            return torch.allclose(out, out_c)

        with config.patch(pre_grad_custom_pass=CustomPrePass()):
            out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "inductor")
        self.assertEqual(out.subsystem, "pre_grad_passes")
        self.assertEqual(out.bisect_number, 3)
        self.assertTrue("pre_grad_custom_pass" in out.debug_info)

    def test_crossref(self):
        with _scoped_library(self.bisector_ns, "FRAGMENT") as lib:
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
                    torch.testing.assert_close(torch.compile(op)(x), op(x))  # noqa: UNSPECIFIED_BACKEND
                except Exception:
                    return False
                return True

            out = CompilerBisector.do_bisect(test_fn)
            self.assertEqual(out.backend, "aot_eager_decomp_partition_crossref")

    def test_eager_backend(self):
        # should indicate problem with first backend
        def test_fn():
            return False

        out = CompilerBisector.do_bisect(test_fn)
        self.assertEqual(out.backend, "eager")
        self.assertEqual(out.subsystem, None)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
