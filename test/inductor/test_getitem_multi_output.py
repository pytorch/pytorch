# Owner(s): ["module: inductor"]
"""
Tests for operator.getitem handling on IR nodes with MultiOutputLayout
in GraphLowering.call_function (graph.py).

When a multi-output operation is lowered to a single IR node with
MultiOutputLayout rather than a Python container, operator.getitem
must create a MultiOutput IR node to extract the indexed result.
"""

import torch
import torch._inductor.ir as ir
from torch._inductor.lowering import lowerings
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


DEVICES = ("cpu", GPU_TYPE) if HAS_GPU else ("cpu",)


def _register_multi_out_op(lib):
    """Register a custom op that returns a tuple of two tensors."""
    lib.define("multi_out(Tensor x) -> (Tensor, Tensor)")

    def _impl(x):
        return (x + 1, x * 2)

    lib.impl("multi_out", _impl, "CompositeExplicitAutograd")

    @torch.library.register_fake("testlib::multi_out", lib=lib)
    def _fake(x):
        return (x.new_empty(x.shape), x.new_empty(x.shape))

    return torch.ops.testlib.multi_out


def _make_packed_multi_output_lowering(op):
    """Create a lowering that returns the packed FallbackKernel directly.

    Instead of returning a tuple of MultiOutput nodes (the normal path),
    this returns the raw FallbackKernel with MultiOutputLayout wrapped in
    TensorBox. This simulates the scenario where a multi-output op's
    lowering produces a single IR node, triggering the operator.getitem
    handling for IR nodes in GraphLowering.call_function.
    """

    def lowering_fn(*args, **kwargs):
        kernel = op.default

        with V.graph.fake_mode:
            (
                example_output,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                unbacked_bindings,
            ) = ir.FallbackKernel.process_kernel(kernel, *args, **kwargs)

        device = ir.FallbackKernel.find_device(tensor_args, example_output)
        packed = ir.FallbackKernel(
            ir.MultiOutputLayout(device=device),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
            kwargs={},
            unbacked_bindings=unbacked_bindings,
        )
        packed.outputs = []
        return packed.wrap_for_lowering()

    return lowering_fn


@instantiate_parametrized_tests
class TestGetitemMultiOutput(InductorTestCase):
    """Tests for operator.getitem on IR nodes in GraphLowering."""

    @parametrize("device", DEVICES)
    def test_multi_output_normal_path(self, device):
        """Baseline: multi-output op with normal fallback lowering works."""
        with torch.library._scoped_library("testlib", "FRAGMENT") as lib:
            _register_multi_out_op(lib)

            def f(x):
                a, b = torch.ops.testlib.multi_out(x)
                return a + b

            x = torch.randn(4, 4, device=device)
            eager_out = f(x)

            compiled_out = torch.compile(f, backend="inductor", fullgraph=True)(x)
            self.assertEqual(compiled_out, eager_out)

    @parametrize("device", DEVICES)
    def test_getitem_on_multi_output_ir_node(self, device):
        """operator.getitem on an IR node with MultiOutputLayout creates MultiOutput."""
        with torch.library._scoped_library("testlib", "FRAGMENT") as lib:
            op = _register_multi_out_op(lib)

            def f(x):
                a, b = torch.ops.testlib.multi_out(x)
                return a + b

            x = torch.randn(4, 4, device=device)
            eager_out = f(x)

            # Patch the lowering to return a packed FallbackKernel (single IR
            # node with MultiOutputLayout) instead of a tuple of MultiOutput
            # nodes. This exercises the new getitem-on-IR-node code path.
            packed_lowering = _make_packed_multi_output_lowering(op)
            original = lowerings.get(op.default)
            try:
                lowerings[op.default] = packed_lowering
                compiled_out = torch.compile(f, backend="inductor", fullgraph=True)(x)
                self.assertEqual(compiled_out, eager_out)
            finally:
                if original is not None:
                    lowerings[op.default] = original
                else:
                    lowerings.pop(op.default, None)

    @parametrize("device", DEVICES)
    def test_getitem_on_multi_output_first_element(self, device):
        """operator.getitem extracts only the first element correctly."""
        with torch.library._scoped_library("testlib", "FRAGMENT") as lib:
            op = _register_multi_out_op(lib)

            def f(x):
                a, _b = torch.ops.testlib.multi_out(x)
                return a

            x = torch.randn(4, 4, device=device)
            eager_out = f(x)

            packed_lowering = _make_packed_multi_output_lowering(op)
            original = lowerings.get(op.default)
            try:
                lowerings[op.default] = packed_lowering
                compiled_out = torch.compile(f, backend="inductor", fullgraph=True)(x)
                self.assertEqual(compiled_out, eager_out)
            finally:
                if original is not None:
                    lowerings[op.default] = original
                else:
                    lowerings.pop(op.default, None)

    @parametrize("device", DEVICES)
    def test_getitem_on_multi_output_second_element(self, device):
        """operator.getitem extracts only the second element correctly."""
        with torch.library._scoped_library("testlib", "FRAGMENT") as lib:
            op = _register_multi_out_op(lib)

            def f(x):
                _a, b = torch.ops.testlib.multi_out(x)
                return b

            x = torch.randn(4, 4, device=device)
            eager_out = f(x)

            packed_lowering = _make_packed_multi_output_lowering(op)
            original = lowerings.get(op.default)
            try:
                lowerings[op.default] = packed_lowering
                compiled_out = torch.compile(f, backend="inductor", fullgraph=True)(x)
                self.assertEqual(compiled_out, eager_out)
            finally:
                if original is not None:
                    lowerings[op.default] = original
                else:
                    lowerings.pop(op.default, None)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
