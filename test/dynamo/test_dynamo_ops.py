# Owner(s): ["module: dynamo"]

"""
Tests for operator behavior during Dynamo tracing.

This file contains tests that use op_db to verify correct behavior of operators
when traced by Dynamo.
"""

import torch
import torch._dynamo.test_case
from torch._dynamo.comptime import comptime, ComptimeContext
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db, skip, skipOps


# Ops that fail the inplace requires_grad propagation test for known reasons
test_inplace_ops_propagate_requires_grad_metadata_skips = {
    # Not implemented for floating point types
    skip("bitwise_and"),
    skip("bitwise_left_shift"),
    skip("bitwise_or"),
    skip("bitwise_right_shift"),
    skip("bitwise_xor"),
    skip("gcd"),
    skip("lcm"),
    # out=... arguments don't support automatic differentiation
    skip("ldexp"),
    # Backward not implemented
    skip("floor_divide"),
    skip("heaviside"),
    skip("nextafter"),
    # Output does not require grad (logical ops return bool)
    skip("logical_and"),
    skip("logical_or"),
    skip("logical_xor"),
    skip("resize_as_"),
    # Dtype issues
    skip("float_power"),
    # Numerical gradient mismatch (not a metadata propagation issue)
    skip("igamma"),
    skip("igammac"),
}


class TestTensorMetaProp(torch._dynamo.test_case.TestCase):
    """
    Test that inplace operations correctly propagate tensor metadata during Dynamo tracing.
    """

    @ops([op for op in op_db if op.get_inplace() is not None])
    @skipOps(
        "TestTensorMetaProp",
        "test_inplace_ops_propagate_requires_grad_metadata",
        test_inplace_ops_propagate_requires_grad_metadata_skips,
    )
    def test_inplace_ops_propagate_requires_grad_metadata(self, device, dtype, op):
        """
        Test that inplace ops from OpInfo propagate requires_grad correctly.

        This test ensures that when an inplace operation is performed on a tensor
        without requires_grad using an argument with requires_grad=True, the metadata
        is correctly propagated in both eager and compiled modes.

        This is critical because if metadata is traced incorrectly, code that branches
        on requires_grad (like custom autograd functions) will take the wrong path,
        leading to silent incorrectness.
        """

        inplace_op = op.get_inplace()
        if inplace_op is None:
            self.skipTest("No inplace variant for this op")

        class CustomAutograd(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x * 2

            @staticmethod
            def backward(ctx, grad_out):
                # Return an obviously wrong gradient (fixed value) to detect
                # when composite implicit autograd is used vs custom backward
                (x,) = ctx.saved_tensors
                return torch.full_like(x, 123.0)

        # Iterate directly over sample_inputs to preserve correct tracking
        # (converting to list first breaks the TrackedInputIter tracking)
        for sample in op.sample_inputs(device, dtype, requires_grad=False):
            # Skip samples that are broadcasted or have 0 elements
            if sample.broadcasts_input or sample.input.numel() == 0:
                continue

            # Skip scatter with reduce modes - backward not implemented for these
            if op.name == "scatter" and "reduce" in sample.kwargs:
                continue

            # Reset between samples to avoid exceeding recompile limit
            torch._dynamo.reset()

            # Setup: x starts with requires_grad=False, one arg has requires_grad=True
            x_eager = sample.input.clone().detach()
            args_eager = [
                arg.clone().detach() if isinstance(arg, torch.Tensor) else arg
                for arg in sample.args
            ]

            # Find a floating point tensor arg to set requires_grad=True
            requires_grad_idx = None
            for idx, arg in enumerate(args_eager):
                if isinstance(arg, torch.Tensor) and arg.dtype.is_floating_point:
                    arg.requires_grad_(True)
                    requires_grad_idx = idx
                    break

            if requires_grad_idx is None or x_eager.requires_grad:
                continue

            # Apply inplace op in eager mode
            inplace_op(x_eager, *args_eager, **sample.kwargs)
            output_eager = CustomAutograd.apply(x_eager)
            output_eager.sum().backward()

            # Setup compiled version
            x_compiled = sample.input.clone().detach()
            args_compiled = [
                arg.clone().detach() if isinstance(arg, torch.Tensor) else arg
                for arg in sample.args
            ]
            args_compiled[requires_grad_idx].requires_grad_(True)

            # Test 1: Verify that the metadata is propagated after the inplace op in compile time
            def compile_time_check(ctx: ComptimeContext) -> None:
                x = ctx.get_local("x")
                x_fake = x.as_fake()
                # Check requires_grad is propagated
                self.assertTrue(x_fake.requires_grad)
                self.assertTrue(x._ComptimeVar__variable.requires_grad)
                # Check that has_grad_fn is set (not for FakeTensor)
                self.assertTrue(x._ComptimeVar__variable.has_grad_fn)
                # Check dtype is preserved
                self.assertEqual(x_fake.dtype, dtype)
                self.assertEqual(x._ComptimeVar__variable.dtype, dtype)

            def fn(x, *args):
                inplace_op(x, *args, **sample.kwargs)
                comptime(compile_time_check)
                r = CustomAutograd.apply(x)
                return r

            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            output_compiled = compiled_fn(x_compiled, *args_compiled)
            output_compiled.sum().backward()

            # Test 2: Verify requires_grad was propagated in runtime
            self.assertEqual(
                x_eager.requires_grad,
                x_compiled.requires_grad,
                msg=f"{op.name}: requires_grad mismatch (eager={x_eager.requires_grad}, compiled={x_compiled.requires_grad})",
            )

            # Test 3: Verify gradients match (with tolerance for float16/bfloat16)
            self.assertEqual(
                args_eager[requires_grad_idx].grad,
                args_compiled[requires_grad_idx].grad,
                atol=1e-2,
                rtol=1e-2,
                msg=f"{op.name}: Gradient mismatch indicates metadata not propagated during tracing",
            )


instantiate_device_type_tests(TestTensorMetaProp, globals())


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
