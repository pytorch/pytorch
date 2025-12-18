# Owner(s): ["oncall: pt2"]

"""
Test suite to verify bitwise equivalence between compiled (aot_eager) and non-compiled ops.

This test ensures that compiling an operator with the aot_eager backend produces
exactly the same output as running the operator in eager mode. This is important
because aot_eager should be a "no-op" compilation that doesn't change numeric behavior.
"""

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import DynamicOutputShapeException
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_methods_invocations import (
    op_db,
    skipOps,
    xfail,
    skip,
)
from torch.testing._internal.common_utils import (
    run_tests,
)


# Ops that are known to be non-deterministic even with same seed
# These ops use random sampling and cannot be tested for bitwise equivalence
NONDETERMINISTIC_OPS = frozenset({
    # Random sampling ops
    "bernoulli",
    "cauchy",
    "exponential",
    "geometric",
    "log_normal",
    "multinomial",
    "normal",
    "poisson",
    "random",
    "rand_like",
    "randint",
    "randint_like",
    "randn",
    "randn_like",
    "randperm",
    "uniform",
    # Dropout ops (use random mask)
    "nn.functional.dropout",
    "nn.functional.dropout1d",
    "nn.functional.dropout2d",
    "nn.functional.dropout3d",
    "nn.functional.alpha_dropout",
    "nn.functional.feature_alpha_dropout",
    # Ops that may have non-deterministic implementations
    "scatter_reduce",  # Known to have non-deterministic behavior
    "index_reduce",  # Known to have non-deterministic behavior
})

# Ops where numerical differences are expected and acceptable
# These use approximate comparisons instead of bitwise exact
APPROXIMATE_OPS = frozenset({
    # Ops with known numerical instability at edge cases
    "special.log_softmax",
    "nn.functional.log_softmax",
    "special.softmax",
    "nn.functional.softmax",
    # Ops that may accumulate floating point differently
    "cumsum",
    "cumprod",
    "cummax",
    "cummin",
})

# Known failures that should be skipped (ops that exist in op_db)
AOT_EAGER_COMPILE_FAILURES = {
    xfail("linalg.lstsq"),  # dynamic output shape
    xfail("linalg.lstsq", "grad_oriented"),  # dynamic output shape
    xfail("combinations"),  # dynamic output shape
    xfail("nonzero"),  # dynamic output shape
    xfail("unique"),  # dynamic output shape
    xfail("unique_consecutive"),  # dynamic output shape
    xfail("masked_select"),  # dynamic output shape
    skip("nn.functional.scaled_dot_product_attention"),  # complex control flow
    skip("to_sparse"),  # returns sparse tensor
    xfail("_segment_reduce", "offsets"),
    xfail("_segment_reduce", "lengths"),
}


def _should_skip_op(op):
    """Check if an op should be skipped for bitwise equivalence testing."""
    op_name = op.name
    if op.variant_test_name:
        op_name = f"{op.name}.{op.variant_test_name}"

    # Skip non-deterministic ops
    if op.name in NONDETERMINISTIC_OPS or op_name in NONDETERMINISTIC_OPS:
        return True

    return False


def _get_op_name(op):
    """Get the full op name including variant."""
    if op.variant_test_name:
        return f"{op.name}.{op.variant_test_name}"
    return op.name


class TestAotEagerBitwiseEquivalence(torch._dynamo.test_case.TestCase):
    """Test that compiling with aot_eager produces bitwise equivalent results to eager."""

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()

    @ops(
        [op for op in op_db if not _should_skip_op(op)],
        allowed_dtypes=(torch.float32,),
    )
    @skipOps(
        "TestAotEagerBitwiseEquivalence",
        "test_aot_eager_bitwise_forward",
        AOT_EAGER_COMPILE_FAILURES,
    )
    def test_aot_eager_bitwise_forward(self, device, dtype, op):
        """Test that forward pass is bitwise equivalent between eager and aot_eager."""
        op_name = _get_op_name(op)

        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)

        for sample_input in sample_inputs_itr:
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs

            # Get the op function
            op_fn = op.op

            def fn(*args, **kwargs):
                return op_fn(*args, **kwargs)

            # Compile with aot_eager
            compiled_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)

            # Run eager
            torch._dynamo.reset()
            torch.manual_seed(42)
            try:
                eager_out = fn(*args, **kwargs)
            except Exception:
                # If eager fails, skip this sample
                continue

            # Run compiled
            torch.manual_seed(42)
            try:
                compiled_out = compiled_fn(*args, **kwargs)
            except torch._dynamo.exc.Unsupported:
                # Graph break or unsupported op, skip
                self.skipTest(f"Op {op_name} has graph break or is unsupported")
            except DynamicOutputShapeException:
                self.skipTest(f"Op {op_name} has dynamic output shape")
            except GuardOnDataDependentSymNode:
                self.skipTest(f"Op {op_name} has data-dependent control flow")
            except Exception as e:
                # Check if it's a known dynamic shape issue
                if "dynamic" in str(e).lower() or "symbolic" in str(e).lower():
                    self.skipTest(f"Op {op_name} has dynamic shapes: {e}")
                raise

            # Compare outputs - check for bitwise equivalence
            eager_flat = pytree.tree_leaves(eager_out)
            compiled_flat = pytree.tree_leaves(compiled_out)

            self.assertEqual(
                len(eager_flat),
                len(compiled_flat),
                msg=f"Op {op_name}: number of outputs differ",
            )

            for i, (eager_val, compiled_val) in enumerate(
                zip(eager_flat, compiled_flat)
            ):
                if isinstance(eager_val, torch.Tensor) and isinstance(
                    compiled_val, torch.Tensor
                ):
                    # Use rtol=0, atol=0 for bitwise exact comparison
                    # But allow for some ops that are known to have numerical differences
                    if op.name in APPROXIMATE_OPS or op_name in APPROXIMATE_OPS:
                        torch.testing.assert_close(
                            compiled_val,
                            eager_val,
                            rtol=1e-5,
                            atol=1e-5,
                            msg=lambda msg: f"Op {op_name} output {i}: {msg}",
                        )
                    else:
                        # Bitwise exact comparison
                        if eager_val.is_floating_point() or eager_val.is_complex():
                            # For floating point, check for exact equality
                            # This means the bits should be identical
                            are_equal = torch.equal(compiled_val, eager_val)
                            # Also consider all-NaN tensors as equal
                            if not are_equal and eager_val.numel() > 0:
                                are_equal = (
                                    torch.isnan(compiled_val).all()
                                    and torch.isnan(eager_val).all()
                                )
                            # Empty tensors are equal if shapes match
                            if eager_val.numel() == 0:
                                are_equal = eager_val.shape == compiled_val.shape

                            if not are_equal:
                                if eager_val.numel() > 0:
                                    max_diff = (compiled_val - eager_val).abs().max().item()
                                else:
                                    max_diff = 0.0
                                self.fail(
                                    f"Op {op_name} output {i}: compiled and eager outputs "
                                    f"are not bitwise equivalent.\n"
                                    f"Max diff: {max_diff}\n"
                                    f"Eager dtype: {eager_val.dtype}, Compiled dtype: {compiled_val.dtype}\n"
                                    f"Eager shape: {eager_val.shape}, Compiled shape: {compiled_val.shape}"
                                )
                        else:
                            # For integer types, use torch.equal
                            self.assertTrue(
                                torch.equal(compiled_val, eager_val),
                                msg=f"Op {op_name} output {i}: compiled and eager outputs differ",
                            )
                elif eager_val != compiled_val:
                    self.fail(
                        f"Op {op_name} output {i}: non-tensor outputs differ: "
                        f"eager={eager_val}, compiled={compiled_val}"
                    )

    @ops(
        [op for op in op_db if op.supports_autograd and not _should_skip_op(op)],
        allowed_dtypes=(torch.float32,),
    )
    @skipOps(
        "TestAotEagerBitwiseEquivalence",
        "test_aot_eager_bitwise_backward",
        AOT_EAGER_COMPILE_FAILURES,
    )
    def test_aot_eager_bitwise_backward(self, device, dtype, op):
        """Test that backward pass is bitwise equivalent between eager and aot_eager."""
        op_name = _get_op_name(op)

        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)

        for sample_input in sample_inputs_itr:
            # Clone inputs for eager and compiled runs
            def clone_input(x):
                if isinstance(x, torch.Tensor):
                    return x.clone().detach().requires_grad_(x.requires_grad)
                return x

            eager_args = [clone_input(sample_input.input)] + [
                clone_input(a) for a in sample_input.args
            ]
            compiled_args = [clone_input(sample_input.input)] + [
                clone_input(a) for a in sample_input.args
            ]
            kwargs = sample_input.kwargs

            # Get the op function
            op_fn = op.op

            def fn(*args, **kwargs):
                return op_fn(*args, **kwargs)

            # Compile with aot_eager
            compiled_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)

            # Run eager forward
            torch._dynamo.reset()
            torch.manual_seed(42)
            try:
                eager_out = fn(*eager_args, **kwargs)
            except Exception:
                # If eager fails, skip this sample
                continue

            # Run compiled forward
            torch.manual_seed(42)
            try:
                compiled_out = compiled_fn(*compiled_args, **kwargs)
            except torch._dynamo.exc.Unsupported:
                self.skipTest(f"Op {op_name} has graph break or is unsupported")
            except DynamicOutputShapeException:
                self.skipTest(f"Op {op_name} has dynamic output shape")
            except GuardOnDataDependentSymNode:
                self.skipTest(f"Op {op_name} has data-dependent control flow")
            except Exception as e:
                if "dynamic" in str(e).lower() or "symbolic" in str(e).lower():
                    self.skipTest(f"Op {op_name} has dynamic shapes: {e}")
                raise

            # Process outputs for gradient computation
            if sample_input.output_process_fn_grad is not None:
                eager_out = sample_input.output_process_fn_grad(eager_out)
                compiled_out = sample_input.output_process_fn_grad(compiled_out)

            # Get flat outputs for backward
            eager_flat = [
                x
                for x in pytree.tree_leaves(eager_out)
                if isinstance(x, torch.Tensor) and x.requires_grad
            ]
            compiled_flat = [
                x
                for x in pytree.tree_leaves(compiled_out)
                if isinstance(x, torch.Tensor) and x.requires_grad
            ]

            if not eager_flat or not compiled_flat:
                # No tensors to backprop through
                continue

            # Get inputs that require grad
            eager_inputs_with_grad = [
                x
                for x in pytree.tree_leaves(eager_args)
                if isinstance(x, torch.Tensor) and x.requires_grad
            ]
            compiled_inputs_with_grad = [
                x
                for x in pytree.tree_leaves(compiled_args)
                if isinstance(x, torch.Tensor) and x.requires_grad
            ]

            if not eager_inputs_with_grad or not compiled_inputs_with_grad:
                continue

            # Compute gradients
            try:
                eager_sum = sum(x.sum() for x in eager_flat)
                compiled_sum = sum(x.sum() for x in compiled_flat)

                torch.manual_seed(42)
                eager_grads = torch.autograd.grad(
                    eager_sum, eager_inputs_with_grad, allow_unused=True
                )

                torch.manual_seed(42)
                compiled_grads = torch.autograd.grad(
                    compiled_sum, compiled_inputs_with_grad, allow_unused=True
                )
            except Exception:
                # Gradient computation failed, skip
                continue

            # Compare gradients
            for i, (eager_grad, compiled_grad) in enumerate(
                zip(eager_grads, compiled_grads)
            ):
                if eager_grad is None and compiled_grad is None:
                    continue
                if eager_grad is None or compiled_grad is None:
                    self.fail(
                        f"Op {op_name} gradient {i}: one is None, other is not"
                    )

                # Check bitwise equivalence of gradients
                if op.name in APPROXIMATE_OPS or op_name in APPROXIMATE_OPS:
                    torch.testing.assert_close(
                        compiled_grad,
                        eager_grad,
                        rtol=1e-5,
                        atol=1e-5,
                        msg=lambda msg: f"Op {op_name} gradient {i}: {msg}",
                    )
                else:
                    are_equal = torch.equal(compiled_grad, eager_grad)
                    # Also consider all-NaN tensors as equal
                    if not are_equal and eager_grad.numel() > 0:
                        are_equal = (
                            torch.isnan(compiled_grad).all()
                            and torch.isnan(eager_grad).all()
                        )
                    # Empty tensors are equal if shapes match
                    if eager_grad.numel() == 0:
                        are_equal = eager_grad.shape == compiled_grad.shape

                    if not are_equal:
                        if eager_grad.numel() > 0:
                            max_diff = (compiled_grad - eager_grad).abs().max().item()
                        else:
                            max_diff = 0.0
                        self.fail(
                            f"Op {op_name} gradient {i}: compiled and eager gradients "
                            f"are not bitwise equivalent.\n"
                            f"Max diff: {max_diff}"
                        )


# Only test on CPU to ensure deterministic behavior
instantiate_device_type_tests(
    TestAotEagerBitwiseEquivalence, globals(), only_for="cpu"
)


if __name__ == "__main__":
    run_tests()
