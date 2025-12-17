# Owner(s): ["module: inductor"]
"""
Tests for useful PyTorch ops under inductor with various compilation modes.

Tests three properties:
1. Batch invariance - output shouldn't change based on batch size
2. Run-to-run determinism - same input should give same output
3. Bitwise equivalence with torch eager mode

Tests three compilation backends:
1. aot_eager_decomp_partition - AOT autograd with eager execution
2. inductor_default - Standard inductor compilation
3. inductor_numerics - Inductor with strict numerics flags

Focuses on ops commonly used in LLMs from unary_ufuncs and binary_ufuncs.
"""

import unittest

import pytest
import torch
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyNativeDeviceTypes,
    ops,
    skipCPUIf,
)
from torch.testing._internal.common_methods_invocations import (
    binary_ufuncs,
    unary_ufuncs,
)
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    parametrize,
    skipIfTorchDynamo,
    TEST_WITH_ASAN,
)
from torch.testing._internal.inductor_utils import HAS_CPU


# Decorators explained:
#
# @onlyNativeDeviceTypes:
#   Skips test on non-native device types. Native devices are:
#   ('cpu', 'cuda', 'xpu', 'meta', 'mps', 'mtia', privateuse1)
#   This filters out any custom/third-party device backends that might be
#   registered but aren't part of core PyTorch.
#
# @skipCPUIf(condition, reason):
#   Skips the test on CPU device if the condition is True.
#   Used here as @skipCPUIf(not HAS_CPU, "Requires CPU compiler") to skip
#   CPU tests when no CPU compiler (like gcc/clang for C++ codegen) is available.
#
# @skipIfTorchDynamo(msg):
#   Skips the test when running under TorchDynamo (TEST_WITH_TORCHDYNAMO=1).
#   Used for tests that themselves use dynamo/compile, to avoid nested dynamo
#   tracing which can cause issues.


# LLM-useful op names to filter from unary_ufuncs
LLM_UNARY_OP_NAMES = {
    "exp",
    "log",
    "sigmoid",
    "tanh",
    "abs",
    "sqrt",
    "rsqrt",
    "neg",
    "reciprocal",
    "sin",
    "cos",
}

# LLM-useful op names to filter from binary_ufuncs
LLM_BINARY_OP_NAMES = {
    "add",
    "mul",
    "sub",
    "div",
    "pow",
}

# Filter OpInfos for LLM-useful ops
llm_unary_ops = [op for op in unary_ufuncs if op.name in LLM_UNARY_OP_NAMES]
llm_binary_ops = [op for op in binary_ufuncs if op.name in LLM_BINARY_OP_NAMES]
llm_ops = llm_unary_ops + llm_binary_ops

# Backends to test
BACKENDS = [
    "aot_eager_decomp_partition",
    "inductor_default",
    "inductor_numerics",
]

# Inductor options for strict numerics matching eager behavior
INDUCTOR_NUMERICS_OPTIONS = {
    "deterministic": True,
    "fallback_random": True,
    "emulate_precision_casts": True,
    # Note: config key has typo "divison" (missing 'i')
    "emulate_divison_rounding": True,
}

# Expected failures for bitwise equivalence tests.
# Maps (device_type, op_name, backend, test_type) -> reason for expected failure.
# test_type is one of: "batch_invariance", "determinism", "eager_equivalence"
#
# These track known numerical differences between eager and compiled execution.
# The goal is to eventually fix these and remove entries from this dict.
EXPECTED_FAILURES = {
    # === CPU failures ===
    # cos has small numerical differences in inductor for certain input values
    ("cpu", "cos", "inductor_default", "batch_invariance"): "cos has ~6e-8 numerical differences (flaky, input-dependent)",
    ("cpu", "cos", "inductor_numerics", "batch_invariance"): "cos has ~6e-8 numerical differences (flaky, input-dependent)",
    # sigmoid has numerical differences in inductor due to different
    # implementation (likely fused exp vs separate ops)
    ("cpu", "sigmoid", "inductor_default", "batch_invariance"): "sigmoid has ~1e-11 numerical differences at small batch sizes",
    ("cpu", "sigmoid", "inductor_default", "eager_equivalence"): "sigmoid has ~1e-7 numerical differences in inductor",
    ("cpu", "sigmoid", "inductor_numerics", "batch_invariance"): "sigmoid has ~1e-11 numerical differences at small batch sizes",
    ("cpu", "sigmoid", "inductor_numerics", "eager_equivalence"): "sigmoid has ~1e-7 numerical differences even with numerics flags",
    # === CUDA failures ===
    # div has numerical differences on CUDA due to Triton's division implementation
    ("cuda", "div", "inductor_default", "eager_equivalence"): "div has ~6e-8 numerical differences on CUDA",
    # reciprocal has numerical differences on CUDA
    ("cuda", "reciprocal", "inductor_default", "eager_equivalence"): "reciprocal has ~6e-8 numerical differences on CUDA",
    ("cuda", "reciprocal", "inductor_numerics", "eager_equivalence"): "reciprocal has ~6e-8 numerical differences on CUDA even with numerics flags",
    # sigmoid has numerical differences on CUDA
    ("cuda", "sigmoid", "inductor_default", "eager_equivalence"): "sigmoid has ~6e-8 numerical differences on CUDA",
    ("cuda", "sigmoid", "inductor_numerics", "eager_equivalence"): "sigmoid has ~6e-8 numerical differences on CUDA even with numerics flags",
}


def is_expected_failure(device_type, op_name, backend, test_type):
    """Check if a test is expected to fail."""
    return (device_type, op_name, backend, test_type) in EXPECTED_FAILURES


def get_expected_failure_reason(device_type, op_name, backend, test_type):
    """Get the reason for an expected failure."""
    return EXPECTED_FAILURES.get((device_type, op_name, backend, test_type), "Unknown")


def compile_fn(fn, backend):
    """Compile a function with the given backend."""
    if backend == "aot_eager_decomp_partition":
        return torch.compile(fn, backend="aot_eager_decomp_partition")
    elif backend == "inductor_default":
        return torch.compile(fn, backend="inductor")
    elif backend == "inductor_numerics":
        return torch.compile(fn, backend="inductor", options=INDUCTOR_NUMERICS_OPTIONS)
    else:
        raise ValueError(f"Unknown backend: {backend}")


@unittest.skipIf(IS_WINDOWS, "Skipped on Windows")
@unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
class TestOpInfoProperties(TestCase):
    """Test op properties under various inductor modes using OpInfo."""

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()

    def _get_sample_input(self, op, device, dtype):
        """Get one sample input from OpInfo."""
        samples = list(op.sample_inputs(device, dtype, requires_grad=False))
        if not samples:
            self.skipTest(f"No samples for {op.name}")
        return samples[0]

    # =========================================================================
    # Batch Invariance Tests
    # =========================================================================

    @onlyNativeDeviceTypes
    @skipCPUIf(not HAS_CPU, "Requires CPU compiler")
    @skipIfTorchDynamo("Test uses dynamo already")
    @ops(llm_ops, allowed_dtypes=(torch.float32,))
    @parametrize("backend", BACKENDS)
    def test_batch_invariance(self, device, dtype, op, backend):
        """Test batch invariance with exponentially decreasing batch sizes."""
        torch._dynamo.reset()
        device_type = torch.device(device).type

        sample = self._get_sample_input(op, device, dtype)
        fn = op.get_op()

        args = (sample.input,) + tuple(sample.args)
        kwargs = sample.kwargs

        # Skip if input is not a tensor or is 0-dim
        if not isinstance(sample.input, torch.Tensor) or sample.input.dim() == 0:
            self.skipTest("Needs tensor input with at least 1 dimension")

        # Need at least size 4 in first dimension for meaningful test
        if sample.input.shape[0] < 4:
            self.skipTest("Needs input with first dimension >= 4")

        compiled_fn = compile_fn(fn, backend)

        # Get reference output at full size
        full_out = compiled_fn(*args, **kwargs)

        if not isinstance(full_out, torch.Tensor):
            self.skipTest("Output is not a tensor")

        # Test with exponentially decreasing sizes: size, size/2, size/4, ...
        full_size = sample.input.shape[0]
        size = full_size
        try:
            while size >= 1:
                # Slice input to current size
                sliced_input = sample.input[:size]
                sliced_args = (sliced_input,) + tuple(sample.args)

                out = compiled_fn(*sliced_args, **kwargs)

                # Verify output matches the corresponding slice of full output (bitwise)
                self.assertEqual(out, full_out[:size], rtol=0, atol=0)

                # Step down exponentially
                size = size // 2
        except AssertionError:
            if is_expected_failure(device_type, op.name, backend, "batch_invariance"):
                pytest.xfail(f"Known failure: {get_expected_failure_reason(device_type, op.name, backend, 'batch_invariance')}")
            raise

    # =========================================================================
    # Run-to-Run Determinism Tests
    # =========================================================================

    @onlyNativeDeviceTypes
    @skipCPUIf(not HAS_CPU, "Requires CPU compiler")
    @skipIfTorchDynamo("Test uses dynamo already")
    @ops(llm_ops, allowed_dtypes=(torch.float32,))
    @parametrize("backend", BACKENDS)
    def test_determinism(self, device, dtype, op, backend):
        """Test run-to-run determinism."""
        torch._dynamo.reset()
        device_type = torch.device(device).type

        sample = self._get_sample_input(op, device, dtype)
        fn = op.get_op()

        args = (sample.input,) + tuple(sample.args)
        kwargs = sample.kwargs

        compiled_fn = compile_fn(fn, backend)

        out1 = compiled_fn(*args, **kwargs)
        out2 = compiled_fn(*args, **kwargs)
        out3 = compiled_fn(*args, **kwargs)

        # Bitwise identical
        try:
            self.assertEqual(out1, out2, rtol=0, atol=0)
            self.assertEqual(out2, out3, rtol=0, atol=0)
        except AssertionError:
            if is_expected_failure(device_type, op.name, backend, "determinism"):
                pytest.xfail(f"Known failure: {get_expected_failure_reason(device_type, op.name, backend, 'determinism')}")
            raise

    # =========================================================================
    # Bitwise Equivalence with Eager Mode Tests
    # =========================================================================

    @onlyNativeDeviceTypes
    @skipCPUIf(not HAS_CPU, "Requires CPU compiler")
    @skipIfTorchDynamo("Test uses dynamo already")
    @ops(llm_ops, allowed_dtypes=(torch.float32,))
    @parametrize("backend", BACKENDS)
    def test_eager_equivalence(self, device, dtype, op, backend):
        """Test bitwise equivalence with eager execution."""
        torch._dynamo.reset()
        device_type = torch.device(device).type

        sample = self._get_sample_input(op, device, dtype)
        fn = op.get_op()

        args = (sample.input,) + tuple(sample.args)
        kwargs = sample.kwargs

        # Eager reference
        eager_out = fn(*args, **kwargs)

        # Compiled output
        compiled_fn = compile_fn(fn, backend)
        compiled_out = compiled_fn(*args, **kwargs)

        # Bitwise identical
        try:
            self.assertEqual(eager_out, compiled_out, rtol=0, atol=0)
        except AssertionError:
            if is_expected_failure(device_type, op.name, backend, "eager_equivalence"):
                pytest.xfail(f"Known failure: {get_expected_failure_reason(device_type, op.name, backend, 'eager_equivalence')}")
            raise


instantiate_device_type_tests(TestOpInfoProperties, globals())

if __name__ == "__main__":
    run_tests()
