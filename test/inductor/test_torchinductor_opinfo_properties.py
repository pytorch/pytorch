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

Focuses on ops commonly used in LLMs from unary_ufuncs, binary_ufuncs, and op_db.
"""

import unittest

import pytest

import torch
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_inductor_cache
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
    ops,
)
from torch.testing._internal.common_methods_invocations import (
    binary_ufuncs,
    op_db,
    unary_ufuncs,
)
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    parametrize,
    skipIfTorchDynamo,
    TEST_WITH_ASAN,
)
from torch.testing._internal.inductor_utils import HAS_GPU


# LLM-useful op names to filter from unary_ufuncs
LLM_UNARY_OP_NAMES = {
    # Basic math
    "abs",
    "neg",
    "reciprocal",
    # Exponential/log
    "exp",
    "exp2",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    # Power/root
    "sqrt",
    "rsqrt",
    # Trigonometric
    "sin",
    "cos",
    "tan",
    "tanh",
    # Activations
    "sigmoid",
    "nn.functional.relu6",
}

# LLM-useful op names to filter from binary_ufuncs
LLM_BINARY_OP_NAMES = {
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "remainder",
    "fmod",
    "maximum",
    "minimum",
}

# LLM-useful op names from op_db (nn.functional and others)
LLM_OP_DB_NAMES = {
    # Activations
    "nn.functional.gelu",
    "nn.functional.silu",
    "nn.functional.leaky_relu",
    "nn.functional.hardswish",
    # Normalization
    "nn.functional.layer_norm",
    "nn.functional.rms_norm",
    # Attention/linear
    "nn.functional.linear",
    "matmul",
    "bmm",
    # Softmax
    "softmax",
    "log_softmax",
}

# Filter OpInfos for LLM-useful ops
llm_unary_ops = [op for op in unary_ufuncs if op.name in LLM_UNARY_OP_NAMES]
llm_binary_ops = [op for op in binary_ufuncs if op.name in LLM_BINARY_OP_NAMES]
llm_op_db_ops = [op for op in op_db if op.name in LLM_OP_DB_NAMES]

# Combine all ops, avoiding duplicates by name
_seen_names = set()
llm_ops = []
for op in llm_unary_ops + llm_binary_ops + llm_op_db_ops:
    if op.name not in _seen_names:
        _seen_names.add(op.name)
        llm_ops.append(op)

# Backends to test
BACKENDS = [
    "aot_eager_decomp_partition",
    "inductor_default",
    "inductor_numerics",
]

# Dtypes to test - common LLM dtypes
DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
]

# Number of samples for numerical testing
NUM_SAMPLES = 65536


def generate_exhaustive_16bit(dtype, device):
    """Generate all 65536 possible values for a 16-bit float dtype.

    For fp16 and bf16, there are exactly 2^16 = 65536 possible bit patterns.
    This allows exhaustive testing of unary functions.
    """
    # Generate all 16-bit patterns as uint16
    all_patterns = torch.arange(0, 65536, dtype=torch.int32, device=device)
    # View as the target dtype
    if dtype == torch.float16:
        return all_patterns.to(torch.int16).view(torch.float16)
    elif dtype == torch.bfloat16:
        return all_patterns.to(torch.int16).view(torch.bfloat16)
    else:
        raise ValueError(f"Unsupported dtype for exhaustive 16-bit testing: {dtype}")


def generate_sampled_fp32(num_samples, device, seed=42):
    """Generate uniformly sampled fp32 bit patterns.

    Samples random 32-bit patterns and interprets them as fp32.
    This provides good coverage of the fp32 space including denormals,
    infinities, and NaNs.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    # Generate random 32-bit patterns
    patterns = torch.randint(
        0, 2**31, (num_samples,), dtype=torch.int32, device=device, generator=gen
    )
    # Also include negative patterns (high bit set)
    signs = torch.randint(
        0, 2, (num_samples,), dtype=torch.int32, device=device, generator=gen
    )
    patterns = patterns | (signs << 31)
    return patterns.view(torch.float32)


def generate_sampled_pairs(num_samples, dtype, device, seed=42):
    """Generate pairs of sampled values for binary function testing.

    For fp32: uses random bit patterns
    For fp16/bf16: samples from all possible values
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    if dtype == torch.float32:
        # Random bit patterns for fp32
        patterns1 = torch.randint(
            0, 2**31, (num_samples,), dtype=torch.int32, device=device, generator=gen
        )
        signs1 = torch.randint(
            0, 2, (num_samples,), dtype=torch.int32, device=device, generator=gen
        )
        patterns1 = patterns1 | (signs1 << 31)
        x = patterns1.view(torch.float32)

        patterns2 = torch.randint(
            0, 2**31, (num_samples,), dtype=torch.int32, device=device, generator=gen
        )
        signs2 = torch.randint(
            0, 2, (num_samples,), dtype=torch.int32, device=device, generator=gen
        )
        patterns2 = patterns2 | (signs2 << 31)
        y = patterns2.view(torch.float32)
    else:
        # Sample indices into the 16-bit space
        indices1 = torch.randint(
            0, 65536, (num_samples,), dtype=torch.int32, device=device, generator=gen
        )
        indices2 = torch.randint(
            0, 65536, (num_samples,), dtype=torch.int32, device=device, generator=gen
        )

        if dtype == torch.float16:
            x = indices1.to(torch.int16).view(torch.float16)
            y = indices2.to(torch.int16).view(torch.float16)
        elif dtype == torch.bfloat16:
            x = indices1.to(torch.int16).view(torch.bfloat16)
            y = indices2.to(torch.int16).view(torch.bfloat16)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    return x, y


# Inductor options for strict numerics matching eager behavior
INDUCTOR_NUMERICS_OPTIONS = {
    "deterministic": True,
    "fallback_random": True,
    "emulate_precision_casts": True,
    # Note: config key has typo "divison" (missing 'i')
    "emulate_divison_rounding": True,
}


def _slice_tensor_or_collection(value, batch_size, original_batch_size, input_ndim):
    """Recursively slice tensors in a value (tensor, list, tuple, or other).

    Only slices tensors that have the same number of dimensions as the input
    to preserve broadcasting patterns.
    """
    if isinstance(value, torch.Tensor):
        if (
            value.dim() == input_ndim  # Same ndim to preserve broadcast pattern
            and value.dim() > 0
            and value.shape[0] == original_batch_size
            and value.shape[0] > 1  # Don't slice broadcast dimensions
        ):
            return value[:batch_size]
        return value
    elif isinstance(value, (list, tuple)):
        sliced = [
            _slice_tensor_or_collection(v, batch_size, original_batch_size, input_ndim)
            for v in value
        ]
        return type(value)(sliced)
    else:
        return value


def slice_tensors_to_batch_size(sample_input, batch_size):
    """Slice all tensors in a SampleInput to the given batch size.

    For batch invariance testing, we need to slice all tensor inputs that have
    matching batch dimensions. This function slices:
    - sample_input.input if it's a tensor with dim > 0
    - All tensor args that have the same size in dim 0 as the input (not broadcast dims)
    - All tensor kwargs that have the same size in dim 0 as the input (not broadcast dims)

    Tensors with size 1 in dim 0 are broadcast dimensions and should not be sliced.
    Also handles nested lists/tuples of tensors (e.g., TensorList).
    Only slices tensors with the same ndim as the input to preserve broadcast patterns.

    Returns a tuple (sliced_input, sliced_args, sliced_kwargs), or None if slicing is not possible.
    """
    inp = sample_input.input
    if not isinstance(inp, torch.Tensor) or inp.dim() == 0:
        return None

    original_batch_size = inp.shape[0]
    input_ndim = inp.dim()
    if batch_size > original_batch_size:
        return None

    # Slice the input
    sliced_input = inp[:batch_size]

    # Slice args - handles tensors and lists/tuples of tensors
    sliced_args = tuple(
        _slice_tensor_or_collection(arg, batch_size, original_batch_size, input_ndim)
        for arg in sample_input.args
    )

    # Slice kwargs - handles tensors and lists/tuples of tensors
    sliced_kwargs = {
        key: _slice_tensor_or_collection(val, batch_size, original_batch_size, input_ndim)
        for key, val in sample_input.kwargs.items()
    }

    return sliced_input, sliced_args, sliced_kwargs


def sample_operates_on_batch_dim(op_name, sample_input):
    """Check if a sample input operates on the batch dimension (dim 0).

    For ops that normalize/reduce over a dimension, if that dimension is 0,
    slicing the batch will change the result and batch invariance doesn't apply.
    """
    # Ops that take a 'dim' argument and normalize/reduce over it
    dim_based_ops = {
        "softmax",
        "log_softmax",
        "nn.functional.softmax",
        "nn.functional.log_softmax",
    }

    if op_name not in dim_based_ops:
        return False

    # Get dim from args or kwargs
    dim = None
    if sample_input.args:
        dim = sample_input.args[0]
    if "dim" in sample_input.kwargs:
        dim = sample_input.kwargs["dim"]

    # If dim is 0 or -ndim (equivalent to 0), the op operates on the batch dimension
    if dim is not None:
        inp = sample_input.input
        if isinstance(inp, torch.Tensor):
            ndim = inp.dim()
            # Normalize negative dim
            if dim < 0:
                dim = dim + ndim
            return dim == 0

    return False


# Expected failures for bitwise equivalence tests.
# Maps (device_type, op_name, backend, test_type, dtype) -> reason for expected failure.
# test_type is one of: "batch_invariance", "determinism", "eager_equivalence",
#                      "unary_numerical", "binary_numerical"
# dtype can be None to match all dtypes, or a specific torch.dtype
#
# These track known numerical differences between eager and compiled execution.
# The goal is to eventually fix these and remove entries from this dict.
EXPECTED_FAILURES = {
    # =========================================================================
    # Eager equivalence failures
    # =========================================================================
    # div has numerical differences on CUDA due to Triton's division implementation
    ("cuda", "div", "inductor_default", "eager_equivalence", None): "div has numerical differences on CUDA",
    # reciprocal has numerical differences on CUDA
    ("cuda", "reciprocal", "inductor_default", "eager_equivalence", None): "reciprocal has numerical differences on CUDA",
    ("cuda", "reciprocal", "inductor_numerics", "eager_equivalence", None): "reciprocal has numerical differences on CUDA",
    # sigmoid has numerical differences on CUDA
    ("cuda", "sigmoid", "inductor_default", "eager_equivalence", None): "sigmoid has numerical differences on CUDA",
    ("cuda", "sigmoid", "inductor_numerics", "eager_equivalence", None): "sigmoid has numerical differences on CUDA",
    # gelu has numerical differences on CUDA
    ("cuda", "nn.functional.gelu", "inductor_default", "eager_equivalence", None): "gelu has numerical differences on CUDA",
    ("cuda", "nn.functional.gelu", "inductor_numerics", "eager_equivalence", None): "gelu has numerical differences on CUDA",
    ("cuda", "nn.functional.gelu", "aot_eager_decomp_partition", "eager_equivalence", torch.float32): "gelu decomposition has numerical differences",
    # rms_norm decomposition has numerical differences on CUDA
    ("cuda", "nn.functional.rms_norm", "aot_eager_decomp_partition", "eager_equivalence", None): "rms_norm decomposition has numerical differences",
    # softmax/log_softmax have numerical differences
    ("cuda", "softmax", "aot_eager_decomp_partition", "eager_equivalence", torch.float32): "softmax has numerical differences",
    ("cuda", "softmax", "inductor_default", "eager_equivalence", torch.float32): "softmax has numerical differences",
    ("cuda", "softmax", "inductor_numerics", "eager_equivalence", torch.float32): "softmax has numerical differences",
    ("cuda", "log_softmax", "aot_eager_decomp_partition", "eager_equivalence", torch.float32): "log_softmax has numerical differences",
    ("cuda", "log_softmax", "inductor_default", "eager_equivalence", torch.float32): "log_softmax has numerical differences",
    ("cuda", "log_softmax", "inductor_numerics", "eager_equivalence", torch.float32): "log_softmax has numerical differences",
    # matmul has numerical differences due to FP associativity
    ("cuda", "matmul", "aot_eager_decomp_partition", "eager_equivalence", torch.float32): "matmul has numerical differences",
    ("cuda", "matmul", "inductor_default", "eager_equivalence", torch.float32): "matmul has numerical differences",
    ("cuda", "matmul", "inductor_numerics", "eager_equivalence", torch.float32): "matmul has numerical differences",
    # layer_norm has numerical differences
    ("cuda", "nn.functional.layer_norm", "aot_eager_decomp_partition", "eager_equivalence", torch.float32): "layer_norm has numerical differences",
    ("cuda", "nn.functional.layer_norm", "inductor_default", "eager_equivalence", torch.float32): "layer_norm has numerical differences",
    ("cuda", "nn.functional.layer_norm", "inductor_numerics", "eager_equivalence", torch.float32): "layer_norm has numerical differences",
    # silu has numerical differences
    ("cuda", "nn.functional.silu", "aot_eager_decomp_partition", "eager_equivalence", torch.float16): "silu has numerical differences",
    ("cuda", "nn.functional.silu", "aot_eager_decomp_partition", "eager_equivalence", torch.float32): "silu has numerical differences",
    ("cuda", "nn.functional.silu", "inductor_default", "eager_equivalence", torch.float16): "silu has numerical differences",
    ("cuda", "nn.functional.silu", "inductor_default", "eager_equivalence", torch.float32): "silu has numerical differences",
    ("cuda", "nn.functional.silu", "inductor_numerics", "eager_equivalence", torch.float16): "silu has numerical differences",
    ("cuda", "nn.functional.silu", "inductor_numerics", "eager_equivalence", torch.float32): "silu has numerical differences",
    # pow has numerical differences
    ("cuda", "pow", "inductor_default", "eager_equivalence", torch.float32): "pow has numerical differences",
    ("cuda", "pow", "inductor_numerics", "eager_equivalence", torch.float32): "pow has numerical differences",
    # =========================================================================
    # Determinism failures
    # =========================================================================
    ("cuda", "matmul", "inductor_default", "determinism", torch.float32): "matmul has non-deterministic behavior",
    # =========================================================================
    # Batch invariance failures
    # =========================================================================
    # matmul is not batch-invariant due to floating-point associativity
    ("cuda", "matmul", "aot_eager_decomp_partition", "batch_invariance", None): "matmul has numerical differences across batch sizes",
    ("cuda", "matmul", "inductor_default", "batch_invariance", None): "matmul has numerical differences across batch sizes",
    ("cuda", "matmul", "inductor_numerics", "batch_invariance", None): "matmul has numerical differences across batch sizes",
    # nn.functional.linear is not batch-invariant (uses matmul internally)
    ("cuda", "nn.functional.linear", "aot_eager_decomp_partition", "batch_invariance", None): "linear has numerical differences across batch sizes",
    ("cuda", "nn.functional.linear", "inductor_default", "batch_invariance", None): "linear has numerical differences across batch sizes",
    ("cuda", "nn.functional.linear", "inductor_numerics", "batch_invariance", None): "linear has numerical differences across batch sizes",
    # div has batch invariance issues
    ("cuda", "div", "inductor_default", "batch_invariance", torch.float32): "div has numerical differences across batch sizes",
    # pow has batch invariance issues
    ("cuda", "pow", "inductor_default", "batch_invariance", torch.float32): "pow has numerical differences across batch sizes",
    ("cuda", "pow", "inductor_numerics", "batch_invariance", torch.float32): "pow has numerical differences across batch sizes",
    # =========================================================================
    # Unary numerical (exhaustive/sampled) failures
    # =========================================================================
    ("cuda", "exp2", "inductor_default", "unary_numerical", torch.bfloat16): "exp2 has numerical differences",
    ("cuda", "exp2", "inductor_default", "unary_numerical", torch.float32): "exp2 has numerical differences",
    ("cuda", "exp2", "inductor_numerics", "unary_numerical", torch.bfloat16): "exp2 has numerical differences",
    ("cuda", "exp2", "inductor_numerics", "unary_numerical", torch.float32): "exp2 has numerical differences",
    ("cuda", "expm1", "inductor_default", "unary_numerical", torch.bfloat16): "expm1 has numerical differences",
    ("cuda", "expm1", "inductor_default", "unary_numerical", torch.float32): "expm1 has numerical differences",
    ("cuda", "expm1", "inductor_numerics", "unary_numerical", torch.bfloat16): "expm1 has numerical differences",
    ("cuda", "expm1", "inductor_numerics", "unary_numerical", torch.float32): "expm1 has numerical differences",
    ("cuda", "log1p", "inductor_default", "unary_numerical", torch.float32): "log1p has numerical differences",
    ("cuda", "log1p", "inductor_numerics", "unary_numerical", torch.float32): "log1p has numerical differences",
    ("cuda", "reciprocal", "inductor_default", "unary_numerical", torch.float32): "reciprocal has numerical differences",
    ("cuda", "reciprocal", "inductor_numerics", "unary_numerical", torch.float32): "reciprocal has numerical differences",
    ("cuda", "rsqrt", "inductor_default", "unary_numerical", torch.bfloat16): "rsqrt has numerical differences",
    ("cuda", "rsqrt", "inductor_default", "unary_numerical", torch.float32): "rsqrt has numerical differences",
    ("cuda", "rsqrt", "inductor_numerics", "unary_numerical", torch.bfloat16): "rsqrt has numerical differences",
    ("cuda", "rsqrt", "inductor_numerics", "unary_numerical", torch.float32): "rsqrt has numerical differences",
    ("cuda", "sigmoid", "inductor_default", "unary_numerical", torch.float32): "sigmoid has numerical differences",
    ("cuda", "sigmoid", "inductor_numerics", "unary_numerical", torch.float32): "sigmoid has numerical differences",
    ("cuda", "sin", "inductor_default", "unary_numerical", torch.float32): "sin has numerical differences",
    ("cuda", "sin", "inductor_numerics", "unary_numerical", torch.float32): "sin has numerical differences",
    ("cuda", "tan", "inductor_default", "unary_numerical", torch.bfloat16): "tan has numerical differences",
    ("cuda", "tan", "inductor_default", "unary_numerical", torch.float32): "tan has numerical differences",
    ("cuda", "tan", "inductor_numerics", "unary_numerical", torch.bfloat16): "tan has numerical differences",
    ("cuda", "tan", "inductor_numerics", "unary_numerical", torch.float32): "tan has numerical differences",
    ("cuda", "tanh", "inductor_default", "unary_numerical", torch.float32): "tanh has numerical differences",
    ("cuda", "tanh", "inductor_numerics", "unary_numerical", torch.float32): "tanh has numerical differences",
    # =========================================================================
    # Binary numerical (sampled) failures
    # =========================================================================
    ("cuda", "div", "inductor_default", "binary_numerical", torch.float16): "div has numerical differences",
    ("cuda", "div", "inductor_default", "binary_numerical", torch.float32): "div has numerical differences",
    ("cuda", "fmod", "inductor_default", "binary_numerical", torch.bfloat16): "fmod has numerical differences",
    ("cuda", "fmod", "inductor_default", "binary_numerical", torch.float32): "fmod has numerical differences",
    ("cuda", "fmod", "inductor_numerics", "binary_numerical", torch.bfloat16): "fmod has numerical differences",
    ("cuda", "fmod", "inductor_numerics", "binary_numerical", torch.float32): "fmod has numerical differences",
    ("cuda", "pow", "inductor_default", "binary_numerical", None): "pow has numerical differences",
    ("cuda", "pow", "inductor_numerics", "binary_numerical", None): "pow has numerical differences",
    ("cuda", "remainder", "inductor_default", "binary_numerical", None): "remainder has numerical differences",
    ("cuda", "remainder", "inductor_numerics", "binary_numerical", None): "remainder has numerical differences",
}


def is_expected_failure(device_type, op_name, backend, test_type, dtype=None):
    """Check if a test is expected to fail.

    First checks for dtype-specific failure, then falls back to dtype=None (all dtypes).
    """
    # Check dtype-specific failure first
    if (device_type, op_name, backend, test_type, dtype) in EXPECTED_FAILURES:
        return True
    # Fall back to dtype=None (matches all dtypes)
    return (device_type, op_name, backend, test_type, None) in EXPECTED_FAILURES


def get_expected_failure_reason(device_type, op_name, backend, test_type, dtype=None):
    """Get the reason for an expected failure."""
    # Check dtype-specific failure first
    key = (device_type, op_name, backend, test_type, dtype)
    if key in EXPECTED_FAILURES:
        return EXPECTED_FAILURES[key]
    # Fall back to dtype=None
    key = (device_type, op_name, backend, test_type, None)
    return EXPECTED_FAILURES.get(key, "Unknown")


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
@unittest.skipIf(not HAS_GPU, "Requires GPU")
class TestOpInfoProperties(TestCase):
    """Test op properties under various inductor modes using OpInfo on CUDA."""

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()

    def _get_sample_inputs(self, op, device, dtype):
        """Get sample inputs from OpInfo using reference_inputs for comprehensive coverage."""
        # Use reference_inputs for more comprehensive test coverage
        # Falls back to sample_inputs if reference_inputs_func is not defined
        try:
            samples = list(op.reference_inputs(device, dtype, requires_grad=False))
        except Exception:
            samples = list(op.sample_inputs(device, dtype, requires_grad=False))
        return samples

    # =========================================================================
    # Batch Invariance Tests
    # =========================================================================

    @onlyCUDA
    @skipIfTorchDynamo("Test uses dynamo already")
    @ops(llm_ops, allowed_dtypes=DTYPES)
    @parametrize("backend", BACKENDS)
    def test_batch_invariance(self, device, dtype, op, backend):
        """Test batch invariance with exponentially decreasing batch sizes.

        For each sample input, this test:
        1. Runs the compiled op at full batch size
        2. Runs at size/2, size/4, etc. (exponentially decreasing)
        3. Verifies the sliced output matches the corresponding slice of the full output

        All tensor inputs with matching batch dimensions are sliced together.
        """
        torch._dynamo.reset()
        device_type = torch.device(device).type

        samples = self._get_sample_inputs(op, device, dtype)
        if not samples:
            self.skipTest(f"No samples for {op.name}")

        fn = op.get_op()
        tested_any = False

        for sample in samples:
            # Skip if input is not a tensor or is 0-dim
            if not isinstance(sample.input, torch.Tensor) or sample.input.dim() == 0:
                continue

            # Need at least size 4 in first dimension for meaningful test
            full_size = sample.input.shape[0]
            if full_size < 4:
                continue

            # Skip broadcast/expanded tensors (stride 0 in batch dim)
            # These don't have meaningful batch invariance since all rows are the same
            if sample.input.stride()[0] == 0:
                continue

            # Skip samples where the op normalizes/reduces over dim 0 (batch dimension)
            # because slicing the batch changes the normalization result
            if sample_operates_on_batch_dim(op.name, sample):
                continue

            compiled_fn = compile_fn(fn, backend)

            # Get reference output at full size
            full_args = (sample.input,) + tuple(sample.args)
            full_kwargs = sample.kwargs
            full_out = compiled_fn(*full_args, **full_kwargs)

            if not isinstance(full_out, torch.Tensor):
                continue

            # Skip if output is 0-dim (scalar) - can't slice it
            if full_out.dim() == 0:
                continue

            # Skip if output's first dimension doesn't match input's batch size
            # (e.g., due to broadcasting or reduction)
            if full_out.shape[0] != full_size:
                continue

            tested_any = True

            # Test with exponentially decreasing sizes: size, size/2, size/4, ...
            size = full_size
            try:
                while size >= 1:
                    # Slice all tensor inputs with matching batch dimensions
                    sliced = slice_tensors_to_batch_size(sample, size)
                    if sliced is None:
                        break
                    sliced_input, sliced_args, sliced_kwargs = sliced

                    out = compiled_fn(sliced_input, *sliced_args, **sliced_kwargs)

                    # Verify output matches the corresponding slice of full output (bitwise)
                    self.assertEqual(out, full_out[:size], rtol=0, atol=0)

                    # Step down exponentially
                    size = size // 2
            except (AssertionError, RuntimeError):
                if is_expected_failure(
                    device_type, op.name, backend, "batch_invariance", dtype
                ):
                    pytest.xfail(
                        f"Known failure: {get_expected_failure_reason(device_type, op.name, backend, 'batch_invariance', dtype)}"
                    )
                raise

        if not tested_any:
            self.skipTest("No suitable samples found for batch invariance test")

    # =========================================================================
    # Run-to-Run Determinism Tests
    # =========================================================================

    @onlyCUDA
    @skipIfTorchDynamo("Test uses dynamo already")
    @ops(llm_ops, allowed_dtypes=DTYPES)
    @parametrize("backend", BACKENDS)
    def test_determinism(self, device, dtype, op, backend):
        """Test run-to-run determinism with fresh cache and benchmark perturbation.

        For enhanced determinism testing, this test:
        1. Resets dynamo and uses fresh inductor cache between compilations
        2. Performs unrelated GPU work between runs to perturb benchmark timing
        3. Tests multiple runs and verifies bitwise identical outputs
        """
        torch._dynamo.reset()
        device_type = torch.device(device).type

        samples = self._get_sample_inputs(op, device, dtype)
        if not samples:
            self.skipTest(f"No samples for {op.name}")

        fn = op.get_op()

        def perturb_benchmarks(device):
            """Run unrelated GPU work to perturb benchmark timing heuristics."""
            # Vary the sizes to create different memory/compute patterns
            for size in [64, 256, 1024]:
                noise = torch.randn(size, size, device=device, dtype=torch.float32)
                _ = torch.matmul(noise, noise)
            torch.cuda.synchronize()

        for sample in samples:
            args = (sample.input,) + tuple(sample.args)
            kwargs = sample.kwargs

            # Run 1: Fresh cache
            torch._dynamo.reset()
            with fresh_inductor_cache():
                compiled_fn1 = compile_fn(fn, backend)
                out1 = compiled_fn1(*args, **kwargs)

            # Perturb benchmarks between runs
            perturb_benchmarks(device)

            # Run 2: Fresh cache again
            torch._dynamo.reset()
            with fresh_inductor_cache():
                compiled_fn2 = compile_fn(fn, backend)
                out2 = compiled_fn2(*args, **kwargs)

            # Perturb benchmarks again
            perturb_benchmarks(device)

            # Run 3: Fresh cache again
            torch._dynamo.reset()
            with fresh_inductor_cache():
                compiled_fn3 = compile_fn(fn, backend)
                out3 = compiled_fn3(*args, **kwargs)

            # Bitwise identical
            try:
                self.assertEqual(out1, out2, rtol=0, atol=0)
                self.assertEqual(out2, out3, rtol=0, atol=0)
            except (AssertionError, RuntimeError):
                if is_expected_failure(
                    device_type, op.name, backend, "determinism", dtype
                ):
                    pytest.xfail(
                        f"Known failure: {get_expected_failure_reason(device_type, op.name, backend, 'determinism', dtype)}"
                    )
                raise

    # =========================================================================
    # Bitwise Equivalence with Eager Mode Tests
    # =========================================================================

    @onlyCUDA
    @skipIfTorchDynamo("Test uses dynamo already")
    @ops(llm_ops, allowed_dtypes=DTYPES)
    @parametrize("backend", BACKENDS)
    def test_eager_equivalence(self, device, dtype, op, backend):
        """Test bitwise equivalence with eager execution."""
        torch._dynamo.reset()
        device_type = torch.device(device).type

        samples = self._get_sample_inputs(op, device, dtype)
        if not samples:
            self.skipTest(f"No samples for {op.name}")

        fn = op.get_op()

        for sample in samples:
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
            except (AssertionError, RuntimeError):
                if is_expected_failure(
                    device_type, op.name, backend, "eager_equivalence", dtype
                ):
                    pytest.xfail(
                        f"Known failure: {get_expected_failure_reason(device_type, op.name, backend, 'eager_equivalence', dtype)}"
                    )
                raise

    # =========================================================================
    # Exhaustive/Sampled Unary Ufunc Tests
    # =========================================================================

    @onlyCUDA
    @skipIfTorchDynamo("Test uses dynamo already")
    @ops(llm_unary_ops, allowed_dtypes=DTYPES)
    @parametrize("backend", BACKENDS)
    def test_unary_ufunc_numerical(self, device, dtype, op, backend):
        """Test unary ufuncs with exhaustive (16-bit) or sampled (fp32) inputs.

        For fp16 and bf16: exhaustively tests all 65536 possible bit patterns.
        For fp32: tests 64k sampled random bit patterns.

        Verifies bitwise equivalence between eager and compiled execution.
        """
        torch._dynamo.reset()
        device_type = torch.device(device).type

        fn = op.get_op()

        # Generate test values based on dtype
        if dtype in (torch.float16, torch.bfloat16):
            # Exhaustive: all 65536 possible values
            test_values = generate_exhaustive_16bit(dtype, device)
        else:
            # Sampled: 64k random bit patterns
            test_values = generate_sampled_fp32(NUM_SAMPLES, device)

        # Eager reference
        eager_out = fn(test_values)

        # Compiled output
        compiled_fn = compile_fn(fn, backend)
        compiled_out = compiled_fn(test_values)

        # Bitwise identical
        try:
            self.assertEqual(eager_out, compiled_out, rtol=0, atol=0)
        except (AssertionError, RuntimeError):
            if is_expected_failure(
                device_type, op.name, backend, "unary_numerical", dtype
            ):
                pytest.xfail(
                    f"Known failure: {get_expected_failure_reason(device_type, op.name, backend, 'unary_numerical', dtype)}"
                )
            raise

    # =========================================================================
    # Sampled Binary Ufunc Tests
    # =========================================================================

    @onlyCUDA
    @skipIfTorchDynamo("Test uses dynamo already")
    @ops(llm_binary_ops, allowed_dtypes=DTYPES)
    @parametrize("backend", BACKENDS)
    def test_binary_ufunc_numerical(self, device, dtype, op, backend):
        """Test binary ufuncs on 64k sampled value pairs.

        For fp32: samples random 32-bit patterns.
        For fp16/bf16: samples from all possible 16-bit values.

        Verifies bitwise equivalence between eager and compiled execution.
        """
        torch._dynamo.reset()
        device_type = torch.device(device).type

        fn = op.get_op()

        # Generate sampled pairs
        x, y = generate_sampled_pairs(NUM_SAMPLES, dtype, device)

        # Eager reference
        eager_out = fn(x, y)

        # Compiled output
        compiled_fn = compile_fn(fn, backend)
        compiled_out = compiled_fn(x, y)

        # Bitwise identical
        try:
            self.assertEqual(eager_out, compiled_out, rtol=0, atol=0)
        except (AssertionError, RuntimeError):
            if is_expected_failure(
                device_type, op.name, backend, "binary_numerical", dtype
            ):
                pytest.xfail(
                    f"Known failure: {get_expected_failure_reason(device_type, op.name, backend, 'binary_numerical', dtype)}"
                )
            raise


instantiate_device_type_tests(TestOpInfoProperties, globals(), except_for=["cpu"])

if __name__ == "__main__":
    run_tests()
