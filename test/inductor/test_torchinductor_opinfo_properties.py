# Owner(s): ["module: inductor"]
"""
OpInfo-based property tests for inductor numerical correctness.

Tests three properties:
1. Batch invariance - output shouldn't change based on batch size
2. Run-to-run determinism - same input should give same output across
   compilations, even with different autotuning choices
3. Bitwise equivalence with torch eager mode

Tests three compilation backends:
1. aot_eager_decomp_partition - AOT autograd with eager execution
2. inductor_default - Standard inductor compilation
3. inductor_numerics - Inductor with strict numerics flags

Focuses on ops commonly used in LLMs from unary_ufuncs, binary_ufuncs, and op_db.

Note: How to modify this test
-----------------------------
To add a new op:
    Add the op name to LLM_UNARY_OP_NAMES, LLM_BINARY_OP_NAMES, or LLM_OP_DB_NAMES
    depending on which OpInfo database contains the op.

To add an expected failure:
    Add to the appropriate *_XFAILS dict (e.g., EAGER_EQUIV_XFAILS):
        "backend": {"op_name": {dtype1, dtype2, ...}}
    Use ALL (None) to match all dtypes.

To remove a stale expected failure:
    Run the tests - any expected failure that now passes will fail with
    "XPASS" and tell you which entry to remove.

To run a specific test:
    pytest test/inductor/test_torchinductor_opinfo_properties.py -k "silu and eager"
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
    # Note: config key has typo: not "division" (missing 'i')
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
        key: _slice_tensor_or_collection(
            val, batch_size, original_batch_size, input_ndim
        )
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
# Structure: backend -> op_name -> set of failing dtypes (None means all dtypes)
# These track known numerical differences between eager and compiled execution.
# The goal is to eventually fix these and remove entries.

# Dtype shorthands
fp32, fp16, bf16, ALL = torch.float32, torch.float16, torch.bfloat16, None

EAGER_EQUIV_XFAILS = {
    "aot_eager_decomp_partition": {
        "nn.functional.gelu": {fp32},
        "nn.functional.layer_norm": {fp32},
        "nn.functional.rms_norm": {ALL},
        "softmax": {fp32},
        "log_softmax": {fp32},
        "matmul": {fp32},
    },
    "inductor_default": {
        "div": {ALL},
        "reciprocal": {ALL},
        "sigmoid": {ALL},
        "nn.functional.gelu": {ALL},
        "nn.functional.layer_norm": {fp32},
        "nn.functional.silu": {fp16, fp32},
        "softmax": {fp32},
        "log_softmax": {fp32},
        "matmul": {fp32},
        "pow": {fp32},
    },
    "inductor_numerics": {
        "reciprocal": {ALL},
        "sigmoid": {ALL},
        "nn.functional.gelu": {ALL},
        "nn.functional.layer_norm": {fp32},
        "softmax": {fp32},
        "log_softmax": {fp32},
        "matmul": {fp32},
        "pow": {fp32},
    },
}

DETERMINISM_XFAILS = {
    "inductor_default": {"matmul": {fp32}},
}

BATCH_INVARIANCE_XFAILS = {
    "aot_eager_decomp_partition": {
        "matmul": {ALL},
        "nn.functional.linear": {ALL},
    },
    "inductor_default": {
        "matmul": {ALL},
        "bmm": {fp32},
        "nn.functional.linear": {ALL},
        "div": {fp32},
        "pow": {fp32},
    },
    "inductor_numerics": {
        "matmul": {ALL},
        "bmm": {fp32},
        "nn.functional.linear": {ALL},
        "pow": {fp32},
    },
}

UNARY_NUMERICAL_XFAILS = {
    "inductor_default": {
        "exp2": {bf16, fp32},
        "expm1": {bf16, fp32},
        "log1p": {fp32},
        "reciprocal": {fp32},
        "rsqrt": {bf16, fp32},
        "sigmoid": {fp32},
        "sin": {fp32},
        "tan": {bf16, fp32},
        "tanh": {fp32},
    },
    "inductor_numerics": {
        "exp2": {bf16, fp32},
        "expm1": {bf16, fp32},
        "log1p": {fp32},
        "reciprocal": {fp32},
        "rsqrt": {bf16, fp32},
        "sigmoid": {fp32},
        "sin": {fp32},
        "tan": {bf16, fp32},
        "tanh": {fp32},
    },
}

BINARY_NUMERICAL_XFAILS = {
    "inductor_default": {
        "div": {fp16, fp32},
        "fmod": {bf16, fp32},
        "pow": {ALL},
        "remainder": {ALL},
    },
    "inductor_numerics": {
        "fmod": {bf16, fp32},
        "pow": {ALL},
        "remainder": {ALL},
    },
}

XFAIL_DICTS = {
    "eager_equivalence": EAGER_EQUIV_XFAILS,
    "determinism": DETERMINISM_XFAILS,
    "batch_invariance": BATCH_INVARIANCE_XFAILS,
    "unary_numerical": UNARY_NUMERICAL_XFAILS,
    "binary_numerical": BINARY_NUMERICAL_XFAILS,
}


def is_expected_failure(device_type, op_name, backend, test_type, dtype=None):
    """Check if a test is expected to fail."""
    xfails = XFAIL_DICTS.get(test_type, {}).get(backend, {}).get(op_name, set())
    return dtype in xfails or ALL in xfails


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

    def _run_with_expected_failure(
        self, device_type, op_name, backend, test_type, dtype, test_fn
    ):
        """Run a test with expected failure handling.

        Uses pytest.xfail for clear test output:
        - If test is expected to fail and does fail: XFAIL (pytest.xfail called)
        - If test is expected to fail but passes: FAILED (strict xpass behavior)
        - If test is not expected to fail: runs normally (PASSED or FAILED)

        Args:
            test_fn: A callable that runs the actual test assertions
        """
        expected = is_expected_failure(device_type, op_name, backend, test_type, dtype)

        if expected:
            try:
                test_fn()
            except (AssertionError, RuntimeError):
                pytest.xfail(f"Known failure: {op_name}/{backend}/{dtype}")
            else:
                self.fail(
                    f"XPASS: {op_name}/{backend}/{dtype} - remove from {test_type} xfails"
                )
        else:
            test_fn()

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

        def run_test():
            tested_any = False

            for sample in samples:
                # Skip if input is not a tensor or is 0-dim
                if (
                    not isinstance(sample.input, torch.Tensor)
                    or sample.input.dim() == 0
                ):
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

            if not tested_any:
                self.skipTest("No suitable samples found for batch invariance test")

        self._run_with_expected_failure(
            device_type, op.name, backend, "batch_invariance", dtype, run_test
        )

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
        2. Uses distort_benchmarking_result config to perturb autotuning choices
        3. Tests multiple runs and verifies bitwise identical outputs
        """
        torch._dynamo.reset()
        device_type = torch.device(device).type

        samples = self._get_sample_inputs(op, device, dtype)
        if not samples:
            self.skipTest(f"No samples for {op.name}")

        fn = op.get_op()

        def run_test():
            import torch._inductor.config as inductor_config

            for sample in samples:
                args = (sample.input,) + tuple(sample.args)
                kwargs = sample.kwargs

                # Run 1: Fresh cache, normal benchmarking
                torch._dynamo.reset()
                with fresh_inductor_cache():
                    compiled_fn1 = compile_fn(fn, backend)
                    out1 = compiled_fn1(*args, **kwargs)

                # Run 2: Fresh cache, inverted benchmark results
                torch._dynamo.reset()
                with fresh_inductor_cache():
                    with inductor_config.patch(
                        {"test_configs.distort_benchmarking_result": "inverse"}
                    ):
                        compiled_fn2 = compile_fn(fn, backend)
                        out2 = compiled_fn2(*args, **kwargs)

                # Run 3: Fresh cache, random benchmark results
                torch._dynamo.reset()
                with fresh_inductor_cache():
                    with inductor_config.patch(
                        {"test_configs.distort_benchmarking_result": "random"}
                    ):
                        compiled_fn3 = compile_fn(fn, backend)
                        out3 = compiled_fn3(*args, **kwargs)

                # Bitwise identical
                self.assertEqual(out1, out2, rtol=0, atol=0)
                self.assertEqual(out2, out3, rtol=0, atol=0)

        self._run_with_expected_failure(
            device_type, op.name, backend, "determinism", dtype, run_test
        )

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

        def run_test():
            for sample in samples:
                args = (sample.input,) + tuple(sample.args)
                kwargs = sample.kwargs

                # Eager reference
                eager_out = fn(*args, **kwargs)

                # Compiled output
                compiled_fn = compile_fn(fn, backend)
                compiled_out = compiled_fn(*args, **kwargs)

                # Bitwise identical
                self.assertEqual(eager_out, compiled_out, rtol=0, atol=0)

        self._run_with_expected_failure(
            device_type, op.name, backend, "eager_equivalence", dtype, run_test
        )

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

        def run_test():
            # Bitwise identical
            self.assertEqual(eager_out, compiled_out, rtol=0, atol=0)

        self._run_with_expected_failure(
            device_type, op.name, backend, "unary_numerical", dtype, run_test
        )

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

        def run_test():
            # Bitwise identical
            self.assertEqual(eager_out, compiled_out, rtol=0, atol=0)

        self._run_with_expected_failure(
            device_type, op.name, backend, "binary_numerical", dtype, run_test
        )


instantiate_device_type_tests(TestOpInfoProperties, globals(), except_for=["cpu"])

if __name__ == "__main__":
    run_tests()
