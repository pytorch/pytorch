# Owner(s): ["oncall: pt2"]
import functools
import re
import sys
import unittest

import torch
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._dynamo.testing import make_test_cls_with_patches
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import IS_CI, IS_WINDOWS
from torch.testing._internal.inductor_utils import HAS_PALLAS
from torch.utils._triton import has_triton


if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")


try:
    from . import test_torchinductor
except ImportError:
    import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library


test_classes = {}


def make_pallas(cls):
    """Create a test class variant that uses Pallas backend."""
    suffix = "_pallas"
    cls_prefix = "Pallas"

    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        (config, "cpu_backend", "pallas"),
        (config, "cuda_backend", "pallas"),
        xfail_prop="_expected_failure_pallas",
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


class PallasTestsMixin:
    """Basic tests for Pallas backend functionality (parameterized by DEVICE). Mixin only, not collected."""

    def _compile(self, fn):
        key = "cuda_backend" if self.DEVICE == "cuda" else "cpu_backend"
        return torch.compile(fn, backend="inductor", options={key: "pallas"})

    def test_simple_add(self):
        """Test basic element-wise addition."""

        def fn(a, b):
            return a + b

        compiled = self._compile(fn)

        a = torch.randn(1024, device=self.DEVICE)
        b = torch.randn(1024, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_simple_mul(self):
        """Test basic element-wise multiplication."""

        def fn(a, b):
            return a * b

        compiled = self._compile(fn)

        a = torch.randn(1024, device=self.DEVICE)
        b = torch.randn(1024, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_sin(self):
        """Test sin operation."""

        def fn(x):
            return torch.sin(x)

        compiled = self._compile(fn)

        x = torch.randn(1024, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_fused_ops(self):
        """Test fused operations (sin + add)."""

        def fn(x, y):
            return x.sin() + y

        compiled = self._compile(fn)

        x = torch.randn(1024, device=self.DEVICE)
        y = torch.randn(1024, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_exp_log(self):
        """Test exp and log operations."""

        def fn(x):
            return torch.log(torch.exp(x))

        compiled = self._compile(fn)

        x = torch.randn(1024, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_sqrt(self):
        """Test sqrt operation."""

        def fn(x):
            return torch.sqrt(x)

        compiled = self._compile(fn)

        x = torch.randn(1024, device=self.DEVICE).abs()  # Ensure positive for sqrt
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_tanh(self):
        """Test tanh operation."""

        def fn(x):
            return torch.tanh(x)

        compiled = self._compile(fn)

        x = torch.randn(1024, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_abs_neg(self):
        """Test abs and neg operations."""

        def fn(x):
            return torch.abs(-x)

        compiled = self._compile(fn)

        x = torch.randn(1024, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_maximum_minimum(self):
        """Test maximum and minimum operations."""

        def fn(a, b):
            return torch.maximum(a, b) + torch.minimum(a, b)

        compiled = self._compile(fn)

        a = torch.randn(1024, device=self.DEVICE)
        b = torch.randn(1024, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    @unittest.skipUnless(has_triton(), "requires triton")
    @unittest.skip("Random ops not yet implemented in Pallas backend")
    def test_random_consistency(self):
        """Test that random number generation is consistent across backends."""
        seed = 1234
        shape = (3, 3)
        dtype = torch.float32

        for rand_fn in [
            functools.partial(torch.rand, shape, dtype=dtype, device="cuda"),
            functools.partial(torch.randn, shape, dtype=dtype, device="cuda"),
        ]:

            @torch.compile(backend="inductor", options={"cuda_backend": "pallas"})
            def get_rand_pallas():
                return rand_fn()

            @torch.compile(backend="inductor", options={"cuda_backend": "triton"})
            def get_rand_triton():
                return rand_fn()

            torch.manual_seed(seed)
            pallas_output = get_rand_pallas()
            torch.manual_seed(seed)
            triton_output = get_rand_triton()

            self.assertEqual(pallas_output, triton_output)

    def test_compile_options(self):
        """Test that Pallas backend is properly configured."""

        @torch.compile(
            backend="inductor",
            options={
                ("cuda_backend" if self.DEVICE == "cuda" else "cpu_backend"): "pallas"
            },
        )
        def pallas_fn(a, b):
            return a.sin() + b.cos()

        _, (code,) = run_and_get_code(
            pallas_fn,
            torch.randn(64, device=self.DEVICE),
            torch.randn(64, device=self.DEVICE),
        )
        # Verify Pallas-specific code generation
        self.assertIn("import jax", code)
        self.assertIn("import jax.numpy as jnp", code)
        self.assertIn("from jax.experimental import pallas as pl", code)

    def test_jax_jit_wrapper_is_emitted(self):
        """Ensure generated Pallas code wraps pl.pallas_call in jax.jit."""

        key = "cuda_backend" if self.DEVICE == "cuda" else "cpu_backend"

        @torch.compile(backend="inductor", options={key: "pallas"})
        def pallas_fn(a, b):
            return a + b

        _, (code,) = run_and_get_code(
            pallas_fn,
            torch.randn(32, device=self.DEVICE),
            torch.randn(32, device=self.DEVICE),
        )

        kernel_match = re.search(r"def (pallas_[A-Za-z0-9_]+)_kernel", code)
        self.assertIsNotNone(kernel_match)
        kernel_name = kernel_match.group(1)
        wrapper_name = f"{kernel_name}_jit_wrapper"
        self.assertIn(wrapper_name, code)
        start = code.index(f"def {wrapper_name}")
        end = code.index(f"def {kernel_name}_main", start)
        wrapper_block = code[start:end]

        self.assertIn("jax.jit", code)
        self.assertIn("donate_argnums", code)
        self.assertIn("input_output_aliases", wrapper_block)
        if self.DEVICE == "cuda":
            self.assertNotIn(".copy_(", code)
        self.assertNotIn("torch.", wrapper_block)

    def test_2d_tensor(self):
        """Test with 2D tensors (though current implementation flattens)."""

        def fn(x, y):
            return x + y

        compiled = self._compile(fn)

        x = torch.randn(32, 32, device=self.DEVICE)
        y = torch.randn(32, 32, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_different_shapes(self):
        """Test with different tensor shapes."""

        def fn(x):
            return x * 2.0

        compiled = self._compile(fn)

        for shape in [(64,), (128,), (256,), (1024,)]:
            x = torch.randn(shape, device=self.DEVICE)
            result = compiled(x)
            expected = fn(x)
            self.assertEqual(result, expected)

    def test_contiguous_index_validation(self):
        """Test that contiguous index validation works correctly end-to-end."""

        # Test 1: Contiguous operations should work
        def contiguous_add(a, b):
            return a + b

        compiled = self._compile(contiguous_add)

        a = torch.randn(1024, device=self.DEVICE)
        b = torch.randn(1024, device=self.DEVICE)
        result = compiled(a, b)
        expected = contiguous_add(a, b)
        self.assertEqual(result, expected)

        # Test 2: Operations on contiguous tensors should work
        def contiguous_mul(x):
            return x * 2.0

        compiled = self._compile(contiguous_mul)

        x = torch.randn(128, 8, device=self.DEVICE)
        result = compiled(x)
        expected = contiguous_mul(x)
        self.assertEqual(result, expected)

        # Test 3: Non-contiguous views should work with the simplified dlpack approach
        # The direct dlpack conversion handles non-contiguous tensors correctly
        def operate_on_tensor(x):
            return x.sin()

        compiled = self._compile(operate_on_tensor)

        # Create a transposed (non-contiguous) view
        x = torch.randn(64, 32, device=self.DEVICE)
        x_t = x.t()  # Non-contiguous view
        self.assertFalse(x_t.is_contiguous())

        # With the simplified dlpack approach, non-contiguous tensors now work
        result = compiled(x_t)
        expected = operate_on_tensor(x_t)
        self.assertEqual(result, expected)

        # Contiguous tensors should also continue to work
        x_t_contiguous = x_t.contiguous()
        self.assertTrue(x_t_contiguous.is_contiguous())
        result = compiled(x_t_contiguous)
        expected = operate_on_tensor(x_t_contiguous)
        self.assertEqual(result, expected)

    def test_strided_int_pallas(self):
        """Test strided access patterns with the Pallas backend."""

        def fn(x):
            # Access every other element (strided access)
            return x[::2] * 2.0

        compiled = self._compile(fn)

        x = torch.arange(16, dtype=torch.float32, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_strided_offset_pallas(self):
        """Test strided access with offset."""

        def fn(x):
            # Access every other element starting from index 1
            return x[1::2] + 1.0

        compiled = self._compile(fn)

        x = torch.arange(16, dtype=torch.float32, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_strided_2d_pallas(self):
        """Test strided access on 2D tensors."""

        def fn(x):
            # Simple operation on 2D tensor
            return x * 3.0

        compiled = self._compile(fn)

        x = torch.randn(8, 16, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_non_power_of_2_sizes(self):
        """Test that non-power-of-2 tensor sizes work with masked ops on GPU.

        On GPU (Triton backend), Pallas requires power-of-2 sizes. We use masked
        loads/stores to handle non-power-of-2 tensors by allocating power-of-2
        blocks and masking out invalid elements.
        """

        def fn(a, b):
            return a + b

        compiled = self._compile(fn)

        # Test a specific non-power-of-2 size (10)
        a = torch.randn(10, device=self.DEVICE)
        b = torch.randn(10, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_non_power_of_2_multiple_ops(self):
        """Test non-power-of-2 sizes with multiple operations."""

        def fn(x, y):
            return x.sin() + y.cos() - (x * y)

        compiled = self._compile(fn)

        # Non-power-of-2 size: 17
        x = torch.randn(17, device=self.DEVICE)
        y = torch.randn(17, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_complex_indexing_gather(self):
        """Test complex indexing with gather-like operations."""

        def fn(x, indices):
            # Use indices to gather elements from x
            return x[indices]

        compiled = self._compile(fn)

        x = torch.arange(16, dtype=torch.float32, device=self.DEVICE)
        # Use power-of-2 size for indices (Pallas Triton requirement)
        indices = torch.tensor(
            [0, 2, 5, 7, 11, 13, 14, 15], dtype=torch.int64, device=self.DEVICE
        )
        result = compiled(x, indices)
        expected = fn(x, indices)
        self.assertEqual(result, expected)

    def test_complex_indexing_2d(self):
        """Test complex indexing on 2D tensors with integer array indexing."""
        if self.DEVICE == "cuda":
            # Pallas Triton backend doesn't support gather operations with array indices
            # This limitation is in the Pallas/Triton lowering, not our implementation
            self.skipTest(
                "Multi-dimensional gather not supported on Pallas Triton (CUDA) backend"
            )

        def fn(x, row_indices):
            # Select specific rows using integer array indexing
            return x[row_indices, :]

        compiled = self._compile(fn)

        x = torch.randn(16, 8, device=self.DEVICE)
        # Use power-of-2 sizes (Pallas Triton requirement)
        row_indices = torch.tensor([0, 2, 5, 7], dtype=torch.int64, device=self.DEVICE)
        result = compiled(x, row_indices)
        expected = fn(x, row_indices)
        self.assertEqual(result, expected)

    def test_complex64_mul(self):
        """Test complex64 multiplication."""

        def fn(a, b):
            return a * b

        compiled = self._compile(fn)

        a = torch.randn(16, dtype=torch.complex64, device=self.DEVICE)
        b = torch.randn(16, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_complex_conj(self):
        """Test complex conjugate."""

        def fn(x):
            return torch.conj(x)

        compiled = self._compile(fn)

        x = torch.randn(16, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_complex_real(self):
        """Test extracting real part of complex tensor."""

        def fn(x):
            return torch.real(x)

        compiled = self._compile(fn)

        x = torch.randn(16, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_complex_imag(self):
        """Test extracting imaginary part of complex tensor."""

        def fn(x):
            return torch.imag(x)

        compiled = self._compile(fn)

        x = torch.randn(16, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_complex_abs(self):
        """Test complex absolute value (magnitude)."""

        def fn(x):
            return torch.abs(x)

        compiled = self._compile(fn)

        x = torch.randn(16, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_complex128_conj(self):
        """Test complex128 conjugate operation."""

        def fn(x):
            return torch.conj(x)

        compiled = self._compile(fn)

        x = torch.randn(16, dtype=torch.complex128, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_complex_mul_scalar(self):
        """Test complex multiplication with scalar."""

        def fn(x):
            return x * 2.5

        compiled = self._compile(fn)

        x = torch.randn(16, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_complex_conj_mul(self):
        """Test conjugate followed by multiplication."""

        def fn(x, y):
            return torch.conj(x) * y

        compiled = self._compile(fn)

        x = torch.randn(16, dtype=torch.complex64, device=self.DEVICE)
        y = torch.randn(16, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_where(self):
        """Test torch.where operation."""

        def fn(x, y):
            return torch.where(x > 0, x, y)

        compiled = self._compile(fn)

        x = torch.randn(16, device=self.DEVICE)
        y = torch.randn(16, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_clamp(self):
        """Test torch.clamp operation."""

        def fn(x):
            return torch.clamp(x, -1.0, 1.0)

        compiled = self._compile(fn)

        x = torch.randn(16, device=self.DEVICE) * 2
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_comparison_ops(self):
        """Test comparison operations."""

        def fn(a, b):
            gt = a > b
            lt = a < b
            eq = a == b
            return gt.float() + lt.float() + eq.float()

        compiled = self._compile(fn)

        a = torch.randn(16, device=self.DEVICE)
        b = torch.randn(16, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_logical_ops(self):
        """Test logical operations."""

        def fn(a, b):
            return torch.logical_and(a > 0, b > 0).float()

        compiled = self._compile(fn)

        a = torch.randn(16, device=self.DEVICE)
        b = torch.randn(16, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_sign(self):
        """Test sign operation."""

        def fn(x):
            return torch.sign(x)

        compiled = self._compile(fn)

        x = torch.randn(16, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_reciprocal(self):
        """Test reciprocal operation."""

        def fn(x):
            return torch.reciprocal(x)

        compiled = self._compile(fn)

        x = torch.randn(16, device=self.DEVICE) + 1.0  # Avoid zeros
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_square(self):
        """Test square operation."""

        def fn(x):
            return torch.square(x)

        compiled = self._compile(fn)

        x = torch.randn(16, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_erf(self):
        """Test erf operation."""
        if self.DEVICE == "cuda":
            self.skipTest("erf not supported in Pallas GPU (Triton) backend")

        def fn(x):
            return torch.erf(x)

        compiled = self._compile(fn)

        x = torch.randn(16, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_atan2(self):
        """Test atan2 operation."""

        def fn(a, b):
            return torch.atan2(a, b)

        compiled = self._compile(fn)

        a = torch.randn(16, device=self.DEVICE)
        b = torch.randn(16, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_sum_reduction(self):
        """Test sum reduction."""

        def fn(x):
            return x.sum()

        compiled = self._compile(fn)

        x = torch.randn(16, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_max_reduction(self):
        """Test max reduction."""

        def fn(x):
            return x.max()

        compiled = self._compile(fn)

        x = torch.randn(16, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_min_reduction(self):
        """Test min reduction."""

        def fn(x):
            return x.min()

        compiled = self._compile(fn)

        x = torch.randn(16, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_prod_reduction(self):
        """Test prod reduction."""
        if self.DEVICE == "cuda":
            self.skipTest("prod reduction not supported in Pallas GPU (Triton) backend")

        def fn(x):
            # Use smaller values to avoid overflow
            return (x * 0.1).prod()

        compiled = self._compile(fn)

        x = torch.randn(16, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)


@unittest.skipUnless(HAS_PALLAS, "requires jax and pallas")
class PallasTestsCUDA(PallasTestsMixin, TestCase):
    DEVICE = "cuda"


@unittest.skipUnless(HAS_PALLAS, "requires jax and pallas")
class PallasTestsCPU(PallasTestsMixin, TestCase):
    DEVICE = "cpu"


if test_torchinductor.HAS_CPU and HAS_PALLAS:
    make_pallas(test_torchinductor.SweepInputsCpuTest)
    # make_pallas(test_torchinductor.CpuTests)


if test_torchinductor.HAS_GPU and HAS_PALLAS:
    # make_pallas(test_torchinductor.SweepInputsGPUTest)
    # make_pallas(test_torchinductor.GPUTests)
    pass


if __name__ == "__main__":
    if HAS_PALLAS:
        run_tests(needs="filelock")
