# Owner(s): ["oncall: pt2"]
import functools
import math
import os
import re
import sys
import unittest
from dataclasses import dataclass
from unittest import mock

import torch
import torch.nn.functional as F
import torch._dynamo
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._dynamo.testing import make_test_cls_with_patches
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import IS_CI, IS_WINDOWS
from torch.utils._pallas import has_cpu_pallas, has_cuda_pallas, has_tpu_pallas
from torch.utils._triton import has_triton


# Load pallas expected failures from sentinel files
_pallas_expected_failures_dir = os.path.join(
    os.path.dirname(__file__), "pallas_expected_failures"
)
if os.path.isdir(_pallas_expected_failures_dir):
    PALLAS_EXPECTED_FAILURES = set(os.listdir(_pallas_expected_failures_dir))
else:
    PALLAS_EXPECTED_FAILURES = set()

# Load pallas skip tests from sentinel files (for flaky tests)
_pallas_skip_tests_dir = os.path.join(os.path.dirname(__file__), "pallas_skip_tests")
if os.path.isdir(_pallas_skip_tests_dir):
    PALLAS_SKIP_TESTS = set(os.listdir(_pallas_skip_tests_dir))
else:
    PALLAS_SKIP_TESTS = set()


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

    # Mark tests based on sentinel files in pallas_expected_failures/ and pallas_skip_tests/
    for name in cls.__dict__:
        if name.startswith("test_"):
            fn = cls.__dict__[name]
            if callable(fn):
                key = f"{cls.__name__}.{name}"
                if key in PALLAS_EXPECTED_FAILURES:
                    fn._expected_failure_pallas = True
                elif key in PALLAS_SKIP_TESTS:
                    fn._skip_pallas = True

    def skip_decorator(fn):
        if hasattr(fn, "_skip_pallas"):
            return unittest.skip("Skipped in Pallas backend")(fn)
        return fn

    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        (config, "cpu_backend", "pallas"),
        (config, "cuda_backend", "pallas"),
        xfail_prop="_expected_failure_pallas",
        decorator=skip_decorator,
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


class PallasTestsMixin:
    """Basic tests for Pallas backend functionality (parameterized by DEVICE). Mixin only, not collected.

    NOTE: CUDA tests disable per-test fresh_cache to avoid JAX Mosaic GPU backend state issues.
    The Mosaic backend fails with "Failed to construct pass pipeline" when dynamically loading
    jit-compiled kernels across mock.patch.dict context boundaries (which fresh_cache uses).
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Disable per-test fresh_cache for CUDA to avoid JAX Mosaic state corruption
        if getattr(cls, "DEVICE", None) == "cuda":
            os.environ["INDUCTOR_TEST_DISABLE_FRESH_CACHE"] = "1"

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "DEVICE", None) == "cuda":
            os.environ.pop("INDUCTOR_TEST_DISABLE_FRESH_CACHE", None)
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        # Clear caches between tests to avoid Mosaic GPU backend state issues
        if self.DEVICE == "cuda":
            import gc

            torch._dynamo.reset()
            gc.collect()
            try:
                import jax

                jax.clear_caches()
            except ImportError:
                pass

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
        if self.DEVICE == "cuda":
            self.skipTest("sqrt not supported in Pallas GPU (Mosaic) backend")

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
        if self.DEVICE == "cuda":
            self.skipTest("abs not supported in Pallas GPU (Mosaic) backend")

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
            torch.randn(128, device=self.DEVICE),
            torch.randn(128, device=self.DEVICE),
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
            torch.randn(128, device=self.DEVICE),
            torch.randn(128, device=self.DEVICE),
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
        if self.DEVICE == "cuda":
            # Mosaic GPU backend uses plgpu.kernel instead of pl.pallas_call
            self.assertIn("plgpu.kernel", wrapper_block)
            self.assertNotIn(".copy_(", code)
        else:
            # CPU backend uses pl.pallas_call with input_output_aliases
            self.assertIn("input_output_aliases", wrapper_block)
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
        if self.DEVICE == "cuda":
            self.skipTest(
                "iteration variables not supported in Pallas GPU (Mosaic) backend"
            )

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
        if self.DEVICE == "cuda":
            self.skipTest("sin not supported in Pallas GPU (Mosaic) backend")

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
        if self.DEVICE == "cuda":
            self.skipTest("strided access not supported in Pallas GPU (Mosaic) backend")

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
        if self.DEVICE == "cuda":
            self.skipTest("strided access not supported in Pallas GPU (Mosaic) backend")

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
        if self.DEVICE == "cuda":
            self.skipTest("strided access not supported in Pallas GPU (Mosaic) backend")

        def fn(x):
            # Simple operation on 2D tensor
            return x * 3.0

        compiled = self._compile(fn)

        x = torch.randn(8, 16, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_stride_non_contiguous_1d(self):
        """Test 1D non-contiguous input patterns."""
        if self.DEVICE == "cuda":
            self.skipTest(
                "non-contiguous access not supported in Pallas GPU (Mosaic) backend"
            )
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_1d = torch.arange(256, dtype=torch.float32, device=self.DEVICE)
        for x in [base_1d[::2], base_1d[::4], base_1d[::2][::2]]:
            self.assertFalse(x.is_contiguous())
            self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_2d_row_stride(self):
        """Test 2D row-strided input patterns."""
        if self.DEVICE == "cuda":
            self.skipTest(
                "non-contiguous access not supported in Pallas GPU (Mosaic) backend"
            )
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_2d = torch.randn(32, 32, device=self.DEVICE)
        x = base_2d[::2, :]  # (16, 32) with stride (64, 1)
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_2d_col_stride(self):
        """Test 2D col-strided input patterns."""
        if self.DEVICE == "cuda":
            self.skipTest(
                "non-contiguous access not supported in Pallas GPU (Mosaic) backend"
            )
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_2d = torch.randn(32, 32, device=self.DEVICE)
        x = base_2d[:, ::2]  # (32, 16) with stride (32, 2)
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_2d_both_stride(self):
        """Test 2D both-strided input patterns."""
        if self.DEVICE == "cuda":
            self.skipTest(
                "non-contiguous access not supported in Pallas GPU (Mosaic) backend"
            )
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_2d = torch.randn(32, 32, device=self.DEVICE)
        x = base_2d[::2, ::2]  # (16, 16) with stride (64, 2)
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_2d_transpose(self):
        """Test 2D transposed input patterns."""
        if self.DEVICE == "cuda":
            self.skipTest(
                "non-contiguous access not supported in Pallas GPU (Mosaic) backend"
            )
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_2d = torch.randn(32, 32, device=self.DEVICE)
        x = base_2d.t()  # (32, 32) with stride (1, 32)
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_3d(self):
        """Test 3D non-contiguous input patterns."""
        if self.DEVICE == "cuda":
            self.skipTest(
                "non-contiguous access not supported in Pallas GPU (Mosaic) backend"
            )
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_3d = torch.randn(8, 8, 8, device=self.DEVICE)
        x = base_3d[::2, ::2, ::2]
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_permuted(self):
        """Test permuted non-contiguous input patterns."""
        if self.DEVICE == "cuda":
            self.skipTest(
                "non-contiguous access not supported in Pallas GPU (Mosaic) backend"
            )
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_3d = torch.randn(8, 8, 8, device=self.DEVICE)
        x = base_3d.permute(2, 0, 1)
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_channels_last(self):
        """Test channels-last (NHWC) non-contiguous input patterns."""
        if self.DEVICE == "cuda":
            self.skipTest(
                "non-contiguous access not supported in Pallas GPU (Mosaic) backend"
            )
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        x = torch.randn(2, 3, 4, 5, device=self.DEVICE).to(
            memory_format=torch.channels_last
        )
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_diagonal(self):
        """Test diagonal (large stride) non-contiguous input patterns."""
        if self.DEVICE == "cuda":
            self.skipTest(
                "non-contiguous access not supported in Pallas GPU (Mosaic) backend"
            )
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_2d = torch.randn(32, 32, device=self.DEVICE)
        x = base_2d.diagonal()
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_as_strided(self):
        """Test as_strided (custom layout) non-contiguous input patterns."""
        if self.DEVICE == "cuda":
            self.skipTest(
                "non-contiguous access not supported in Pallas GPU (Mosaic) backend"
            )
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_flat = torch.randn(256, device=self.DEVICE)
        x = torch.as_strided(base_flat, size=(4, 8), stride=(16, 2))
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_select_stride(self):
        """Test select then stride on non-contiguous input patterns."""
        if self.DEVICE == "cuda":
            self.skipTest(
                "non-contiguous access not supported in Pallas GPU (Mosaic) backend"
            )
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_2d = torch.randn(32, 32, device=self.DEVICE)
        x = base_2d[3, ::2]
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_unsqueeze(self):
        """Test unsqueeze on strided non-contiguous input patterns."""
        if self.DEVICE == "cuda":
            self.skipTest(
                "non-contiguous access not supported in Pallas GPU (Mosaic) backend"
            )
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_2d = torch.randn(32, 32, device=self.DEVICE)
        x = base_2d[::2, ::2].unsqueeze(0)
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_dtypes(self):
        """Test non-contiguous patterns with various dtypes."""
        if self.DEVICE == "cuda":
            self.skipTest(
                "non-contiguous access not supported in Pallas GPU (Mosaic) backend"
            )
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        for dtype in [torch.float64, torch.int32, torch.complex64]:
            if dtype == torch.int32:
                base = torch.arange(32, dtype=dtype, device=self.DEVICE)
            else:
                base = torch.randn(32, dtype=dtype, device=self.DEVICE)
            x = base[::2]
            self.assertEqual(compiled(x), x * 2.0 + 1.0)

    @unittest.skip(
        "Expanded tensors (stride=0) generate index expressions that don't match "
        "the contiguous layout after .contiguous() is called at runtime, causing "
        "gather ops on GPU which aren't supported by Pallas GPU lowering"
    )
    def test_stride_expanded_tensors(self):
        """Test expanded tensors with stride=0 (distinct code path)."""
        compiled = self._compile(lambda x, y: x + y)

        # Single dim expansion
        x = torch.randn(1, 16, device=self.DEVICE).expand(8, 16)
        y = torch.randn(8, 16, device=self.DEVICE)
        self.assertEqual(x.stride()[0], 0)
        self.assertEqual(compiled(x, y), x + y)

        # Multi-dim expansion
        x = torch.randn(1, 1, 16, device=self.DEVICE).expand(4, 8, 16)
        self.assertEqual(compiled(x, x), x + x)

    def test_stride_multiple_inputs(self):
        """Test multiple strided inputs and broadcasting."""
        if self.DEVICE == "cuda":
            self.skipTest(
                "non-contiguous access not supported in Pallas GPU (Mosaic) backend"
            )
        compiled = self._compile(lambda a, b, c: a * b + c)

        # Use separate base tensors to create strided inputs with the SAME stride pattern
        # This avoids triggering scatter operations which aren't supported in Pallas GPU lowering
        base_a = torch.randn(32, 32, device=self.DEVICE)
        base_b = torch.randn(32, 32, device=self.DEVICE)

        # Multiple strided inputs with the same stride pattern
        a = base_a[::2, ::2]  # (16, 16) with stride (64, 2)
        b = base_b[::2, ::2]  # (16, 16) with stride (64, 2)
        c = torch.randn(16, 16, device=self.DEVICE)
        self.assertEqual(a.stride(), b.stride())
        self.assertFalse(a.is_contiguous())
        self.assertFalse(b.is_contiguous())
        self.assertEqual(compiled(a, b, c), a * b + c)

        # Broadcasting with strided
        x = base_a[::2, ::2]  # (16, 16)
        y = torch.randn(16, device=self.DEVICE)  # broadcasts
        s = torch.tensor(2.0, device=self.DEVICE)  # scalar
        compiled_bcast = self._compile(lambda x, y, s: x + y * s)
        self.assertEqual(compiled_bcast(x, y, s), x + y * s)

    def test_non_power_of_2_sizes(self):
        """Test that non-power-of-2 tensor sizes work with masked ops on GPU.

        On GPU (Mosaic backend), Pallas requires power-of-2 sizes. We use masked
        loads/stores to handle non-power-of-2 tensors by allocating power-of-2
        blocks and masking out invalid elements.
        """
        if self.DEVICE == "cuda":
            self.skipTest(
                "non-power-of-2 sizes not supported in Pallas GPU (Mosaic) backend"
            )

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
        if self.DEVICE == "cuda":
            self.skipTest(
                "non-power-of-2 sizes not supported in Pallas GPU (Mosaic) backend"
            )

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
        if self.DEVICE == "cuda":
            self.skipTest("gather not supported in Pallas GPU (Mosaic) backend")

        def fn(x, indices):
            # Use indices to gather elements from x
            return x[indices]

        compiled = self._compile(fn)

        x = torch.arange(16, dtype=torch.float32, device=self.DEVICE)
        # Use power-of-2 size for indices (Pallas Mosaic requirement)
        indices = torch.tensor(
            [0, 2, 5, 7, 11, 13, 14, 15], dtype=torch.int64, device=self.DEVICE
        )
        result = compiled(x, indices)
        expected = fn(x, indices)
        self.assertEqual(result, expected)

    def test_complex_indexing_2d(self):
        """Test complex indexing on 2D tensors with integer array indexing."""
        if self.DEVICE == "cuda":
            # Pallas Mosaic backend doesn't support gather operations with array indices
            # This limitation is in the Pallas/Mosaic lowering, not our implementation
            self.skipTest(
                "Multi-dimensional gather not supported on Pallas Mosaic (CUDA) backend"
            )

        def fn(x, row_indices):
            # Select specific rows using integer array indexing
            return x[row_indices, :]

        compiled = self._compile(fn)

        x = torch.randn(16, 8, device=self.DEVICE)
        # Use power-of-2 sizes (Pallas Mosaic requirement)
        row_indices = torch.tensor([0, 2, 5, 7], dtype=torch.int64, device=self.DEVICE)
        result = compiled(x, row_indices)
        expected = fn(x, row_indices)
        self.assertEqual(result, expected)

    def test_complex64_mul(self):
        """Test complex64 multiplication."""

        def fn(a, b):
            return a * b

        compiled = self._compile(fn)

        a = torch.randn(128, dtype=torch.complex64, device=self.DEVICE)
        b = torch.randn(128, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_complex_conj(self):
        """Test complex conjugate."""

        def fn(x):
            return torch.conj(x)

        compiled = self._compile(fn)

        x = torch.randn(128, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_complex_real(self):
        """Test extracting real part of complex tensor."""

        def fn(x):
            return torch.real(x)

        compiled = self._compile(fn)

        x = torch.randn(128, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_complex_imag(self):
        """Test extracting imaginary part of complex tensor."""

        def fn(x):
            return torch.imag(x)

        compiled = self._compile(fn)

        x = torch.randn(128, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_complex_abs(self):
        """Test complex absolute value (magnitude)."""

        def fn(x):
            return torch.abs(x)

        compiled = self._compile(fn)

        x = torch.randn(128, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_complex128_conj(self):
        """Test complex128 conjugate operation."""

        def fn(x):
            return torch.conj(x)

        compiled = self._compile(fn)

        x = torch.randn(128, dtype=torch.complex128, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_complex_mul_scalar(self):
        """Test complex multiplication with scalar."""

        def fn(x):
            return x * 2.5

        compiled = self._compile(fn)

        x = torch.randn(128, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_complex_conj_mul(self):
        """Test conjugate followed by multiplication."""

        def fn(x, y):
            return torch.conj(x) * y

        compiled = self._compile(fn)

        x = torch.randn(128, dtype=torch.complex64, device=self.DEVICE)
        y = torch.randn(128, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_where(self):
        """Test torch.where operation."""

        def fn(x, y):
            return torch.where(x > 0, x, y)

        compiled = self._compile(fn)

        x = torch.randn(1024, device=self.DEVICE)
        y = torch.randn(1024, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_clamp(self):
        """Test torch.clamp operation."""

        def fn(x):
            return torch.clamp(x, -1.0, 1.0)

        compiled = self._compile(fn)

        x = torch.randn(1024, device=self.DEVICE) * 2
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

        a = torch.randn(1024, device=self.DEVICE)
        b = torch.randn(1024, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_logical_ops(self):
        """Test logical operations."""

        def fn(a, b):
            return torch.logical_and(a > 0, b > 0).float()

        compiled = self._compile(fn)

        a = torch.randn(1024, device=self.DEVICE)
        b = torch.randn(1024, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_sign(self):
        """Test sign operation."""
        if self.DEVICE == "cuda":
            self.skipTest("sign not supported in Pallas GPU (Mosaic) backend")

        def fn(x):
            return torch.sign(x)

        compiled = self._compile(fn)

        x = torch.randn(16, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_reciprocal(self):
        """Test reciprocal operation."""
        if self.DEVICE == "cuda":
            self.skipTest("reciprocal not supported in Pallas GPU (Mosaic) backend")

        def fn(x):
            return torch.reciprocal(x)

        compiled = self._compile(fn)

        x = torch.randn(16, device=self.DEVICE) + 1.0  # Avoid zeros
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_square(self):
        """Test square operation."""
        if self.DEVICE == "cuda":
            self.skipTest("square not supported in Pallas GPU (Mosaic) backend")

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
            self.skipTest("erf not supported in Pallas GPU (Mosaic) backend")

        def fn(x):
            return torch.erf(x)

        compiled = self._compile(fn)

        x = torch.randn(16, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_atan2(self):
        """Test atan2 operation."""
        if self.DEVICE == "cuda":
            self.skipTest("atan2 not supported in Pallas GPU (Mosaic) backend")

        def fn(a, b):
            return torch.atan2(a, b)

        compiled = self._compile(fn)

        a = torch.randn(16, device=self.DEVICE)
        b = torch.randn(16, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_residual_connection(self):
        """Test residual connection pattern: x + relu(linear(x)).

        This test verifies that view/reshape operations fused into kernels work
        correctly when input buffers have different shapes (e.g., matmul output
        shape (8, 8) vs input shape (2, 4, 8) that need element-wise addition).
        """
        if self.DEVICE == "cuda":
            self.skipTest(
                "Residual connection not supported in Pallas GPU (Mosaic) backend"
            )

        class ResidualBlock(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear = torch.nn.Linear(dim, dim)

            def forward(self, x):
                return x + torch.relu(self.linear(x))

        model = ResidualBlock(8)
        model.eval()
        if self.DEVICE != "cpu":
            model = model.to(self.DEVICE)

        x = torch.randn(2, 4, 8, device=self.DEVICE)
        expected = model(x)
        compiled_model = self._compile(model)
        result = compiled_model(x)
        self.assertEqual(result, expected)

    def test_embedding_with_positional(self):
        """Test token + positional embeddings pattern (GPT-style).

        This test verifies that embedding lookups with different indexing patterns
        work correctly when fused together. Token embeddings use input indices while
        positional embeddings use torch.arange-generated indices. The resulting tensors
        (batch, seq, dim) and (seq, dim) must broadcast correctly.
        """
        if self.DEVICE == "cuda":
            self.skipTest(
                "Embedding with positional not supported in Pallas GPU (Mosaic) backend"
            )

        class EmbeddingWithPositions(torch.nn.Module):
            def __init__(self, vocab_size, max_seq_len, dim):
                super().__init__()
                self.tok_emb = torch.nn.Embedding(vocab_size, dim)
                self.pos_emb = torch.nn.Embedding(max_seq_len, dim)

            def forward(self, tokens):
                batch, seq_len = tokens.shape
                tok = self.tok_emb(tokens)  # (batch, seq_len, dim)
                pos_indices = torch.arange(seq_len, device=tokens.device)
                pos = self.pos_emb(pos_indices)  # (seq_len, dim)
                return tok + pos

        model = EmbeddingWithPositions(vocab_size=256, max_seq_len=64, dim=64)
        model.eval()
        if self.DEVICE != "cpu":
            model = model.to(self.DEVICE)

        tokens = torch.randint(0, 256, (4, 16), device=self.DEVICE)
        expected = model(tokens)
        compiled_model = self._compile(model)
        result = compiled_model(tokens)
        self.assertEqual(result, expected)

    def test_mlp_block_with_residual(self):
        """Test MLP block with LayerNorm and residual connection.

        This test verifies that the common transformer MLP block pattern works
        correctly, combining LayerNorm, linear layers, GELU activation, and
        residual connections. This exercises both the view/reshape fix for
        residual broadcasting and the LayerNorm partial reduction handling.
        """
        if self.DEVICE == "cuda":
            self.skipTest(
                "MLP block with residual not supported in Pallas GPU (Mosaic) backend"
            )

        class MLPBlock(torch.nn.Module):
            def __init__(self, dim, hidden_dim):
                super().__init__()
                self.norm = torch.nn.LayerNorm(dim)
                self.w1 = torch.nn.Linear(dim, hidden_dim)
                self.w2 = torch.nn.Linear(hidden_dim, dim)

            def forward(self, x):
                h = self.norm(x)
                h = self.w2(torch.nn.functional.gelu(self.w1(h)))
                return x + h

        model = MLPBlock(dim=8, hidden_dim=32)
        model.eval()
        if self.DEVICE != "cpu":
            model = model.to(self.DEVICE)

        x = torch.randn(2, 4, 8, device=self.DEVICE)
        expected = model(x)
        compiled_model = self._compile(model)
        result = compiled_model(x)
        self.assertEqual(result, expected)

    def test_torch_nn_LayerNorm(self):
        """Test nn.LayerNorm with Pallas backend.

        This test verifies that nn.LayerNorm works correctly with the Pallas backend,
        including proper handling of partial reductions with symbolic dimensions.
        """
        if self.DEVICE == "cuda":
            self.skipTest("LayerNorm not supported in Pallas GPU (Mosaic) backend")

        # Warm up dynamo cache with other operations to reproduce the issue
        def add_fn(a, b):
            return a + b

        compiled_add = self._compile(add_fn)
        a = torch.randn(4, 4, device=self.DEVICE)
        b = torch.randn(4, 4, device=self.DEVICE)
        _ = compiled_add(a, b)

        def mul_fn(a, b):
            return a * b

        compiled_mul = self._compile(mul_fn)
        _ = compiled_mul(a, b)

        def relu_fn(x):
            return torch.relu(x)

        compiled_relu = self._compile(relu_fn)
        _ = compiled_relu(a)

        def sum_fn(x):
            return x.sum()

        compiled_sum = self._compile(sum_fn)
        _ = compiled_sum(a)

        linear = torch.nn.Linear(8, 4)
        linear.eval()
        if self.DEVICE != "cpu":
            linear = linear.to(self.DEVICE)
        compiled_linear = self._compile(linear)
        _ = compiled_linear(torch.randn(2, 8, device=self.DEVICE))

        # Now test nn.LayerNorm module
        ln = torch.nn.LayerNorm(8)
        ln.eval()
        if self.DEVICE != "cpu":
            ln = ln.to(self.DEVICE)

        x = torch.randn(2, 4, 8, device=self.DEVICE)
        expected = ln(x)
        compiled_ln = self._compile(ln)
        result = compiled_ln(x)
        self.assertEqual(result, expected)

    def test_repro_layernorm_after_linear_symbolic_shape(self):
        """Repro: LayerNorm after Linear produces wrong results due to symbolic shape pollution.

        When compiling multiple models without torch._dynamo.reset(), the symbolic
        shapes from one kernel (e.g., Linear's ks0*ks1) can pollute iteration variable
        reshaping in subsequent kernels (e.g., LayerNorm), causing incorrect results.
        """
        if self.DEVICE == "cuda":
            self.skipTest("LayerNorm not supported in Pallas GPU (Mosaic) backend")

        # First compile and run a Linear model
        linear = torch.nn.Linear(8, 4)
        linear.eval()
        if self.DEVICE != "cpu":
            linear = linear.to(self.DEVICE)
        compiled_linear = self._compile(linear)
        _ = compiled_linear(torch.randn(2, 8, device=self.DEVICE))

        # Then compile and run LayerNorm WITHOUT reset
        # This should still produce correct results
        ln = torch.nn.LayerNorm(8)
        ln.eval()
        if self.DEVICE != "cpu":
            ln = ln.to(self.DEVICE)

        x = torch.randn(2, 4, 8, device=self.DEVICE)
        expected = ln(x)
        compiled_ln = self._compile(ln)
        result = compiled_ln(x)
        self.assertEqual(result, expected)

    def test_repro_inline_causal_mask(self):
        """Repro: Inline causal mask creation causes broadcast shape mismatch."""
        if self.DEVICE == "cuda":
            self.skipTest(
                "Inline causal mask not supported in Pallas GPU (Mosaic) backend"
            )

        batch, heads, seq, dim = 2, 4, 8, 16
        q = torch.randn(batch, heads, seq, dim, device=self.DEVICE)
        k = torch.randn(batch, heads, seq, dim, device=self.DEVICE)
        v = torch.randn(batch, heads, seq, dim, device=self.DEVICE)

        def attention_with_inline_mask(q, k, v):
            scale = dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            seq_len = scores.size(-1)
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=scores.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.nn.functional.softmax(scores, dim=-1)
            return torch.matmul(attn, v)

        expected = attention_with_inline_mask(q, k, v)
        compiled_fn = self._compile(attention_with_inline_mask)
        result = compiled_fn(q, k, v)
        self.assertEqual(result, expected)

    def test_repro_embedding_layernorm_fusion(self):
        """Repro: Embedding + LayerNorm fusion causes shape mismatch."""
        if self.DEVICE == "cuda":
            self.skipTest(
                "Embedding + LayerNorm fusion not supported in Pallas GPU (Mosaic) backend"
            )

        emb = torch.nn.Embedding(256, 64)
        ln = torch.nn.LayerNorm(64)
        emb.eval()
        ln.eval()
        if self.DEVICE != "cpu":
            emb = emb.to(self.DEVICE)
            ln = ln.to(self.DEVICE)

        tokens = torch.randint(0, 256, (4, 16), device=self.DEVICE)

        def fn(tokens):
            x = emb(tokens)
            return ln(x)

        expected = fn(tokens)
        compiled_fn = self._compile(fn)
        result = compiled_fn(tokens)
        self.assertEqual(result, expected)

    def test_repro_groupnorm(self):
        """Repro: GroupNorm causes broadcast shape mismatch."""
        if self.DEVICE == "cuda":
            self.skipTest("GroupNorm not supported in Pallas GPU (Mosaic) backend")

        gn = torch.nn.GroupNorm(num_groups=4, num_channels=64)
        gn.eval()
        if self.DEVICE != "cpu":
            gn = gn.to(self.DEVICE)

        x = torch.randn(4, 64, 16, device=self.DEVICE)
        expected = gn(x)
        compiled = self._compile(gn)
        result = compiled(x)
        self.assertEqual(result, expected)

    def test_repro_conv_transpose(self):
        """Repro: Conv1d with transpose produces incorrect results."""
        if self.DEVICE == "cuda":
            self.skipTest(
                "Conv1d transpose pattern not supported in Pallas GPU (Mosaic) backend"
            )

        conv = torch.nn.Conv1d(64, 32, kernel_size=3, padding=1)
        conv.eval()
        if self.DEVICE != "cpu":
            conv = conv.to(self.DEVICE)

        x = torch.randn(4, 16, 64, device=self.DEVICE)

        def fn(x):
            return conv(x.transpose(1, 2)).transpose(1, 2)

        expected = fn(x)
        compiled_fn = self._compile(fn)
        result = compiled_fn(x)
        self.assertEqual(result, expected)

    def test_repro_transpose_same_dim_sizes(self):
        """Repro: Transpose with all dimensions having the same size.

        When output shape has multiple dimensions with the same size (e.g., [8, 8, 8]),
        dimension matching by size fails. Must use coefficient/stride analysis instead.
        """
        if self.DEVICE == "cuda":
            self.skipTest(
                "Transpose same-dim-size not supported in Pallas GPU (Mosaic) backend"
            )

        x = torch.randn(8, 8, 8, device=self.DEVICE)

        def fn(x):
            # Transpose dimensions 1 and 2
            return x.transpose(1, 2).contiguous()

        expected = fn(x)
        compiled_fn = self._compile(fn)
        result = compiled_fn(x)
        self.assertEqual(result, expected)

    def test_repro_attention_softmax(self):
        """Repro: Multi-head attention softmax reduction issue."""
        if self.DEVICE == "cuda":
            self.skipTest(
                "Attention softmax not supported in Pallas GPU (Mosaic) backend"
            )

        class SimpleAttention(torch.nn.Module):
            def __init__(self, dim, n_heads):
                super().__init__()
                assert dim % n_heads == 0
                self.n_heads = n_heads
                self.head_dim = dim // n_heads

                self.wq = torch.nn.Linear(dim, dim, bias=False)
                self.wk = torch.nn.Linear(dim, dim, bias=False)
                self.wv = torch.nn.Linear(dim, dim, bias=False)
                self.wo = torch.nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                bsz, seq_len, _ = x.size()
                q = (
                    self.wq(x)
                    .view(bsz, seq_len, self.n_heads, self.head_dim)
                    .transpose(1, 2)
                )
                k = (
                    self.wk(x)
                    .view(bsz, seq_len, self.n_heads, self.head_dim)
                    .transpose(1, 2)
                )
                v = (
                    self.wv(x)
                    .view(bsz, seq_len, self.n_heads, self.head_dim)
                    .transpose(1, 2)
                )
                scale = self.head_dim ** -0.5
                scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device), diagonal=1
                ).bool()
                scores = scores.masked_fill(mask, float("-inf"))
                attn = torch.nn.functional.softmax(scores, dim=-1)
                output = torch.matmul(attn, v)
                output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
                return self.wo(output)

        model = SimpleAttention(dim=64, n_heads=4)
        model.eval()
        if self.DEVICE != "cpu":
            model = model.to(self.DEVICE)

        x = torch.randn(2, 8, 64, device=self.DEVICE)
        expected = model(x)
        compiled_model = self._compile(model)
        result = compiled_model(x)
        self.assertEqual(result, expected)

    def test_repro_embedding_indexing(self):
        """Repro: Embedding with torch.arange indexing issue."""
        if self.DEVICE == "cuda":
            self.skipTest(
                "Embedding indexing not supported in Pallas GPU (Mosaic) backend"
            )

        class EmbeddingWithPositions(torch.nn.Module):
            def __init__(self, vocab_size, max_seq_len, dim):
                super().__init__()
                self.tok_emb = torch.nn.Embedding(vocab_size, dim)
                self.pos_emb = torch.nn.Embedding(max_seq_len, dim)

            def forward(self, tokens):
                batch, seq_len = tokens.shape
                tok = self.tok_emb(tokens)
                pos_indices = torch.arange(seq_len, device=tokens.device)
                pos = self.pos_emb(pos_indices)
                return tok + pos

        model = EmbeddingWithPositions(vocab_size=256, max_seq_len=64, dim=64)
        model.eval()
        if self.DEVICE != "cpu":
            model = model.to(self.DEVICE)

        tokens = torch.randint(0, 256, (4, 16), device=self.DEVICE)
        expected = model(tokens)
        compiled_model = self._compile(model)
        result = compiled_model(tokens)
        self.assertEqual(result, expected)

    def test_repro_layernorm(self):
        """Repro: LayerNorm numerical precision issue after cache warmup.

        Matches repro_tpu_test_cases_v2/repro_layernorm.py.
        """
        if self.DEVICE == "cuda":
            self.skipTest("LayerNorm not supported in Pallas GPU (Mosaic) backend")

        def run_test(fn, inputs):
            expected = fn(*inputs)
            compiled_fn = self._compile(fn)
            result = compiled_fn(*inputs)
            self.assertEqual(result, expected)

        run_test(lambda a, b: a + b, (torch.randn(4, 4, device=self.DEVICE), torch.randn(4, 4, device=self.DEVICE)))
        run_test(lambda a, b: a * b, (torch.randn(4, 4, device=self.DEVICE), torch.randn(4, 4, device=self.DEVICE)))
        run_test(lambda x: torch.relu(x), (torch.randn(4, 4, device=self.DEVICE),))
        run_test(lambda x: x.sum(), (torch.randn(4, 4, device=self.DEVICE),))

        linear = torch.nn.Linear(8, 4)
        linear.eval()
        if self.DEVICE != "cpu":
            linear = linear.to(self.DEVICE)
        run_test(linear, (torch.randn(2, 8, device=self.DEVICE),))

        ln = torch.nn.LayerNorm(8)
        ln.eval()
        if self.DEVICE != "cpu":
            ln = ln.to(self.DEVICE)

        x = torch.randn(2, 4, 8, device=self.DEVICE)
        expected = ln(x)
        compiled_ln = self._compile(ln)
        result = compiled_ln(x)
        self.assertEqual(result, expected)

        torch._dynamo.reset()
        ln2 = torch.nn.LayerNorm(8)
        ln2.eval()
        if self.DEVICE != "cpu":
            ln2 = ln2.to(self.DEVICE)

        x2 = torch.randn(2, 4, 8, device=self.DEVICE)
        expected2 = ln2(x2)
        compiled_ln2 = self._compile(ln2)
        result2 = compiled_ln2(x2)
        self.assertEqual(result2, expected2)

    def test_repro_mlp_block_with_residual(self):
        """Repro: MLP block with LayerNorm and residual connection issue."""
        if self.DEVICE == "cuda":
            self.skipTest(
                "MLP block with residual not supported in Pallas GPU (Mosaic) backend"
            )

        class MLPBlock(torch.nn.Module):
            def __init__(self, dim, hidden_dim):
                super().__init__()
                self.norm = torch.nn.LayerNorm(dim)
                self.w1 = torch.nn.Linear(dim, hidden_dim)
                self.w2 = torch.nn.Linear(hidden_dim, dim)

            def forward(self, x):
                h = self.norm(x)
                h = self.w2(torch.nn.functional.gelu(self.w1(h)))
                return x + h

        model = MLPBlock(dim=8, hidden_dim=32)
        model.eval()
        if self.DEVICE != "cpu":
            model = model.to(self.DEVICE)

        x = torch.randn(2, 4, 8, device=self.DEVICE)
        expected = model(x)
        compiled_model = self._compile(model)
        result = compiled_model(x)
        self.assertEqual(result, expected)

    def test_repro_residual_connection(self):
        """Repro: Residual connection broadcasting issue."""
        if self.DEVICE == "cuda":
            self.skipTest(
                "Residual connection not supported in Pallas GPU (Mosaic) backend"
            )

        class ResidualBlock(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear = torch.nn.Linear(dim, dim)

            def forward(self, x):
                return x + torch.nn.functional.relu(self.linear(x))

        model = ResidualBlock(8)
        model.eval()
        if self.DEVICE != "cpu":
            model = model.to(self.DEVICE)

        x = torch.randn(2, 4, 8, device=self.DEVICE)
        expected = model(x)
        compiled_model = self._compile(model)
        result = compiled_model(x)
        self.assertEqual(result, expected)

    def test_nanogpt(self):
        """Test real Karpathy NanoGPT model.

        This is the actual NanoGPT implementation from:
        https://github.com/karpathy/nanoGPT/blob/master/model.py

        Tests the full transformer architecture including:
        - Token and position embeddings
        - Multi-head causal self-attention
        - MLP with GELU activation
        - LayerNorm (pre-norm architecture)
        - Residual connections
        - Weight tying between embeddings and output
        """
        if self.DEVICE == "cuda":
            self.skipTest("NanoGPT not supported in Pallas GPU (Mosaic) backend")

        # ============================================================
        # NanoGPT model from https://github.com/karpathy/nanoGPT
        # ============================================================

        class LayerNorm(torch.nn.Module):
            """LayerNorm but with an optional bias."""

            def __init__(self, ndim, bias):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(ndim))
                self.bias = torch.nn.Parameter(torch.zeros(ndim)) if bias else None

            def forward(self, input):
                return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

        class CausalSelfAttention(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                assert config.n_embd % config.n_head == 0
                self.c_attn = torch.nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
                self.c_proj = torch.nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
                self.attn_dropout = torch.nn.Dropout(config.dropout)
                self.resid_dropout = torch.nn.Dropout(config.dropout)
                self.n_head = config.n_head
                self.n_embd = config.n_embd
                self.dropout = config.dropout
                self.flash = hasattr(F, 'scaled_dot_product_attention')
                if not self.flash:
                    self.register_buffer(
                        "bias",
                        torch.tril(torch.ones(config.block_size, config.block_size)).view(
                            1, 1, config.block_size, config.block_size
                        ),
                    )

            def forward(self, x):
                B, T, C = x.size()
                q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
                k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

                if self.flash:
                    y = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=None,
                        dropout_p=self.dropout if self.training else 0,
                        is_causal=True,
                    )
                else:
                    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                    att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
                    att = F.softmax(att, dim=-1)
                    att = self.attn_dropout(att)
                    y = att @ v
                y = y.transpose(1, 2).contiguous().view(B, T, C)
                y = self.resid_dropout(self.c_proj(y))
                return y

        class MLP(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.c_fc = torch.nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
                self.gelu = torch.nn.GELU()
                self.c_proj = torch.nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
                self.dropout = torch.nn.Dropout(config.dropout)

            def forward(self, x):
                x = self.c_fc(x)
                x = self.gelu(x)
                x = self.c_proj(x)
                x = self.dropout(x)
                return x

        class Block(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
                self.attn = CausalSelfAttention(config)
                self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
                self.mlp = MLP(config)

            def forward(self, x):
                x = x + self.attn(self.ln_1(x))
                x = x + self.mlp(self.ln_2(x))
                return x

        @dataclass
        class GPTConfig:
            block_size: int = 1024
            vocab_size: int = 50304
            n_layer: int = 12
            n_head: int = 12
            n_embd: int = 768
            dropout: float = 0.0
            bias: bool = True

        class GPT(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                assert config.vocab_size is not None
                assert config.block_size is not None
                self.config = config

                self.transformer = torch.nn.ModuleDict(
                    dict(
                        wte=torch.nn.Embedding(config.vocab_size, config.n_embd),
                        wpe=torch.nn.Embedding(config.block_size, config.n_embd),
                        drop=torch.nn.Dropout(config.dropout),
                        h=torch.nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                        ln_f=LayerNorm(config.n_embd, bias=config.bias),
                    )
                )
                self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
                self.transformer.wte.weight = self.lm_head.weight  # weight tying

            def forward(self, idx, targets=None):
                device = idx.device
                b, t = idx.size()
                assert t <= self.config.block_size
                pos = torch.arange(0, t, dtype=torch.long, device=device)

                tok_emb = self.transformer.wte(idx)
                pos_emb = self.transformer.wpe(pos)
                x = self.transformer.drop(tok_emb + pos_emb)
                for block in self.transformer.h:
                    x = block(x)
                x = self.transformer.ln_f(x)

                if targets is not None:
                    logits = self.lm_head(x)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
                    )
                else:
                    logits = self.lm_head(x[:, [-1], :])
                    loss = None

                return logits, loss

        # Small config for testing
        config = GPTConfig(
            vocab_size=256,
            block_size=32,
            n_layer=2,
            n_head=4,
            n_embd=64,
            dropout=0.0,
            bias=False,
        )

        model = GPT(config)
        model.eval()
        if self.DEVICE != "cpu":
            model = model.to(self.DEVICE)

        # Test input
        x = torch.randint(0, config.vocab_size, (2, 16), device=self.DEVICE)

        # Run eager
        with torch.no_grad():
            expected, _ = model(x)

        # Run compiled
        compiled_model = self._compile(model)
        with torch.no_grad():
            result, _ = compiled_model(x)

        self.assertEqual(result, expected)

    def test_llama3(self):
        """Test Llama 3 model architecture.

        This is adapted from the official Meta Llama 3 implementation:
        https://github.com/meta-llama/llama3/blob/main/llama/model.py

        Tests the Llama 3 architecture including:
        - RMSNorm (Root Mean Square Layer Normalization)
        - Rotary Position Embeddings (RoPE)
        - Grouped Query Attention (GQA)
        - SwiGLU Feed-Forward Network
        - Residual connections
        """
        if self.DEVICE == "cuda":
            self.skipTest("Llama3 not supported in Pallas GPU (Mosaic) backend")

        # ============================================================
        # Llama 3 model from https://github.com/meta-llama/llama3
        # Adapted to use standard PyTorch (no FairScale dependencies)
        # ============================================================

        from typing import Optional, Tuple

        @dataclass
        class ModelArgs:
            dim: int = 64  # Small for testing (original: 4096)
            n_layers: int = 2  # Small for testing (original: 32)
            n_heads: int = 4  # Small for testing (original: 32)
            n_kv_heads: Optional[int] = 2  # For GQA (original: 8 for 70B)
            vocab_size: int = 256  # Small for testing
            multiple_of: int = 64  # Make SwiGLU hidden layer size multiple of this
            ffn_dim_multiplier: Optional[float] = None
            norm_eps: float = 1e-5
            rope_theta: float = 500000.0
            max_seq_len: int = 32

        class RMSNorm(torch.nn.Module):
            """Root Mean Square Layer Normalization."""

            def __init__(self, dim: int, eps: float = 1e-6):
                super().__init__()
                self.eps = eps
                self.weight = torch.nn.Parameter(torch.ones(dim))

            def _norm(self, x):
                return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

            def forward(self, x):
                output = self._norm(x.float()).type_as(x)
                return output * self.weight

        def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
            """Precompute the frequency tensor for rotary embeddings."""
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
            t = torch.arange(end, device=freqs.device, dtype=torch.float32)
            freqs = torch.outer(t, freqs)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
            return freqs_cis

        def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
            """Reshape frequency tensor for broadcasting with x."""
            ndim = x.ndim
            assert 0 <= 1 < ndim
            assert freqs_cis.shape == (x.shape[1], x.shape[-1])
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
            return freqs_cis.view(*shape)

        def apply_rotary_emb(
            xq: torch.Tensor,
            xk: torch.Tensor,
            freqs_cis: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Apply rotary embeddings to query and key tensors."""
            xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
            xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
            freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
            xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
            xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
            return xq_out.type_as(xq), xk_out.type_as(xk)

        def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
            """Repeat key/value heads for grouped query attention."""
            bs, slen, n_kv_heads, head_dim = x.shape
            if n_rep == 1:
                return x
            return (
                x[:, :, :, None, :]
                .expand(bs, slen, n_kv_heads, n_rep, head_dim)
                .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
            )

        class Attention(torch.nn.Module):
            """Multi-head attention with Grouped Query Attention (GQA)."""

            def __init__(self, args: ModelArgs):
                super().__init__()
                self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
                self.n_heads = args.n_heads
                self.n_rep = self.n_heads // self.n_kv_heads
                self.head_dim = args.dim // args.n_heads

                self.wq = torch.nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
                self.wk = torch.nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
                self.wv = torch.nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
                self.wo = torch.nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

            def forward(
                self,
                x: torch.Tensor,
                freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor],
            ):
                bsz, seqlen, _ = x.shape
                xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

                xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
                xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
                xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

                xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

                # Repeat k/v heads if n_kv_heads < n_heads (GQA)
                keys = repeat_kv(xk, self.n_rep)
                values = repeat_kv(xv, self.n_rep)

                xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
                keys = keys.transpose(1, 2)
                values = values.transpose(1, 2)

                scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
                if mask is not None:
                    scores = scores + mask
                scores = F.softmax(scores.float(), dim=-1).type_as(xq)
                output = torch.matmul(scores, values)
                output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
                return self.wo(output)

        class FeedForward(torch.nn.Module):
            """SwiGLU Feed-Forward Network."""

            def __init__(
                self,
                dim: int,
                hidden_dim: int,
                multiple_of: int,
                ffn_dim_multiplier: Optional[float],
            ):
                super().__init__()
                hidden_dim = int(2 * hidden_dim / 3)
                if ffn_dim_multiplier is not None:
                    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
                hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

                self.w1 = torch.nn.Linear(dim, hidden_dim, bias=False)
                self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)
                self.w3 = torch.nn.Linear(dim, hidden_dim, bias=False)

            def forward(self, x):
                return self.w2(F.silu(self.w1(x)) * self.w3(x))

        class TransformerBlock(torch.nn.Module):
            """Single Transformer block with attention and feed-forward."""

            def __init__(self, layer_id: int, args: ModelArgs):
                super().__init__()
                self.n_heads = args.n_heads
                self.dim = args.dim
                self.head_dim = args.dim // args.n_heads
                self.attention = Attention(args)
                self.feed_forward = FeedForward(
                    dim=args.dim,
                    hidden_dim=4 * args.dim,
                    multiple_of=args.multiple_of,
                    ffn_dim_multiplier=args.ffn_dim_multiplier,
                )
                self.layer_id = layer_id
                self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
                self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

            def forward(
                self,
                x: torch.Tensor,
                freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor],
            ):
                h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
                out = h + self.feed_forward(self.ffn_norm(h))
                return out

        class Transformer(torch.nn.Module):
            """Llama 3 Transformer model."""

            def __init__(self, params: ModelArgs):
                super().__init__()
                self.params = params
                self.vocab_size = params.vocab_size
                self.n_layers = params.n_layers

                self.tok_embeddings = torch.nn.Embedding(params.vocab_size, params.dim)
                self.layers = torch.nn.ModuleList()
                for layer_id in range(params.n_layers):
                    self.layers.append(TransformerBlock(layer_id, params))
                self.norm = RMSNorm(params.dim, eps=params.norm_eps)
                self.output = torch.nn.Linear(params.dim, params.vocab_size, bias=False)

                # Precompute rotary embeddings
                self.freqs_cis = precompute_freqs_cis(
                    params.dim // params.n_heads,
                    params.max_seq_len * 2,
                    params.rope_theta,
                )

            def forward(self, tokens: torch.Tensor):
                bsz, seqlen = tokens.shape
                h = self.tok_embeddings(tokens)
                self.freqs_cis = self.freqs_cis.to(h.device)
                freqs_cis = self.freqs_cis[:seqlen]

                # Causal mask
                mask = None
                if seqlen > 1:
                    mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
                    mask = torch.triu(mask, diagonal=1)
                    mask = mask.type_as(h)

                for layer in self.layers:
                    h = layer(h, freqs_cis, mask)
                h = self.norm(h)
                output = self.output(h).float()
                return output

        # Small config for testing (Llama 3 70B would have much larger dims)
        args = ModelArgs(
            dim=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,  # GQA: 2 KV heads shared among 4 query heads
            vocab_size=256,
            multiple_of=64,
            norm_eps=1e-5,
            rope_theta=500000.0,
            max_seq_len=32,
        )

        model = Transformer(args)
        model.eval()
        if self.DEVICE != "cpu":
            model = model.to(self.DEVICE)

        # Test input
        x = torch.randint(0, args.vocab_size, (2, 16), device=self.DEVICE)

        # Run eager
        with torch.no_grad():
            expected = model(x)

        # Run compiled
        compiled_model = self._compile(model)
        with torch.no_grad():
            result = compiled_model(x)

        self.assertEqual(result, expected)

    def test_sum_reduction(self):
        """Test sum reduction."""

        def fn(x):
            return x.sum()

        compiled = self._compile(fn)

        x = torch.randn(1024, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_max_reduction(self):
        """Test max reduction."""

        def fn(x):
            return x.max()

        compiled = self._compile(fn)

        x = torch.randn(1024, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_min_reduction(self):
        """Test min reduction."""
        if self.DEVICE == "cuda":
            self.skipTest("min reduction not supported in Pallas GPU (Mosaic) backend")

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
            self.skipTest("prod reduction not supported in Pallas GPU (Mosaic) backend")

        def fn(x):
            # Use smaller values to avoid overflow
            return (x * 0.1).prod()

        compiled = self._compile(fn)

        x = torch.randn(16, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_arange_multi_output(self):
        """Test arange with view and multiple outputs."""
        if self.DEVICE == "cuda":
            self.skipTest("arange not supported in Pallas GPU (Mosaic) backend")

        def fn(x):
            rng1 = torch.arange(8 * 8, dtype=torch.float32, device=x.device).view(8, 8)
            rng2 = torch.arange(10, 18, device=x.device)
            tmp = x * rng1
            return tmp, tmp + rng2

        compiled = self._compile(fn)

        x = torch.randn(8, 8, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(len(result), len(expected))
        for r, e in zip(result, expected):
            self.assertEqual(r, e)

    def test_dtype_bitcast(self):
        """Test dtype bitcast (view tensor as different dtype)."""

        def fn(x):
            # View float32 tensor as int32 (same byte size)
            return x.view(torch.int32)

        compiled = self._compile(fn)

        x = torch.randn(128, device=self.DEVICE, dtype=torch.float32)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_dtype_bitcast_float16_to_int16(self):
        """Test dtype bitcast from float16 to int16."""

        def fn(x):
            return x.view(torch.int16)

        compiled = self._compile(fn)

        x = torch.randn(128, device=self.DEVICE, dtype=torch.float16)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_warpgroup_size_exact_128(self):
        """Test with exactly 128 elements (1 warpgroup)."""

        def fn(a, b):
            return a + b

        compiled = self._compile(fn)

        # 128 = WARPGROUP_SIZE (exactly 1 warpgroup)
        a = torch.randn(128, device=self.DEVICE)
        b = torch.randn(128, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_warpgroup_size_multiple_256(self):
        """Test with 256 elements (multiple of 128)."""

        def fn(a, b):
            return a * b + a

        compiled = self._compile(fn)

        # 256 = 2 * WARPGROUP_SIZE
        a = torch.randn(256, device=self.DEVICE)
        b = torch.randn(256, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_warpgroup_size_multiple_512(self):
        """Test with 512 elements (multiple of 128)."""

        def fn(a, b):
            return a + b * 2

        compiled = self._compile(fn)

        # 512 = 4 * WARPGROUP_SIZE
        a = torch.randn(512, device=self.DEVICE)
        b = torch.randn(512, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_warpgroup_size_below_128(self):
        """Test with fewer than 128 elements.

        Mosaic GPU requires tensor sizes to be multiples of 128 (WARPGROUP_SIZE).
        Automatic padding aligns tensors to 128 for GPU.
        """

        def fn(a, b):
            return a + b

        compiled = self._compile(fn)

        # 100 < 128, automatic padding aligns to 128
        a = torch.randn(100, device=self.DEVICE)
        b = torch.randn(100, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_warpgroup_size_non_aligned(self):
        """Test with non-aligned size (not a multiple of 128).

        Mosaic GPU requires tensor sizes to be multiples of 128 (WARPGROUP_SIZE).
        Automatic padding aligns tensors to 128 for GPU.
        """

        def fn(a, b):
            return a * b

        compiled = self._compile(fn)

        # 200 is not a multiple of 128, automatic padding aligns to 256
        a = torch.randn(200, device=self.DEVICE)
        b = torch.randn(200, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_warpgroup_size_2d_128x128(self):
        """Test 2D tensor with 128x128 elements."""

        def fn(x, y):
            return x + y

        compiled = self._compile(fn)

        # 128x128 = 16384 elements, multiple of 128
        x = torch.randn(128, 128, device=self.DEVICE)
        y = torch.randn(128, 128, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_warpgroup_size_small_tensor(self):
        """Test with very small tensor (less than warpgroup size).

        Mosaic GPU requires tensor sizes to be multiples of 128 (WARPGROUP_SIZE).
        Automatic padding aligns tensors to 128 for GPU.
        """

        def fn(a, b):
            return a + b

        compiled = self._compile(fn)

        # Small tensor, automatic padding aligns to 128
        a = torch.randn(64, device=self.DEVICE)
        b = torch.randn(64, device=self.DEVICE)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_warpgroup_size_2d_non_aligned_10x10(self):
        """Test 2D tensor with 10x10 = 100 elements (not multiple of 128).

        Mosaic GPU requires total tensor size to be multiples of 128.
        Automatic padding aligns tensors to 128 for GPU.
        """

        def fn(x, y):
            return x + y

        compiled = self._compile(fn)

        # 10x10 = 100 elements, automatic padding aligns to 128
        x = torch.randn(10, 10, device=self.DEVICE)
        y = torch.randn(10, 10, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_warpgroup_size_2d_non_aligned_15x15(self):
        """Test 2D tensor with 15x15 = 225 elements (not multiple of 128).

        Mosaic GPU requires total tensor size to be multiples of 128.
        Automatic padding aligns tensors to 256 for GPU.
        """

        def fn(x, y):
            return x * y + x

        compiled = self._compile(fn)

        # 15x15 = 225 elements, automatic padding aligns to 256
        x = torch.randn(15, 15, device=self.DEVICE)
        y = torch.randn(15, 15, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_warpgroup_size_3d_non_aligned(self):
        """Test 3D tensor with non-aligned size (5x5x5 = 125 elements).

        Mosaic GPU requires total tensor size to be multiples of 128.
        Automatic padding aligns tensors to 128 for GPU.
        """

        def fn(x, y):
            return x + y

        compiled = self._compile(fn)

        # 5x5x5 = 125 elements, automatic padding aligns to 128
        x = torch.randn(5, 5, 5, device=self.DEVICE)
        y = torch.randn(5, 5, 5, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_warpgroup_size_3d_aligned(self):
        """Test 3D tensor with aligned size (4x4x8 = 128 elements)."""

        def fn(x, y):
            return x + y

        compiled = self._compile(fn)

        # 4x4x8 = 128 elements, exactly 1 warpgroup
        x = torch.randn(4, 4, 8, device=self.DEVICE)
        y = torch.randn(4, 4, 8, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_warpgroup_size_2d_non_aligned_7x19(self):
        """Test 2D tensor with 7x19 = 133 elements (not multiple of 128).

        Mosaic GPU requires tensor sizes to be multiples of 128 (WARPGROUP_SIZE).
        The backend automatically pads inputs and unpads outputs to handle this.
        """

        def fn(x, y):
            return x - y

        compiled = self._compile(fn)

        # 7x19 = 133 elements, just above 128 but not aligned
        x = torch.randn(7, 19, device=self.DEVICE)
        y = torch.randn(7, 19, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_warpgroup_size_4d_non_aligned(self):
        """Test 4D tensor with non-aligned size (2x3x4x5 = 120 elements).

        Mosaic GPU requires tensor sizes to be multiples of 128 (WARPGROUP_SIZE).
        The backend automatically pads inputs and unpads outputs to handle this.
        """

        def fn(x, y):
            return x + y

        compiled = self._compile(fn)

        # 2x3x4x5 = 120 elements, not a multiple of 128
        x = torch.randn(2, 3, 4, 5, device=self.DEVICE)
        y = torch.randn(2, 3, 4, 5, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_warpgroup_size_4d_aligned(self):
        """Test 4D tensor with aligned size (2x2x4x16 = 256 elements)."""

        def fn(x, y):
            return x * y

        compiled = self._compile(fn)

        # 2x2x4x16 = 256 elements, multiple of 128
        x = torch.randn(2, 2, 4, 16, device=self.DEVICE)
        y = torch.randn(2, 2, 4, 16, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_warpgroup_size_2d_aligned_16x8(self):
        """Test 2D tensor with 16x8 = 128 elements (exactly 1 warpgroup)."""

        def fn(x, y):
            return x + y * 2

        compiled = self._compile(fn)

        # 16x8 = 128 elements, exactly 1 warpgroup
        x = torch.randn(16, 8, device=self.DEVICE)
        y = torch.randn(16, 8, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_warpgroup_size_2d_aligned_32x8(self):
        """Test 2D tensor with 32x8 = 256 elements (2 warpgroups)."""

        def fn(x, y):
            return x + y

        compiled = self._compile(fn)

        # 32x8 = 256 elements, 2 warpgroups
        x = torch.randn(32, 8, device=self.DEVICE)
        y = torch.randn(32, 8, device=self.DEVICE)
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)


if test_torchinductor.RUN_CPU and has_cpu_pallas():

    class PallasTestsCPU(PallasTestsMixin, TestCase):
        DEVICE = "cpu"

    make_pallas(test_torchinductor.SweepInputsCpuTest)
    make_pallas(test_torchinductor.CpuTests)


if test_torchinductor.RUN_GPU and has_cuda_pallas():

    class PallasTestsCUDA(PallasTestsMixin, TestCase):
        DEVICE = "cuda"

    make_pallas(test_torchinductor.SweepInputsGPUTest)
    # make_pallas(test_torchinductor.GPUTests)

if test_torchinductor.RUN_TPU and has_tpu_pallas():

    @config.patch({"_debug_cpu_to_tpu_pallas": True})
    class PallasTestsTPU(PallasTestsMixin, TestCase):
        DEVICE = "cpu"

        @mock.patch("torch._inductor.codegen.pallas.has_tpu_pallas", return_value=False)
        def test_tpu_not_available_raises_error(self, mock_has_tpu_pallas):
            def fn(a, b):
                return a + b

            with self.assertRaisesRegex(
                RuntimeError,
                (
                    "PALLAS_TARGET_TPU is set, but no TPU device was found. "
                    "Please make sure that you have a TPU available and that JAX is configured correctly."
                ),
            ):
                torch.compile(
                    fn, backend="inductor", options={"cpu_backend": "pallas"}
                )(torch.randn(16), torch.randn(16))


if __name__ == "__main__":
    run_tests(needs="filelock")
