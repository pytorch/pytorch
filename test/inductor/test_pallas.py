# Owner(s): ["oncall: pt2"]
import functools
import os
import re
import sys
import unittest

import torch
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
    """Create a test class variant that uses Pallas backend.

    Args:
        cls: The test class to create a Pallas variant of.
    """
    patches = [
        (config, "cpu_backend", "pallas"),
        (config, "cuda_backend", "pallas"),
    ]
    cls_prefix = "Pallas"
    suffix = "_pallas"

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
        *patches,
        xfail_prop="_expected_failure_pallas",
        decorator=skip_decorator,
    )

    # Pallas does not support float64 or int64
    test_class._unsupported_input_gen_types = {"double"}

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


def _skip_if(condition_fn, reason):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if condition_fn(self):
                self.skipTest(reason)
            fn(self, *args, **kwargs)

        return wrapper

    return decorator


skip_if_tpu = _skip_if(lambda self: self.DEVICE == "tpu", "Not yet working on TPU")
skip_if_cpu = _skip_if(lambda self: self.DEVICE == "cpu", "Not yet working on CPU")
skip_if_cuda = _skip_if(lambda self: self.DEVICE == "cuda", "Not yet working on GPU")


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
        device_to_backend_key = {
            "cuda": "cuda_backend",
            "cpu": "cpu_backend",
            "tpu": "tpu_backend",
        }
        key = device_to_backend_key[self.DEVICE]
        return torch.compile(
            fn, backend="inductor", options={key: "pallas"}, dynamic=False
        )

    def test_simple_add(self):
        """Test basic element-wise addition."""

        def fn(a, b):
            return a + b

        shapes = [(1024,)]
        if self.DEVICE != "cuda":
            shapes += [(2048,), (2048, 128)]
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                a = torch.randn(shape, device=self.DEVICE)
                b = torch.randn(shape, device=self.DEVICE)
                result = compiled(a, b)
                expected = fn(a, b)
                self.assertEqual(result, expected)

    def test_simple_mul(self):
        """Test basic element-wise multiplication."""

        def fn(a, b):
            return a * b

        shapes = [(1024,)]
        if self.DEVICE != "cuda":
            shapes += [(2048,), (2048, 128)]
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                a = torch.randn(shape, device=self.DEVICE)
                b = torch.randn(shape, device=self.DEVICE)
                result = compiled(a, b)
                expected = fn(a, b)
                self.assertEqual(result, expected)

    def test_sin(self):
        """Test sin operation."""

        def fn(x):
            return torch.sin(x)

        shapes = [(1024,)]
        if self.DEVICE != "cuda":
            shapes.append((2048,))
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                x = torch.randn(shape, device=self.DEVICE)
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
            self.skipTest("sqrt primitive not implemented in Pallas Mosaic GPU")

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

        pallas_fn = self._compile(lambda a, b: a.sin() + b.cos())

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

        pallas_fn = self._compile(lambda a, b: a + b)

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
        """Test with 2D tensors."""

        def fn(x, y):
            return x + y

        shapes = [(32, 32)]
        if self.DEVICE != "cuda":
            shapes.append((16, 2048))
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                x = torch.randn(shape, device=self.DEVICE)
                y = torch.randn(shape, device=self.DEVICE)
                result = compiled(x, y)
                expected = fn(x, y)
                self.assertEqual(result, expected)

    @skip_if_tpu
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

    @skip_if_cuda
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
        x = torch.randn(128, 128, device=self.DEVICE)
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

    @skip_if_tpu
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

    @skip_if_tpu
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
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_1d = torch.arange(256, dtype=torch.float32, device=self.DEVICE)
        for x in [base_1d[::2], base_1d[::4], base_1d[::2][::2]]:
            self.assertFalse(x.is_contiguous())
            self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_2d_row_stride(self):
        """Test 2D row-strided input patterns."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_2d = torch.randn(32, 32, device=self.DEVICE)
        x = base_2d[::2, :]  # (16, 32) with stride (64, 1)
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_2d_col_stride(self):
        """Test 2D col-strided input patterns."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_2d = torch.randn(32, 32, device=self.DEVICE)
        x = base_2d[:, ::2]  # (32, 16) with stride (32, 2)
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_2d_both_stride(self):
        """Test 2D both-strided input patterns."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_2d = torch.randn(32, 32, device=self.DEVICE)
        x = base_2d[::2, ::2]  # (16, 16) with stride (64, 2)
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    @skip_if_cuda
    def test_stride_non_contiguous_2d_transpose(self):
        """Test 2D transposed input patterns."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        for rows, cols in [(32, 32), (2048, 2048)]:
            with self.subTest(rows=rows, cols=cols):
                base_2d = torch.randn(rows, cols, device=self.DEVICE)
                x = base_2d.t()
                self.assertFalse(x.is_contiguous())
                self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_3d(self):
        """Test 3D non-contiguous input patterns."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_3d = torch.randn(8, 8, 8, device=self.DEVICE)
        x = base_3d[::2, ::2, ::2]
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    @skip_if_cuda
    def test_stride_non_contiguous_permuted(self):
        """Test permuted non-contiguous input patterns."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_3d = torch.randn(8, 8, 8, device=self.DEVICE)
        x = base_3d.permute(2, 0, 1)
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    @skip_if_cuda
    def test_stride_non_contiguous_channels_last(self):
        """Test channels-last (NHWC) non-contiguous input patterns."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        x = torch.randn(2, 3, 4, 5, device=self.DEVICE).to(
            memory_format=torch.channels_last
        )
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_diagonal(self):
        """Test diagonal (large stride) non-contiguous input patterns."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_2d = torch.randn(32, 32, device=self.DEVICE)
        x = base_2d.diagonal()
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_as_strided(self):
        """Test as_strided (custom layout) non-contiguous input patterns."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_flat = torch.randn(256, device=self.DEVICE)
        x = torch.as_strided(base_flat, size=(4, 8), stride=(16, 2))
        self.assertFalse(x.is_contiguous())
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_select_stride(self):
        """Test select then stride on non-contiguous input patterns."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_2d = torch.randn(32, 32, device=self.DEVICE)
        x = base_2d[3, ::2]
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    def test_stride_non_contiguous_unsqueeze(self):
        """Test unsqueeze on strided non-contiguous input patterns."""
        compiled = self._compile(lambda x: x * 2.0 + 1.0)

        base_2d = torch.randn(32, 32, device=self.DEVICE)
        x = base_2d[::2, ::2].unsqueeze(0)
        self.assertEqual(compiled(x), x * 2.0 + 1.0)

    @skip_if_tpu
    def test_stride_non_contiguous_dtypes(self):
        """Test non-contiguous patterns with various dtypes."""
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

    @skip_if_tpu
    def test_stride_multiple_inputs(self):
        """Test multiple strided inputs and broadcasting."""
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
        """Test that non-power-of-2 tensor sizes work correctly.

        On GPU (Mosaic backend), TMA automatically handles OOB masking for
        non-power-of-2 sizes. On CPU/TPU, masked loads/stores are used.
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

    @skip_if_tpu
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

    @skip_if_tpu
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

    @skip_if_tpu
    def test_complex64_mul(self):
        """Test complex64 multiplication."""

        def fn(a, b):
            return a * b

        sizes = [128]
        if self.DEVICE != "cuda":
            sizes.append(2048)
        for size in sizes:
            with self.subTest(size=size):
                compiled = self._compile(fn)
                a = torch.randn(size, dtype=torch.complex64, device=self.DEVICE)
                b = torch.randn(size, dtype=torch.complex64, device=self.DEVICE)
                result = compiled(a, b)
                expected = fn(a, b)
                self.assertEqual(result, expected)

    @skip_if_tpu
    def test_complex_conj(self):
        """Test complex conjugate."""

        def fn(x):
            return torch.conj(x)

        compiled = self._compile(fn)

        x = torch.randn(128, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_tpu
    def test_complex_real(self):
        """Test extracting real part of complex tensor."""

        def fn(x):
            return torch.real(x)

        compiled = self._compile(fn)

        x = torch.randn(128, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_tpu
    def test_complex_imag(self):
        """Test extracting imaginary part of complex tensor."""

        def fn(x):
            return torch.imag(x)

        compiled = self._compile(fn)

        x = torch.randn(128, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_tpu
    def test_complex_abs(self):
        """Test complex absolute value (magnitude)."""

        def fn(x):
            return torch.abs(x)

        compiled = self._compile(fn)

        x = torch.randn(128, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_tpu
    def test_complex128_conj(self):
        """Test complex128 conjugate operation."""

        def fn(x):
            return torch.conj(x)

        compiled = self._compile(fn)

        x = torch.randn(128, dtype=torch.complex128, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_tpu
    def test_complex_mul_scalar(self):
        """Test complex multiplication with scalar."""

        def fn(x):
            return x * 2.5

        compiled = self._compile(fn)

        x = torch.randn(128, dtype=torch.complex64, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_tpu
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

        shapes = [(1024,)]
        if self.DEVICE != "cuda":
            shapes += [(2048,), (2048, 128)]
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                x = torch.randn(shape, device=self.DEVICE)
                y = torch.randn(shape, device=self.DEVICE)
                result = compiled(x, y)
                expected = fn(x, y)
                self.assertEqual(result, expected)

    def test_clamp(self):
        """Test torch.clamp operation."""

        def fn(x):
            return torch.clamp(x, -1.0, 1.0)

        shapes = [(1024,)]
        if self.DEVICE != "cuda":
            shapes += [(2048,), (2048, 128)]
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                x = torch.randn(shape, device=self.DEVICE) * 2
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

        shapes = [(1024,)]
        if self.DEVICE != "cuda":
            shapes += [(2048,), (2048, 128)]
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                a = torch.randn(shape, device=self.DEVICE)
                b = torch.randn(shape, device=self.DEVICE)
                result = compiled(a, b)
                expected = fn(a, b)
                self.assertEqual(result, expected)

    def test_logical_ops(self):
        """Test logical operations."""

        def fn(a, b):
            return torch.logical_and(a > 0, b > 0).float()

        shapes = [(1024,)]
        if self.DEVICE != "cuda":
            shapes += [(2048,), (2048, 128)]
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                a = torch.randn(shape, device=self.DEVICE)
                b = torch.randn(shape, device=self.DEVICE)
                result = compiled(a, b)
                expected = fn(a, b)
                self.assertEqual(result, expected)

    @skip_if_tpu
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
        if self.DEVICE == "cuda":
            self.skipTest("integer_pow primitive not implemented in Pallas Mosaic GPU")

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

    @skip_if_tpu
    def test_erf(self):
        """Test erf operation."""

        def fn(x):
            return torch.erf(x)

        compiled = self._compile(fn)

        x = torch.randn(16, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_tpu
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

        shapes = [(1024,)]
        if self.DEVICE != "cuda":
            shapes.append((2048,))
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                x = torch.randn(shape, device=self.DEVICE)
                result = compiled(x)
                expected = fn(x)
                self.assertEqual(result, expected)

    def test_max_reduction(self):
        """Test max reduction."""

        def fn(x):
            return x.max()

        shapes = [(1024,)]
        if self.DEVICE != "cuda":
            shapes.append((2048,))
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                x = torch.randn(shape, device=self.DEVICE)
                result = compiled(x)
                expected = fn(x)
                self.assertEqual(result, expected)

    def test_min_reduction(self):
        """Test min reduction."""

        def fn(x):
            return x.min()

        shapes = [(16,)]
        if self.DEVICE != "cuda":
            shapes.append((2048,))
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                x = torch.randn(shape, device=self.DEVICE)
                result = compiled(x)
                expected = fn(x)
                self.assertEqual(result, expected)

    @skip_if_tpu
    def test_prod_reduction(self):
        """Test prod reduction."""
        if self.DEVICE == "cuda":
            self.skipTest("reduce_prod primitive not implemented in Pallas Mosaic GPU")

        def fn(x):
            # Use smaller values to avoid overflow
            return (x * 0.1).prod()

        shapes = [(16,)]
        if self.DEVICE != "cuda":
            shapes.append((2048,))
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                x = torch.randn(shape, device=self.DEVICE)
                result = compiled(x)
                expected = fn(x)
                self.assertEqual(result, expected)

    @skip_if_cuda
    def test_softmax_two_pass(self):
        """Test two-pass softmax (max reduction + sum reduction)."""

        for shape in [(32, 64), (2048, 64)]:
            with self.subTest(shape=shape):
                torch._dynamo.reset()

                def fn(x):
                    return torch.softmax(x, dim=-1)

                compiled = self._compile(fn)
                x = torch.randn(shape, device=self.DEVICE)
                result = compiled(x)
                expected = fn(x)
                self.assertEqual(result, expected)

    @skip_if_cuda
    def test_rms_norm(self):
        """Test RMS normalization (mean-of-squares reduction + rsqrt)."""

        for rows, cols in [(32, 64), (2048, 64)]:
            with self.subTest(rows=rows, cols=cols):
                torch._dynamo.reset()

                def fn(x, weight):
                    variance = x.pow(2).mean(-1, keepdim=True)
                    x = x * torch.rsqrt(variance + 1e-6)
                    return x * weight

                compiled = self._compile(fn)
                x = torch.randn(rows, cols, device=self.DEVICE)
                weight = torch.randn(cols, device=self.DEVICE)
                result = compiled(x, weight)
                expected = fn(x, weight)
                self.assertEqual(result, expected)

    @skip_if_cuda
    @skip_if_tpu
    def test_welford(self):
        """Test Welford variance/mean computation (two-pass fallback)."""

        for shape in [(32, 64), (2048, 64)]:
            with self.subTest(shape=shape):
                torch._dynamo.reset()

                def fn(x):
                    return torch.var_mean(x, dim=-1, keepdim=True)

                compiled = self._compile(fn)
                x = torch.randn(shape, device=self.DEVICE)
                var_result, mean_result = compiled(x)
                var_expected, mean_expected = fn(x)
                self.assertEqual(mean_result, mean_expected)
                self.assertEqual(var_result, var_expected)

    @skip_if_cuda
    def test_layer_norm(self):
        """Test layer normalization (mean + variance reduction, normalize, scale + shift)."""

        for rows, cols in [(32, 64), (2048, 64)]:
            with self.subTest(rows=rows, cols=cols):
                torch._dynamo.reset()

                def fn(x, weight, bias):
                    mean = x.mean(-1, keepdim=True)
                    variance = (x - mean).pow(2).mean(-1, keepdim=True)
                    x = (x - mean) * torch.rsqrt(variance + 1e-6)
                    return x * weight + bias

                compiled = self._compile(fn)
                x = torch.randn(rows, cols, device=self.DEVICE)
                weight = torch.randn(cols, device=self.DEVICE)
                bias = torch.randn(cols, device=self.DEVICE)
                result = compiled(x, weight, bias)
                expected = fn(x, weight, bias)
                self.assertEqual(result, expected)

    @skip_if_cuda
    @skip_if_tpu
    def test_rope(self):
        """Test Rotary Position Embedding with slice + cat.

        Splits input into halves, applies cos/sin rotation, and concatenates
        back. Exercises non-contiguous output aliases via torch.cat.
        """

        def fn(x, cos, sin):
            d = x.shape[-1]
            x1 = x[..., : d // 2]
            x2 = x[..., d // 2 :]
            return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

        compiled = self._compile(fn)

        x = torch.randn(32, 64, device=self.DEVICE)
        cos = torch.randn(32, 32, device=self.DEVICE)
        sin = torch.randn(32, 32, device=self.DEVICE)
        result = compiled(x, cos, sin)
        expected = fn(x, cos, sin)
        self.assertEqual(result, expected)

    @skip_if_cuda
    @skip_if_tpu
    def test_rope_interleaved(self):
        """Test Rotary Position Embedding with interleaved halves.

        Uses even/odd stride-2 slicing instead of contiguous halves, then
        reassembles via stack + reshape. Exercises strided input access.
        """

        def fn(x, cos, sin):
            x1 = x[..., 0::2]
            x2 = x[..., 1::2]
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            return torch.stack([o1, o2], dim=-1).reshape_as(x)

        compiled = self._compile(fn)

        x = torch.randn(32, 64, device=self.DEVICE)
        cos = torch.randn(32, 32, device=self.DEVICE)
        sin = torch.randn(32, 32, device=self.DEVICE)
        result = compiled(x, cos, sin)
        expected = fn(x, cos, sin)
        self.assertEqual(result, expected)

    @skip_if_cuda
    @skip_if_tpu
    def test_chained_stride_slice(self):
        """Test that chained stride slices compose into a single strided access.

        x[:, 1::2][:, 2::3][:, 3::4] should compose to x[:, 23::24].
        """

        def fn(x):
            return x[:, 1::2][:, 2::3][:, 3::4] + 1

        compiled = self._compile(fn)

        # last dim = 480 which is divisible by 24
        x = torch.randn(4, 480, device=self.DEVICE)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    @skip_if_tpu
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

        sizes = [128]
        if self.DEVICE != "cuda":
            sizes.append(2048)
        for size in sizes:
            with self.subTest(size=size):
                compiled = self._compile(fn)
                x = torch.randn(size, device=self.DEVICE, dtype=torch.float32)
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

    @skip_if_tpu
    def test_warpgroup_size_2d_128x128(self):
        """Test 2D tensor with 128x128 and tiling-exercising sizes."""

        def fn(x, y):
            return x + y

        shapes = [(128, 128)]
        if self.DEVICE != "cuda":
            shapes.append((2048, 2048))
        for shape in shapes:
            with self.subTest(shape=shape):
                compiled = self._compile(fn)
                x = torch.randn(shape, device=self.DEVICE)
                y = torch.randn(shape, device=self.DEVICE)
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

    def test_simple_mlp(self):
        def fn(w_0, w_1, w_2, x):
            x = (x @ w_0).relu()
            x = (x @ w_1).relu()
            x = (x @ w_2).relu()
            return x

        compiled = self._compile(fn)

        ws = [torch.rand(32, 32, device=self.DEVICE) for _ in range(3)]
        x = torch.randn(32, 32, device=self.DEVICE)

        result = compiled(*ws, x)
        expected = fn(*ws, x)
        self.assertEqual(result, expected)

    @skip_if_cuda
    @skip_if_tpu
    def test_nanogpt(self):
        """Test a minimal NanoGPT-style transformer block.

        Tests the core transformer operations used in GPT-style models:
        - Single-head self-attention with causal masking
        - MLP (feed-forward) block with GELU activation
        - Residual connections

        Uses 2D tensors (seq_len, embed_dim) to match the existing test_simple_mlp
        pattern and avoid 3D tensor shape issues in Pallas codegen.

        Note: Skipped on CUDA because JAX's Mosaic GPU backend doesn't support
        axis-based reductions (reduce with axis= parameter), which is required
        for softmax. Full reductions work, but partial reductions do not.
        """
        seq_len = 32
        n_embd = 64

        def transformer_block(x, w_q, w_k, w_v, w_proj, w_fc, w_out, mask):
            T, C = x.shape

            # === Self-Attention ===
            q = x @ w_q  # (T, C)
            k = x @ w_k
            v = x @ w_v

            # Scaled dot-product attention
            scale = 1.0 / (C**0.5)
            att = (q @ k.t()) * scale  # (T, T)
            att = att + mask  # Apply causal mask
            att = torch.softmax(att, dim=-1)
            attn_out = att @ v  # (T, C)

            # Output projection + residual
            x = x + (attn_out @ w_proj)

            # === MLP (Feed-Forward) ===
            h = x @ w_fc  # (T, 4C)
            # GELU activation (tanh approximation)
            h = 0.5 * h * (1.0 + torch.tanh(0.7978845608 * (h + 0.044715 * h * h * h)))
            x = x + (h @ w_out)  # Residual + project back

            return x

        compiled = self._compile(transformer_block)

        # Initialize weights
        w_q = torch.randn(n_embd, n_embd, device=self.DEVICE) * 0.02
        w_k = torch.randn(n_embd, n_embd, device=self.DEVICE) * 0.02
        w_v = torch.randn(n_embd, n_embd, device=self.DEVICE) * 0.02
        w_proj = torch.randn(n_embd, n_embd, device=self.DEVICE) * 0.02
        w_fc = torch.randn(n_embd, 4 * n_embd, device=self.DEVICE) * 0.02
        w_out = torch.randn(4 * n_embd, n_embd, device=self.DEVICE) * 0.02

        # Causal mask
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=self.DEVICE),
            diagonal=1,
        )

        x = torch.randn(seq_len, n_embd, device=self.DEVICE)

        result = compiled(x, w_q, w_k, w_v, w_proj, w_fc, w_out, mask)
        expected = transformer_block(x, w_q, w_k, w_v, w_proj, w_fc, w_out, mask)
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
    from torch_tpu import api as tpu_api  # type: ignore[import-not-found]

    tpu_api.tpu_device()  # initialize TPU runtime

    class PallasTestsTPU(PallasTestsMixin, TestCase):
        DEVICE = "tpu"

    make_pallas(test_torchinductor.SweepInputsTpuTest)


if __name__ == "__main__":
    run_tests(needs="filelock")
