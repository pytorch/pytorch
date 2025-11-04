# Owner(s): ["oncall: pt2"]
import functools
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
        (config, "cuda_backend", "pallas"),
        xfail_prop="_expected_failure_pallas",
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


@unittest.skipUnless(HAS_PALLAS, "requires jax and pallas")
class PallasTests(TestCase):
    """Basic tests for Pallas backend functionality."""

    def test_simple_add(self):
        """Test basic element-wise addition."""

        def fn(a, b):
            return a + b

        compiled = torch.compile(
            fn, backend="inductor", options={"cuda_backend": "pallas"}
        )

        a = torch.randn(1024, device="cuda")
        b = torch.randn(1024, device="cuda")
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_simple_mul(self):
        """Test basic element-wise multiplication."""

        def fn(a, b):
            return a * b

        compiled = torch.compile(
            fn, backend="inductor", options={"cuda_backend": "pallas"}
        )

        a = torch.randn(1024, device="cuda")
        b = torch.randn(1024, device="cuda")
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result, expected)

    def test_sin(self):
        """Test sin operation."""

        def fn(x):
            return torch.sin(x)

        compiled = torch.compile(
            fn, backend="inductor", options={"cuda_backend": "pallas"}
        )

        x = torch.randn(1024, device="cuda")
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_fused_ops(self):
        """Test fused operations (sin + add)."""

        def fn(x, y):
            return x.sin() + y

        compiled = torch.compile(
            fn, backend="inductor", options={"cuda_backend": "pallas"}
        )

        x = torch.randn(1024, device="cuda")
        y = torch.randn(1024, device="cuda")
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_exp_log(self):
        """Test exp and log operations."""

        def fn(x):
            return torch.log(torch.exp(x))

        compiled = torch.compile(
            fn, backend="inductor", options={"cuda_backend": "pallas"}
        )

        x = torch.randn(1024, device="cuda")
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_sqrt(self):
        """Test sqrt operation."""

        def fn(x):
            return torch.sqrt(x)

        compiled = torch.compile(
            fn, backend="inductor", options={"cuda_backend": "pallas"}
        )

        x = torch.randn(1024, device="cuda").abs()  # Ensure positive for sqrt
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_tanh(self):
        """Test tanh operation."""

        def fn(x):
            return torch.tanh(x)

        compiled = torch.compile(
            fn, backend="inductor", options={"cuda_backend": "pallas"}
        )

        x = torch.randn(1024, device="cuda")
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_abs_neg(self):
        """Test abs and neg operations."""

        def fn(x):
            return torch.abs(-x)

        compiled = torch.compile(
            fn, backend="inductor", options={"cuda_backend": "pallas"}
        )

        x = torch.randn(1024, device="cuda")
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)

    def test_maximum_minimum(self):
        """Test maximum and minimum operations."""

        def fn(a, b):
            return torch.maximum(a, b) + torch.minimum(a, b)

        compiled = torch.compile(
            fn, backend="inductor", options={"cuda_backend": "pallas"}
        )

        a = torch.randn(1024, device="cuda")
        b = torch.randn(1024, device="cuda")
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
            options={"cuda_backend": "pallas"},
        )
        def pallas_fn(a, b):
            return a.sin() + b.cos()

        _, (code,) = run_and_get_code(
            pallas_fn,
            torch.randn(64, device="cuda"),
            torch.randn(64, device="cuda"),
        )
        # Verify Pallas-specific code generation
        self.assertIn("import jax", code)
        self.assertIn("import jax.numpy as jnp", code)
        self.assertIn("from jax.experimental import pallas as pl", code)

    def test_2d_tensor(self):
        """Test with 2D tensors (though current implementation flattens)."""

        def fn(x, y):
            return x + y

        compiled = torch.compile(
            fn, backend="inductor", options={"cuda_backend": "pallas"}
        )

        x = torch.randn(32, 32, device="cuda")
        y = torch.randn(32, 32, device="cuda")
        result = compiled(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_different_shapes(self):
        """Test with different tensor shapes."""

        def fn(x):
            return x * 2.0

        compiled = torch.compile(
            fn, backend="inductor", options={"cuda_backend": "pallas"}
        )

        for shape in [(64,), (128,), (256,), (1024,)]:
            x = torch.randn(shape, device="cuda")
            result = compiled(x)
            expected = fn(x)
            self.assertEqual(result, expected)

    def test_contiguous_index_validation(self):
        """Test that contiguous index validation works correctly end-to-end."""

        # Test 1: Contiguous operations should work
        def contiguous_add(a, b):
            return a + b

        compiled = torch.compile(
            contiguous_add, backend="inductor", options={"cuda_backend": "pallas"}
        )

        a = torch.randn(1024, device="cuda")
        b = torch.randn(1024, device="cuda")
        result = compiled(a, b)
        expected = contiguous_add(a, b)
        self.assertEqual(result, expected)

        # Test 2: Operations on contiguous tensors should work
        def contiguous_mul(x):
            return x * 2.0

        compiled = torch.compile(
            contiguous_mul, backend="inductor", options={"cuda_backend": "pallas"}
        )

        x = torch.randn(128, 8, device="cuda")
        result = compiled(x)
        expected = contiguous_mul(x)
        self.assertEqual(result, expected)

        # Test 3: Non-contiguous views will fail at runtime with JAX/Pallas
        # This demonstrates that the Pallas backend requires contiguous memory layout
        def operate_on_tensor(x):
            return x.sin()

        compiled = torch.compile(
            operate_on_tensor, backend="inductor", options={"cuda_backend": "pallas"}
        )

        # Create a transposed (non-contiguous) view
        x = torch.randn(64, 32, device="cuda")
        x_t = x.t()  # Non-contiguous view
        self.assertFalse(x_t.is_contiguous())

        # This will fail because JAX/Pallas cannot handle non-contiguous layout via DLPack
        # The error indicates that our contiguous-only approach is correct
        with self.assertRaises((RuntimeError, Exception)) as cm:
            result = compiled(x_t)

        # Verify the error is related to layout/contiguous issues
        error_msg = str(cm.exception)
        self.assertTrue(
            "layout" in error_msg.lower()
            or "contiguous" in error_msg.lower()
            or "non-default" in error_msg.lower(),
            f"Expected layout/contiguous error, got: {error_msg}",
        )

        # But if we make it contiguous first, it should work
        x_t_contiguous = x_t.contiguous()
        self.assertTrue(x_t_contiguous.is_contiguous())
        result = compiled(x_t_contiguous)
        expected = operate_on_tensor(x_t_contiguous)
        self.assertEqual(result, expected)


# Create test variants using the main test suite
# Note: Only enable GPU tests since Pallas primarily targets GPU
if test_torchinductor.HAS_GPU and HAS_PALLAS:
    # Uncomment these to run full test suite with Pallas backend
    # make_pallas(test_torchinductor.SweepInputsGPUTest)
    # make_pallas(test_torchinductor.GPUTests)
    pass

if __name__ == "__main__":
    if HAS_PALLAS:
        run_tests(needs="filelock")
