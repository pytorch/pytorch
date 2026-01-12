# Owner(s): ["oncall: distributed"]
"""
Test suite for the elementwise addition Triton extern library.

This test suite verifies that the CUDA kernels exposed via core.extern_elementwise
mechanism work correctly when called from Triton kernels.

To run:
    python test/distributed/test_elementwise_add_triton.py

Prerequisites:
    1. Build the CUDA library to bitcode:
       cd torch/csrc/_extern_triton && make CUDA_ARCH=sm_80
    2. Or set ELEMENTWISE_ADD_LIB_PATH environment variable to point to the .bc file
"""

import sys

# Import TEST_WITH_ROCM first to check for ROCm before importing CUDA-specific modules
from torch.testing._internal.common_utils import TEST_WITH_ROCM
from torch.testing._internal.inductor_utils import requires_triton


# Skip entire module on ROCm before importing CUDA-specific modules
if TEST_WITH_ROCM:
    print("Elementwise add extern library not available on ROCm, skipping tests")
    sys.exit(0)


import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TestCase,
)


# Skip if CUDA is not available
if not TEST_CUDA:
    print("CUDA not available, skipping tests")
    sys.exit(0)


# Import Triton and our extern library
try:
    import triton
    import triton.language as tl

    from torch._extern_triton import (
        requires_elementwise_add_lib,
        scalar_add_f16,
        scalar_add_f32,
        scalar_add_f64,
    )
    from torch._extern_triton._elementwise_add_triton import ElementwiseAddLibFinder

    TRITON_AVAILABLE = True
except ImportError as e:
    print(f"Triton not available, skipping tests: {e}")
    TRITON_AVAILABLE = False

# Check if the bitcode library is available
LIB_AVAILABLE = False
LIB_PATH = None
if TRITON_AVAILABLE:
    try:
        LIB_PATH = ElementwiseAddLibFinder.find_device_library()
        LIB_AVAILABLE = True
    except RuntimeError as e:
        print(f"Elementwise add library not found: {e}")


def requires_extern_lib():
    """Skip test if the extern library is not available."""
    return skip_but_pass_in_sandcastle_if(
        not LIB_AVAILABLE,
        "Elementwise add bitcode library not available. "
        "Compile with: cd torch/csrc/_extern_triton && make",
    )


# So that tests are written in device-agnostic way
device_type = "cuda"


# =============================================================================
# TRITON KERNELS FOR TESTING
# =============================================================================


if TRITON_AVAILABLE and LIB_AVAILABLE:

    @requires_elementwise_add_lib
    @triton.jit
    def add_kernel_f32(
        a_ptr,
        b_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Triton kernel that uses the external scalar_add_f32 function.
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load inputs
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)

        # Use external CUDA kernel for addition
        result = scalar_add_f32(a, b)

        # Store result
        tl.store(output_ptr + offsets, result, mask=mask)

    @requires_elementwise_add_lib
    @triton.jit
    def add_kernel_f16(
        a_ptr,
        b_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Triton kernel that uses the external scalar_add_f16 function.
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)

        result = scalar_add_f16(a, b)

        tl.store(output_ptr + offsets, result, mask=mask)

    @requires_elementwise_add_lib
    @triton.jit
    def add_kernel_f64(
        a_ptr,
        b_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Triton kernel that uses the external scalar_add_f64 function.
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)

        result = scalar_add_f64(a, b)

        tl.store(output_ptr + offsets, result, mask=mask)

    @requires_elementwise_add_lib
    @triton.jit
    def composite_kernel_f32(
        a_ptr,
        b_ptr,
        c_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Triton kernel that chains multiple extern add operations.
        Computes: output = (a + b) + c
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        c = tl.load(c_ptr + offsets, mask=mask)

        # Chain two extern add operations
        temp = scalar_add_f32(a, b)
        result = scalar_add_f32(temp, c)

        tl.store(output_ptr + offsets, result, mask=mask)

    @requires_elementwise_add_lib
    @triton.jit
    def mixed_native_extern_kernel_f32(
        a_ptr,
        b_ptr,
        c_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Triton kernel that mixes native Triton operations with extern calls.
        Computes: output = (a + b) * c, where a + b uses extern, * uses native
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        c = tl.load(c_ptr + offsets, mask=mask)

        # Use extern for addition
        sum_ab = scalar_add_f32(a, b)
        # Use native Triton for multiplication
        result = sum_ab * c

        tl.store(output_ptr + offsets, result, mask=mask)


# =============================================================================
# TEST CLASS
# =============================================================================


@instantiate_parametrized_tests
class TestElementwiseAddTriton(TestCase):
    """Test suite for elementwise addition extern library."""

    @requires_triton()
    @requires_extern_lib()
    def test_scalar_add_f32_basic(self):
        """Test basic float32 addition with simple values."""
        size = 1024
        a = torch.randn(size, device=device_type, dtype=torch.float32)
        b = torch.randn(size, device=device_type, dtype=torch.float32)
        output = torch.empty_like(a)

        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
        add_kernel_f32[grid](a, b, output, size, BLOCK_SIZE=256)

        expected = a + b
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    # @requires_triton()
    # @requires_extern_lib()
    # def test_scalar_add_f16_basic(self):
    #     """Test basic float16 addition with simple values."""
    #     size = 1024
    #     a = torch.randn(size, device=device_type, dtype=torch.float16)
    #     b = torch.randn(size, device=device_type, dtype=torch.float16)
    #     output = torch.empty_like(a)

    #     grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
    #     add_kernel_f16[grid](a, b, output, size, BLOCK_SIZE=256)

    #     expected = a + b
    #     torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)

    @requires_triton()
    @requires_extern_lib()
    def test_scalar_add_f64_basic(self):
        """Test basic float64 addition with simple values."""
        size = 1024
        a = torch.randn(size, device=device_type, dtype=torch.float64)
        b = torch.randn(size, device=device_type, dtype=torch.float64)
        output = torch.empty_like(a)

        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
        add_kernel_f64[grid](a, b, output, size, BLOCK_SIZE=256)

        expected = a + b
        torch.testing.assert_close(output, expected, rtol=1e-10, atol=1e-10)

    @requires_triton()
    @requires_extern_lib()
    @parametrize("size", [1, 7, 256, 1024, 4096, 65536])
    def test_scalar_add_f32_various_sizes(self, size):
        """Test float32 addition with various tensor sizes."""
        a = torch.randn(size, device=device_type, dtype=torch.float32)
        b = torch.randn(size, device=device_type, dtype=torch.float32)
        output = torch.empty_like(a)

        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
        add_kernel_f32[grid](a, b, output, size, BLOCK_SIZE=256)

        expected = a + b
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    @requires_triton()
    @requires_extern_lib()
    @parametrize("block_size", [32, 64, 128, 256, 512, 1024])
    def test_scalar_add_f32_various_block_sizes(self, block_size):
        """Test float32 addition with various block sizes."""
        size = 4096
        a = torch.randn(size, device=device_type, dtype=torch.float32)
        b = torch.randn(size, device=device_type, dtype=torch.float32)
        output = torch.empty_like(a)

        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
        add_kernel_f32[grid](a, b, output, size, BLOCK_SIZE=block_size)

        expected = a + b
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    @requires_triton()
    @requires_extern_lib()
    def test_scalar_add_f32_zeros(self):
        """Test float32 addition with zero tensors."""
        size = 1024
        a = torch.zeros(size, device=device_type, dtype=torch.float32)
        b = torch.randn(size, device=device_type, dtype=torch.float32)
        output = torch.empty_like(a)

        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
        add_kernel_f32[grid](a, b, output, size, BLOCK_SIZE=256)

        expected = a + b
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    @requires_triton()
    @requires_extern_lib()
    def test_scalar_add_f32_ones(self):
        """Test float32 addition with ones tensors."""
        size = 1024
        a = torch.ones(size, device=device_type, dtype=torch.float32)
        b = torch.ones(size, device=device_type, dtype=torch.float32)
        output = torch.empty_like(a)

        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
        add_kernel_f32[grid](a, b, output, size, BLOCK_SIZE=256)

        expected = torch.full((size,), 2.0, device=device_type, dtype=torch.float32)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    @requires_triton()
    @requires_extern_lib()
    def test_scalar_add_f32_large_values(self):
        """Test float32 addition with large values."""
        size = 1024
        a = torch.full((size,), 1e30, device=device_type, dtype=torch.float32)
        b = torch.full((size,), 1e30, device=device_type, dtype=torch.float32)
        output = torch.empty_like(a)

        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
        add_kernel_f32[grid](a, b, output, size, BLOCK_SIZE=256)

        expected = a + b
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    @requires_triton()
    @requires_extern_lib()
    def test_scalar_add_f32_negative_values(self):
        """Test float32 addition with negative values."""
        size = 1024
        a = -torch.randn(size, device=device_type, dtype=torch.float32).abs()
        b = -torch.randn(size, device=device_type, dtype=torch.float32).abs()
        output = torch.empty_like(a)

        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
        add_kernel_f32[grid](a, b, output, size, BLOCK_SIZE=256)

        expected = a + b
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    @requires_triton()
    @requires_extern_lib()
    def test_composite_kernel_f32(self):
        """Test chaining multiple extern add operations."""
        size = 1024
        a = torch.randn(size, device=device_type, dtype=torch.float32)
        b = torch.randn(size, device=device_type, dtype=torch.float32)
        c = torch.randn(size, device=device_type, dtype=torch.float32)
        output = torch.empty_like(a)

        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
        composite_kernel_f32[grid](a, b, c, output, size, BLOCK_SIZE=256)

        expected = (a + b) + c
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    @requires_triton()
    @requires_extern_lib()
    def test_mixed_native_extern_kernel_f32(self):
        """Test mixing native Triton operations with extern calls."""
        size = 1024
        a = torch.randn(size, device=device_type, dtype=torch.float32)
        b = torch.randn(size, device=device_type, dtype=torch.float32)
        c = torch.randn(size, device=device_type, dtype=torch.float32)
        output = torch.empty_like(a)

        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
        mixed_native_extern_kernel_f32[grid](a, b, c, output, size, BLOCK_SIZE=256)

        expected = (a + b) * c
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    @requires_triton()
    @requires_extern_lib()
    def test_scalar_add_f32_2d_tensor(self):
        """Test float32 addition with 2D tensors."""
        rows, cols = 32, 64
        a = torch.randn(rows, cols, device=device_type, dtype=torch.float32)
        b = torch.randn(rows, cols, device=device_type, dtype=torch.float32)
        output = torch.empty_like(a)

        size = a.numel()
        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
        add_kernel_f32[grid](
            a.view(-1), b.view(-1), output.view(-1), size, BLOCK_SIZE=256
        )

        expected = a + b
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    @requires_triton()
    @requires_extern_lib()
    def test_scalar_add_f32_3d_tensor(self):
        """Test float32 addition with 3D tensors."""
        shape = (8, 16, 32)
        a = torch.randn(shape, device=device_type, dtype=torch.float32)
        b = torch.randn(shape, device=device_type, dtype=torch.float32)
        output = torch.empty_like(a)

        size = a.numel()
        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
        add_kernel_f32[grid](
            a.view(-1), b.view(-1), output.view(-1), size, BLOCK_SIZE=256
        )

        expected = a + b
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    @requires_triton()
    @requires_extern_lib()
    def test_scalar_add_f32_contiguous(self):
        """Test float32 addition with contiguous tensors."""
        size = 1024
        a = torch.randn(size, device=device_type, dtype=torch.float32).contiguous()
        b = torch.randn(size, device=device_type, dtype=torch.float32).contiguous()
        output = torch.empty_like(a)

        self.assertTrue(a.is_contiguous())
        self.assertTrue(b.is_contiguous())

        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
        add_kernel_f32[grid](a, b, output, size, BLOCK_SIZE=256)

        expected = a + b
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    @requires_triton()
    @requires_extern_lib()
    def test_multiple_kernel_launches(self):
        """Test multiple kernel launches in sequence."""
        size = 1024
        a = torch.randn(size, device=device_type, dtype=torch.float32)
        b = torch.randn(size, device=device_type, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)

        # Launch multiple times
        for _ in range(5):
            output = torch.empty_like(a)
            add_kernel_f32[grid](a, b, output, size, BLOCK_SIZE=256)
            expected = a + b
            torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    @requires_triton()
    @requires_extern_lib()
    def test_inplace_like_operation(self):
        """Test using output tensor that overlaps with input (copy first)."""
        size = 1024
        a = torch.randn(size, device=device_type, dtype=torch.float32)
        b = torch.randn(size, device=device_type, dtype=torch.float32)
        expected = a + b

        # Copy a to output, then add b
        output = a.clone()

        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
        add_kernel_f32[grid](output, b, output, size, BLOCK_SIZE=256)

        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


class TestElementwiseAddLibFinder(TestCase):
    """Test suite for the library finder utility."""

    @requires_triton()
    def test_finder_caches_path(self):
        """Test that the library finder caches the found path."""
        if not LIB_AVAILABLE:
            self.skipTest("Library not available")

        # Clear cache
        ElementwiseAddLibFinder.found_device_lib_path = None

        # First call should search
        path1 = ElementwiseAddLibFinder.find_device_library()
        self.assertIsNotNone(path1)
        self.assertTrue(path1.endswith(".bc"))

        # Second call should return cached
        path2 = ElementwiseAddLibFinder.find_device_library()
        self.assertEqual(path1, path2)

    @requires_triton()
    def test_finder_respects_env_var(self):
        """Test that the library finder respects ELEMENTWISE_ADD_LIB_PATH."""
        import os

        # Clear cache
        ElementwiseAddLibFinder.found_device_lib_path = None

        # Set a fake path
        original_env = os.environ.get("ELEMENTWISE_ADD_LIB_PATH")
        try:
            os.environ["ELEMENTWISE_ADD_LIB_PATH"] = "/nonexistent/path.bc"

            with self.assertRaises(RuntimeError) as context:
                ElementwiseAddLibFinder.find_device_library()

            self.assertIn("not found", str(context.exception))
        finally:
            # Restore original environment
            if original_env is not None:
                os.environ["ELEMENTWISE_ADD_LIB_PATH"] = original_env
            else:
                os.environ.pop("ELEMENTWISE_ADD_LIB_PATH", None)

            # Clear cache again
            ElementwiseAddLibFinder.found_device_lib_path = None


if __name__ == "__main__":
    run_tests()
