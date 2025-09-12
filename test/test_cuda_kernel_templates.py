"""
Tests for CUDA kernel compilation with C++ template support.
"""

import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
class TestCudaKernelTemplates(TestCase):
    """Tests for C++ template support in torch.cuda._compile_kernel"""

    def test_simple_template_kernel(self):
        """Test compilation of a simple templated kernel."""
        from torch.cuda._compile_kernel_with_templates import (
            _compile_kernel_with_templates,
        )

        # Define a simple template kernel
        template_code = """
        template<typename T>
        __global__ void add_template(T* a, T* b, T* c, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                c[i] = a[i] + b[i];
            }
        }
        """

        # Compile for float
        add_float = _compile_kernel_with_templates(
            template_code,
            "add_template",
            is_template=True,
            template_types=["float"],
            wrapper_signature="float* a, float* b, float* c, int n",
            wrapper_body="    add_template<float>(a, b, c, n);",
        )

        # Test the kernel
        n = 1024
        a = torch.rand(n, device="cuda", dtype=torch.float32)
        b = torch.rand(n, device="cuda", dtype=torch.float32)
        c = torch.empty_like(a)

        threads = 256
        blocks = (n + threads - 1) // threads

        add_float(grid=(blocks, 1, 1), block=(threads, 1, 1), args=[a, b, c, n])

        expected = a + b
        self.assertTrue(torch.allclose(c, expected))

    def test_multi_type_template(self):
        """Test template instantiation with different types."""
        from torch.cuda._compile_kernel_with_templates import (
            _compile_kernel_with_templates,
        )

        template_code = """
        template<typename T>
        __global__ void scale_template(T* data, T scalar, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                data[i] *= scalar;
            }
        }
        """

        # Compile for double
        scale_double = _compile_kernel_with_templates(
            template_code,
            "scale_template",
            is_template=True,
            template_types=["double"],
            wrapper_signature="double* data, double scalar, int n",
            wrapper_body="    scale_template<double>(data, scalar, n);",
        )

        # Test with double precision
        n = 512
        data = torch.rand(n, device="cuda", dtype=torch.float64)
        original = data.clone()
        scalar = 2.5

        threads = 256
        blocks = (n + threads - 1) // threads

        scale_double(grid=(blocks, 1, 1), block=(threads, 1, 1), args=[data, scalar, n])

        expected = original * scalar
        self.assertTrue(torch.allclose(data, expected))

    def test_multiple_template_parameters(self):
        """Test templates with multiple type parameters."""
        from torch.cuda._compile_kernel_with_templates import (
            _compile_kernel_with_templates,
        )

        template_code = """
        template<typename T1, typename T2, typename TOut>
        __global__ void mixed_types(T1* a, T2* b, TOut* c, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                c[i] = static_cast<TOut>(a[i]) + static_cast<TOut>(b[i]);
            }
        }
        """

        # Compile with mixed types
        mixed_kernel = _compile_kernel_with_templates(
            template_code,
            "mixed_types",
            is_template=True,
            template_types=["float", "double", "float"],
            wrapper_signature="float* a, double* b, float* c, int n",
            wrapper_body="    mixed_types<float, double, float>(a, b, c, n);",
        )

        # Test mixed precision
        n = 256
        a = torch.rand(n, device="cuda", dtype=torch.float32)
        b = torch.rand(n, device="cuda", dtype=torch.float64)
        c = torch.empty(n, device="cuda", dtype=torch.float32)

        threads = 128
        blocks = (n + threads - 1) // threads

        mixed_kernel(grid=(blocks, 1, 1), block=(threads, 1, 1), args=[a, b, c, n])

        # Verify result (b will be cast to float)
        expected = a + b.float()
        self.assertTrue(torch.allclose(c, expected, rtol=1e-5))

    def test_cutlass_style_gemm(self):
        """Test CUTLASS-style GEMM kernel compilation."""
        from torch.cuda._compile_kernel_with_templates import _compile_kernel_with_templates
        
        # Simple GEMM template without CUTLASS headers
        gemm_template = """
        template<typename T>
        __global__ void gemm_kernel(
            T const* A,
            T const* B,
            T* C,
            int M, int N, int K,
            T alpha,
            T beta
        ) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (row < M && col < N) {
                T sum = 0;
                for (int i = 0; i < K; ++i) {
                    sum += A[row * K + i] * B[i * N + col];
                }
                C[row * N + col] = alpha * sum + beta * C[row * N + col];
            }
        }
        """
        
        # Compile GEMM kernel
        gemm_kernel = _compile_kernel_with_templates(
            gemm_template,
            "gemm_kernel",
            is_template=True,
            template_types=["float"],
            wrapper_signature="float const* A, float const* B, float* C, int M, int N, int K, float alpha, float beta",
            wrapper_body="    gemm_kernel<float>(A, B, C, M, N, K, alpha, beta);"
        )

        # Create test matrices
        M, N, K = 64, 64, 32
        A = torch.rand((M, K), device="cuda", dtype=torch.float32)
        B = torch.rand((K, N), device="cuda", dtype=torch.float32)
        C = torch.zeros((M, N), device="cuda", dtype=torch.float32)

        alpha = 1.0
        beta = 0.0

        # Launch kernel
        block_dim = (16, 16, 1)
        grid_dim = (
            (N + block_dim[0] - 1) // block_dim[0],
            (M + block_dim[1] - 1) // block_dim[1],
            1,
        )

        gemm_kernel(
            grid=grid_dim, block=block_dim, args=[A, B, C, M, N, K, alpha, beta]
        )

        # Verify against PyTorch matmul
        expected = torch.matmul(A, B)
        
        # Check if results are close with relaxed tolerance for simple GEMM
        if not torch.allclose(C, expected, rtol=1e-3, atol=1e-3):
            max_diff = torch.max(torch.abs(C - expected)).item()
            print(f"Max difference: {max_diff}")
            print(f"C sample: {C[:2, :2]}")
            print(f"Expected sample: {expected[:2, :2]}")
        
        self.assertTrue(torch.allclose(C, expected, rtol=1e-3, atol=1e-3))

    def test_template_with_shared_memory(self):
        """Test template kernel using shared memory."""
        from torch.cuda._compile_kernel_with_templates import (
            _compile_kernel_with_templates,
        )

        template_code = """
        template<typename T, int BLOCK_SIZE>
        __global__ void reduction_template(T* input, T* output, int n) {
            extern __shared__ T sdata[];

            int tid = threadIdx.x;
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            // Load to shared memory
            sdata[tid] = (idx < n) ? input[idx] : 0;
            __syncthreads();

            // Reduction in shared memory
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }

            // Write result
            if (tid == 0) {
                output[blockIdx.x] = sdata[0];
            }
        }
        """

        # Compile with template parameters
        reduction_kernel = _compile_kernel_with_templates(
            template_code,
            "reduction_template",
            is_template=True,
            template_types=["float", "256"],  # Type and block size
            wrapper_signature="float* input, float* output, int n",
            wrapper_body="    reduction_template<float, 256>(input, output, n);",
        )

        # Test reduction
        n = 1024
        block_size = 256
        num_blocks = (n + block_size - 1) // block_size

        input_data = torch.ones(n, device="cuda", dtype=torch.float32)
        output = torch.zeros(num_blocks, device="cuda", dtype=torch.float32)

        reduction_kernel(
            grid=(num_blocks, 1, 1),
            block=(block_size, 1, 1),
            args=[input_data, output, n],
            shared_mem=block_size * 4,  # float is 4 bytes
        )

        # Each block should sum block_size elements (or less for last block)
        expected_sum = torch.sum(input_data).item()
        actual_sum = torch.sum(output).item()
        self.assertAlmostEqual(actual_sum, expected_sum, places=4)

    def test_template_with_cuda_headers(self):
        """Test template kernel that uses CUDA headers."""
        from torch.cuda._compile_kernel_with_templates import (
            _compile_kernel_with_templates,
        )

        template_code = """
        #ifndef __HIPCC__
        #include <cuda_fp16.h>
        #endif

        template<typename T>
        __global__ void convert_kernel(float* input, T* output, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                #ifndef __HIPCC__
                if constexpr (sizeof(T) == 2) {
                    output[i] = __float2half(input[i]);
                } else {
                    output[i] = (T)input[i];
                }
                #else
                output[i] = (T)input[i];
                #endif
            }
        }
        """

        # Compile for half precision  
        convert_half = _compile_kernel_with_templates(
            template_code,
            "convert_kernel", 
            is_template=True,
            template_types=["__half"],
            wrapper_signature="float* input, __half* output, int n",
            wrapper_body="    convert_kernel<__half>(input, output, n);",
        )

        # Test conversion
        n = 512
        input_data = torch.rand(n, device="cuda", dtype=torch.float32)
        output = torch.empty(n, device="cuda", dtype=torch.float16)

        threads = 256
        blocks = (n + threads - 1) // threads

        convert_half(
            grid=(blocks, 1, 1), block=(threads, 1, 1), args=[input_data, output, n]
        )

        # Verify conversion
        expected = input_data.half()
        self.assertTrue(torch.allclose(output, expected))

    def test_backward_compatibility(self):
        """Test that non-templated kernels still work."""
        from torch.cuda._compile_kernel_with_templates import (
            _compile_kernel_with_templates,
        )

        # Regular non-templated kernel
        kernel_code = """
        extern "C"
        __global__ void simple_add(float* a, float* b, float* c, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                c[i] = a[i] + b[i];
            }
        }
        """

        # Should work without template parameters
        add_kernel = _compile_kernel_with_templates(
            kernel_code,
            "simple_add",
            is_template=False,  # Explicitly non-templated
        )

        # Test
        n = 256
        a = torch.rand(n, device="cuda")
        b = torch.rand(n, device="cuda")
        c = torch.empty_like(a)

        add_kernel(grid=(1, 1, 1), block=(256, 1, 1), args=[a, b, c, n])

        expected = a + b
        self.assertTrue(torch.allclose(c, expected))


if __name__ == "__main__":
    run_tests()
