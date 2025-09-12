# C++ Template Support for torch.cuda._compile_kernel()

## Problem
NVRTC cannot compile C++ templates directly, preventing use of CUTLASS device API and other templated CUDA libraries.

## Solution
Wrap templates in `extern "C"` functions by generating explicit instantiation and C-linkage wrappers.

## Usage

### Basic Template
```python
template_code = '''
template<typename T>
__global__ void vector_add(T* a, T* b, T* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
'''

add_kernel = torch.cuda._compile_kernel(
    template_code,
    "vector_add",
    is_template=True,
    template_types=["float"],
    wrapper_signature="float* a, float* b, float* c, int n",
    wrapper_body="    vector_add<float>(a, b, c, n);"
)
```

### CUTLASS GEMM
```python
cutlass_gemm = '''
template<typename T>
__global__ void gemm_kernel(T const* A, T const* B, T* C,
                            int M, int N, int K, T alpha, T beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        T sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}
'''

gemm = torch.cuda._compile_kernel(
    cutlass_gemm, "gemm_kernel",
    is_template=True,
    template_types=["float"],
    wrapper_signature="float const* A, float const* B, float* C, int M, int N, int K, float alpha, float beta",
    wrapper_body="    gemm_kernel<float>(A, B, C, M, N, K, alpha, beta);"
)
```

### Multiple Template Parameters
```python
template_code = '''
template<typename T, int BLOCK_SIZE>
__global__ void optimized_kernel(T* data, int n) {
    __shared__ T shared[BLOCK_SIZE];
}
'''

kernel = torch.cuda._compile_kernel(
    template_code, "optimized_kernel",
    is_template=True,
    template_types=["float", "256"],
    wrapper_signature="float* data, int n",
    wrapper_body="    optimized_kernel<float, 256>(data, n);"
)
```

## New Parameters
- `is_template`: Enable template mode
- `template_types`: List of types to instantiate
- `wrapper_signature`: C function parameters
- `wrapper_body`: Template invocation code
- `wrapper_name`: Optional wrapper function name

## Files
- `torch/cuda/__init__.py` - Enhanced _compile_kernel
- `torch/cuda/_template_utils.py` - Template utilities
- `torch/cuda/_compile_kernel_with_templates.py` - Advanced features
- `test/test_cuda_kernel_templates.py` - Test suite