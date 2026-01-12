# Elementwise Addition External Triton Library

This directory contains CUDA device functions for elementwise tensor addition that can be linked with Triton kernels via the `extern_elementwise` mechanism.

## Overview

The library provides:
- **Scalar implementations**: Process one element at a time (fp16, fp32, fp64)
- **Vectorized implementations**: Use vector types (float2, float4, half2) for better throughput
- **Bit-representation versions**: Useful for PTX integration

## Files

- `elementwise_add.cu` - CUDA device functions
- `elementwise_add.bc` - Compiled LLVM bitcode (generated)
- `Makefile` - Build script

## Building

### Using Make (recommended)

```bash
# Build for default architecture (sm_80/Ampere)
make

# Build for specific architecture
make CUDA_ARCH=sm_90  # Hopper
make CUDA_ARCH=sm_70  # Volta

# Clean build artifacts
make clean
```

### Manual compilation

```bash
clang++ -x cuda --cuda-device-only -emit-llvm -c elementwise_add.cu \
        -o elementwise_add.bc --cuda-gpu-arch=sm_80 -O3 -I/usr/local/cuda/include
```

## Usage in Triton

### 1. Import the library wrapper

```python
from torch._extern_triton import requires_elementwise_add_lib, scalar_add_f32
```

### 2. Decorate your kernel

```python
@requires_elementwise_add_lib
@triton.jit
def my_add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    # Use the external CUDA kernel
    result = scalar_add_f32(a, b)

    tl.store(out_ptr + offsets, result, mask=mask)
```

### 3. Launch the kernel

```python
import torch
import triton

size = 1024
a = torch.randn(size, device='cuda', dtype=torch.float32)
b = torch.randn(size, device='cuda', dtype=torch.float32)
output = torch.empty_like(a)

grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
my_add_kernel[grid](a, b, output, size, BLOCK_SIZE=256)
```

## Available Functions

### Scalar Operations (recommended for Triton)

| Function | Input Types | Description |
|----------|-------------|-------------|
| `scalar_add_f32(a, b)` | fp32, fp32 | Float32 addition |
| `scalar_add_f16(a, b)` | fp16, fp16 | Float16 addition |
| `scalar_add_f64(a, b)` | fp64, fp64 | Float64 addition |

### Vectorized Operations (in CUDA)

| Function | Input Types | Description |
|----------|-------------|-------------|
| `vectorized_add_f32x4` | float4, float4 | 4-way float32 addition |
| `vectorized_add_f32x2` | float2, float2 | 2-way float32 addition |
| `vectorized_add_f16x2` | half2, half2 | 2-way float16 addition |

### Bonus Operations (in CUDA)

| Function | Description |
|----------|-------------|
| `scalar_fma_f32` | Fused multiply-add: a * b + c |
| `vectorized_fma_f32x4` | 4-way fused multiply-add |

## Environment Variables

- `ELEMENTWISE_ADD_LIB_PATH`: Custom path to the compiled `.bc` file

## Testing

```bash
# Build the bitcode first
cd torch/csrc/_extern_triton && make

# Run tests
python test/distributed/test_elementwise_add_triton.py
```

## Architecture Support

| Architecture | GPU Series | Command |
|--------------|------------|---------|
| sm_70 | Volta (V100) | `make sm_70` |
| sm_75 | Turing (T4) | `make sm_75` |
| sm_80 | Ampere (A100) | `make sm_80` |
| sm_86 | Ampere (A10, A40) | `make sm_86` |
| sm_89 | Ada Lovelace (L40) | `make sm_89` |
| sm_90 | Hopper (H100) | `make sm_90` |

## How It Works

1. **CUDA Compilation**: The `.cu` file is compiled to LLVM bitcode (`.bc`) using clang
2. **Library Registration**: The `@requires_elementwise_add_lib` decorator registers the bitcode with Triton
3. **Extern Linkage**: When Triton compiles the kernel, it links against the bitcode
4. **Function Dispatch**: `core.extern_elementwise` maps Triton types to CUDA function names
