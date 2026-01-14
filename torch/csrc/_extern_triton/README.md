# External Triton Libraries

This directory contains CUDA device functions that can be linked with Triton kernels via the `extern_elementwise` mechanism.

## Overview

The library provides:
- **Elementwise operations**: Simple tensor addition (fp16, fp32, fp64)
- **Symmetric All-Reduce**: Unified all-reduce with NCCL/NVSHMEM backend dispatch

## Files

### Core Libraries
- `elementwise_add.cu` - CUDA device functions for elementwise ops
- `symm_all_reduce.cu` - Unified symmetric all-reduce with backend dispatch
- `symm_comm.cuh` - Unified header with SymmContext, NCCLSymmContext, NVSHMEMSymmContext

### Legacy (backward compatibility)
- `nccl_symm_comm.cuh` - Wrapper that includes symm_comm.cuh
- `nccl_symm_comm.hpp` - Host-side NCCL communicator management
- `nccl_symm_comm.cpp` - Host-side implementation

## Building

### Using Make (recommended)

```bash
# Build for default architecture (sm_80/Ampere)
make

# Build for specific architecture
make CUDA_ARCH=sm_90  # Hopper
make CUDA_ARCH=sm_70  # Volta

# Build only symm_all_reduce
make symm_all_reduce.bc

# Clean build artifacts
make clean
```

### Manual compilation

```bash
clang++ -x cuda --cuda-device-only -emit-llvm -c symm_all_reduce.cu \
        -o symm_all_reduce.bc --cuda-gpu-arch=sm_80 -O3 -I/usr/local/cuda/include
```

---

# Symmetric All-Reduce Library

## Overview

The symmetric all-reduce library provides a unified frontend that automatically dispatches to either NCCL or NVSHMEM backend based on the SymmContext type.

### Backend Support

| Backend | Status | Device Library |
|---------|--------|----------------|
| NVSHMEM | ✅ Functional | libnvshmem_device.bc |
| NCCL | ❌ Not functional | libnccl_device.bc (not provided by NCCL) |

**Note**: Currently only NVSHMEM backend is functional because NCCL does not ship a device bitcode library.

## Architecture

### Context Types

```cpp
// Base class for runtime type dispatch
struct SymmContext {
  enum class Type { NCCL = 0, NVSHMEM = 1 };
  Type type;
  int32_t rank;
  int32_t world_size;
};

// NCCL-specific context (for future use)
struct NCCLSymmContext : public SymmContext {
  ncclWindow_t buffer_window;
  ncclWindow_t signal_window;
  ncclDevComm* dev_comm;
  void* local_buffer;
  size_t buffer_size;
  int32_t device_idx;
};

// NVSHMEM-specific context (functional)
struct NVSHMEMSymmContext : public SymmContext {
  void* local_buffer;  // Symmetric memory address
  size_t buffer_size;
  int32_t device_idx;
  size_t offset;
};
```

### Frontend Function

The unified `symm_all_reduce_sum_f32` function dispatches based on context type:

```cpp
__device__ int32_t symm_all_reduce_sum_f32(
    int64_t ctx_ptr,      // Pointer to SymmContext
    int64_t local_ptr,    // Pointer to local buffer
    int64_t byte_offset,  // Byte offset within symmetric buffer
    int64_t num_elements  // Number of float32 elements
);
// Returns: 0 on success, -1 for null context, -2 for unknown type
```

## Usage in Triton

### 1. Import the unified library

```python
from torch._extern_triton import (
    requires_symm_all_reduce,
    symm_all_reduce_sum_f32,
)
```

### 2. Decorate your kernel

```python
@requires_symm_all_reduce
@triton.jit
def my_allreduce_kernel(ctx_ptr, buffer_ptr, byte_offset, num_elements):
    # The frontend dispatches to NCCL or NVSHMEM based on ctx_ptr->type
    result = symm_all_reduce_sum_f32(ctx_ptr, buffer_ptr, byte_offset, num_elements)
```

### Backend-Specific Wrappers

If you know the backend at compile time, you can use backend-specific wrappers:

```python
from torch._extern_triton import (
    nccl_symm_all_reduce_sum_f32,    # For NCCLSymmContext
    nvshmem_symm_all_reduce_sum_f32, # For NVSHMEMSymmContext
)
```

## Environment Variables

- `SYMM_ALL_REDUCE_LIB_PATH`: Custom path to symm_all_reduce.bc
- `NVSHMEM_LIB_DIR`: Directory containing libnvshmem_device.bc

---

# Elementwise Addition Library

## Overview

Simple elementwise tensor addition operations compiled to LLVM bitcode.

## Available Functions

| Function | Input Types | Description |
|----------|-------------|-------------|
| `scalar_add_f32(a, b)` | fp32, fp32 | Float32 addition |
| `scalar_add_f16(a, b)` | fp16, fp16 | Float16 addition |
| `scalar_add_f64(a, b)` | fp64, fp64 | Float64 addition |

## Usage

```python
from torch._extern_triton import requires_elementwise_add_lib, scalar_add_f32

@requires_elementwise_add_lib
@triton.jit
def my_add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    result = scalar_add_f32(a, b)
    tl.store(out_ptr + offsets, result, mask=mask)
```

## Environment Variables

- `ELEMENTWISE_ADD_LIB_PATH`: Custom path to elementwise_add.bc

---

## Architecture Support

| Architecture | GPU Series | Command |
|--------------|------------|---------|
| sm_70 | Volta (V100) | `make sm_70` |
| sm_75 | Turing (T4) | `make sm_75` |
| sm_80 | Ampere (A100) | `make sm_80` |
| sm_86 | Ampere (A10, A40) | `make sm_86` |
| sm_89 | Ada Lovelace (L40) | `make sm_89` |
| sm_90 | Hopper (H100) | `make sm_90` |

## Testing

```bash
# Build the bitcode first
cd torch/csrc/_extern_triton && make

# Run tests
python test/distributed/test_symm_all_reduce_triton.py
python test/distributed/test_elementwise_add_triton.py
```

## How It Works

1. **CUDA Compilation**: `.cu` files are compiled to LLVM bitcode (`.bc`) using clang
2. **Library Registration**: Decorators (`@requires_*`) register bitcode with Triton
3. **Extern Linkage**: Triton links against the bitcode during kernel compilation
4. **Function Dispatch**: `core.extern_elementwise` maps Triton types to CUDA functions
