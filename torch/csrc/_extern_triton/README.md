# External Triton Libraries

This directory contains CUDA device functions that can be linked with Triton kernels via the `extern_elementwise` mechanism.

## Overview

The library provides:
- **Symmetric Memory Primitives**: Unified all-reduce and other collective operations with NCCL/NVSHMEM backend dispatch

## Files

### Core Libraries
- `torch_symm.cu` - Unified symmetric memory primitives with backend dispatch
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

# Build only torch_symm
make torch_symm.bc

# Clean build artifacts
make clean
```

### Manual compilation

```bash
clang++ -x cuda --cuda-device-only -emit-llvm -c torch_symm.cu \
        -o torch_symm.bc --cuda-gpu-arch=sm_80 -O3 -I/usr/local/cuda/include
```

---

# Symmetric Memory Library

## Overview

The symmetric memory library provides a unified frontend that automatically dispatches to either NCCL or NVSHMEM backend based on the SymmContext type.

**DEMONSTRATION ONLY**: The `symm_all_reduce` kernel implementation is intentionally simple and NOT efficient. It is provided solely to demonstrate the symmetric memory abstraction layer API. This implementation should NOT be used as a reference for production kernels and is NOT part of the proposed set of kernels that constitute the symmetric memory abstraction layer.

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

The unified `symm_all_reduce` function dispatches based on context type:

```cpp
__device__ int32_t symm_all_reduce(
    int64_t ctx_ptr,      // Pointer to SymmContext
    int64_t local_ptr,    // Pointer to local buffer
    int32_t byte_offset,  // Byte offset within symmetric buffer
    int32_t num_elements, // Number of elements
    int32_t reduce_op,    // Reduction operation (0=SUM, only SUM supported)
    int32_t dtype         // Data type (0=float32, only float32 supported)
);
// Returns: 0 on success
//          -1 for null context
//          -2 for unknown type
//          -3 for unsupported reduce_op
//          -4 for unsupported dtype
```

### Constants

```cpp
// Reduction operations
#define REDUCE_OP_SUM 0    // Sum reduction (only supported value)

// Data types
#define DTYPE_FLOAT32 0    // float32 (only supported value)
```

## Usage in Triton

### Recommended Pattern

Use a factory function to create kernels with a specific backend hint:

```python
from torch._extern_triton import (
    BACKEND_DEFAULT,
    BACKEND_NVSHMEM,
    DTYPE_FLOAT32,
    REDUCE_OP_SUM,
    requires_torch_symm,
    symm_all_reduce,
)

def make_my_kernel(backend: int):
    @requires_torch_symm(backend=backend)
    @triton.jit
    def my_kernel(
        ctx_ptr,
        buffer_ptr,
        num_elements: tl.constexpr,
        backend_hint: tl.constexpr,
    ):
        byte_offset: tl.int64 = 0
        n_elems: tl.int64 = num_elements
        result = symm_all_reduce(
            ctx_ptr, buffer_ptr, byte_offset, n_elems,
            REDUCE_OP_SUM, DTYPE_FLOAT32, backend_hint
        )
    return my_kernel

# Create kernel variants
my_kernel_dynamic = make_my_kernel(BACKEND_DEFAULT)   # Runtime dispatch
my_kernel_nvshmem = make_my_kernel(BACKEND_NVSHMEM)   # Direct NVSHMEM

# Launch with matching backend hint
my_kernel_nvshmem[(1,)](ctx_ptr, buf_ptr, n_elems, BACKEND_NVSHMEM)
```

## Environment Variables

- `TORCH_SYMM_LIB_PATH`: Custom path to torch_symm.bc
- `NVSHMEM_LIB_DIR`: Directory containing libnvshmem_device.bc

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
python test/distributed/test_torch_symm_triton.py
```

## How It Works

1. **CUDA Compilation**: `.cu` files are compiled to LLVM bitcode (`.bc`) using clang
2. **Library Registration**: Decorators (`@requires_torch_symm`) register bitcode with Triton
3. **Extern Linkage**: Triton links against the bitcode during kernel compilation
4. **Function Dispatch**: `core.extern_elementwise` maps Triton types to CUDA functions
