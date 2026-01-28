# _extern_triton: Symmetric Memory Abstraction Layer for Triton

## 1. Goal and Motivation

The primary goal of `_extern_triton` is to **enable developers to write Triton kernels
that leverage symmetric memory without having a direct dependency on a particular backend**
(e.g., NVSHMEM, NCCL, NCCLx).

### Problem Statement

Symmetric memory operations (one-sided communication, direct memory access across GPUs,
signaling) are increasingly important for high-performance distributed GPU computing.
However, different backends provide different APIs and programming models:

- **NVSHMEM**: PGAS model with `nvshmem_ptr()`, `nvshmemx_barrier_all_block()`, etc.
- **NCCL**: Window-based LSA API with `ncclGetLsaPointer()`, `ncclLsaBarrierSession`, etc.
- **NCCLx**: Extended NCCL with sparse operations and enhanced device-side capabilities

Kernel authors should not need to rewrite kernels for each backend, nor should they
need to maintain multiple kernel variants or manage complex build dependencies.

### Solution Approach

`_extern_triton` provides an **abstraction layer** that:

1. **Generalizes backend differences** behind a unified API
2. **Exposes frontend primitives** to Triton kernel authors
3. **Enables compile-time (static) or runtime (dynamic) dispatch** to backends
4. **Compiles to LLVM bitcode** for linking with Triton kernels

```
+-----------------------------------------------------------------------------------+
|                              TRITON KERNEL AUTHOR                                 |
+-----------------------------------------------------------------------------------+
                                      |
                                      | Uses unified primitives
                                      v
+-----------------------------------------------------------------------------------+
|                           ABSTRACTION LAYER                                       |
|  +---------------------------+  +---------------------------+                     |
|  |   Python Frontend         |  |   CUDA Frontend           |                     |
|  |   (_torch_symm_triton.py) |  |   (torch_symm.cu)         |                     |
|  +---------------------------+  +---------------------------+                     |
+-----------------------------------------------------------------------------------+
                                      |
              Dynamic dispatch        |        Static dispatch
              (runtime)               |        (compile-time)
              +-----------------------+-------------------------+
              |                       |                         |
              v                       v                         v
+-------------------+     +-------------------+     +-------------------+
|  NVSHMEM Backend  |     |   NCCL Backend    |     |  NCCLx Backend    |
|  (nvshmem_symm)   |     |   (nccl_symm)     |     |   (future)        |
+-------------------+     +-------------------+     +-------------------+
              |                       |                         |
              v                       v                         v
+-------------------+     +-------------------+     +-------------------+
| libnvshmem_device |     | libnccl_device.bc |     | libncclx_device   |
|       .bc         |     |  (not available)  |     |      .bc          |
+-------------------+     +-------------------+     +-------------------+
```

---

## 2. Primitives in the Abstraction Layer

The abstraction layer exposes the following unified primitives to kernel authors:

### 2.1 Memory Access Primitives

| Primitive                | Purpose                                                      |
|--------------------------|--------------------------------------------------------------|
| `symm_lsa_ptr`           | Get pointer to symmetric memory on a peer rank (P2P access)  |
| `symm_lsa_multicast_ptr` | Get multicast address for broadcasting to all team members   |
| `symm_lsa_signal_ptr`    | Get pointer to peer's signal pad for direct load/store       |

### 2.2 Synchronization Primitives

| Primitive                | Purpose                                                      |
|--------------------------|--------------------------------------------------------------|
| `symm_barrier`           | Barrier across all ranks in the communicator                 |
| `symm_lsa_barrier`       | Barrier within the LSA (Local Symmetric Access) domain only  |
| `symm_fence`             | Memory fence with configurable scope (CTA/GPU/system)        |
| `symm_quiet`             | Ensure all prior one-sided operations have completed         |

### 2.3 Signaling Primitives

| Primitive                | Purpose                                                      |
|--------------------------|--------------------------------------------------------------|
| `symm_signal`            | Atomically update a signal at a remote rank's signal pad     |
| `symm_signal_wait_until` | Block until a local signal meets a specified condition       |
| `symm_signal_reset`      | Reset a local signal to zero for reuse                       |

### 2.4 Data Transfer Primitives

| Primitive                | Purpose                                                      |
|--------------------------|--------------------------------------------------------------|
| `symm_put_async`         | Non-blocking one-sided put (local → remote)                  |
| `symm_put_signal_async`  | Non-blocking put with remote signal notification on arrival  |

### 2.5 Team (Topology) Primitives

| Primitive                | Purpose                                                      |
|--------------------------|--------------------------------------------------------------|
| `symm_team_size`         | Get number of ranks in the team                              |
| `symm_team_rank`         | Get this process's rank within the team                      |
| `symm_team_lsa_size`     | Get number of ranks in the LSA domain                        |
| `symm_team_lsa`          | Check if a peer is in the same LSA domain (NVLink-connected) |

### 2.6 Collective Primitives (Demonstration Only)

| Primitive                | Purpose                                                      |
|--------------------------|--------------------------------------------------------------|
| `symm_all_reduce`        | **Demo only** - Simple all-reduce (not production-ready)     |

---

## 3. Architecture Overview

### 3.1 Core Concepts

#### 3.1.1 Symmetric Context (`SymmContext`)

The **Symmetric Context** is the central data structure that encapsulates all
backend-specific state. It is created on the host and passed to device kernels.

```
+----------------------------------------------------------------------+
|                         SymmContext (Base)                           |
+----------------------------------------------------------------------+
|  Type type            <- Backend identifier (NCCL=0, NVSHMEM=1)      |
|  int32_t rank         <- Process rank in the communicator            |
|  int32_t world_size   <- Total number of processes                   |
+----------------------------------------------------------------------+
              ^                                      ^
              |                                      |
+-----------------------------+      +-----------------------------+
|      NCCLSymmContext        |      |    NVSHMEMSymmContext       |
+-----------------------------+      +-----------------------------+
| ncclWindow_t buffer_window  |      | void* local_buffer          |
| ncclWindow_t signal_window  |      | uint64_t* lsa_signal_pad    |
| ncclDevComm* dev_comm       |      | uint64_t* gin_signal_pad    |
| NCCLWindowEntry[] windows   |      | int32_t global_rank         |
| uint64_t** signal_pad_ptrs  |      | int32_t global_world_size   |
| ...                         |      | ...                         |
+-----------------------------+      +-----------------------------+
```

**Key Design Decisions:**

- Base class contains the `type` discriminator for runtime dispatch
- Derived classes contain backend-specific handles and state
- Context is passed as `int64_t` pointer through Triton's `extern_elementwise`
- Context is immutable during kernel execution

#### 3.1.2 Symmetric Team (`SymmTeam`)

The **Symmetric Team** decouples synchronization scope from memory resources and
provides topology information:

```
+----------------------------------------------------------------------+
|                          SymmTeam (Base)                             |
+----------------------------------------------------------------------+
|  Type type            <- Backend identifier (matches SymmContext)    |
|  int32_t team_size    <- Number of ranks in this team                |
|  int32_t team_rank    <- This process's rank within the team         |
|  int32_t lsa_size     <- Number of ranks in the LSA domain           |
|  int32_t lsa_base_rank<- First rank in the LSA domain                |
|  uint64_t lsa_mask    <- Bitmask for LSA membership (small teams)    |
+----------------------------------------------------------------------+
              ^                                      ^
              |                                      |
+-----------------------------+      +-----------------------------+
|       NCCLSymmTeam          |      |      NVSHMEMSymmTeam        |
+-----------------------------+      +-----------------------------+
| ncclDevComm* dev_comm       |      | nvshmem_team_t nvshmem_team |
| int32_t barrier_id          |      |                             |
+-----------------------------+      +-----------------------------+
```

**LSA (Local Symmetric Access) Domain:**

The LSA domain represents peers that can directly access each other's memory via
load/store operations (e.g., GPUs connected via NVLink on the same node). The team
tracks LSA membership to enable optimized local-only operations.

### 3.2 Kernel Structure: Frontend and Backend

Each primitive has a **frontend kernel** (in `torch_symm.cu`) and **backend kernels**
(in `nvshmem_symm.cuh`, `nccl_symm.cuh`, etc.).

```
+-------------------------------------------------------------------------+
|                    Anatomy of a Primitive                               |
+-------------------------------------------------------------------------+

   Triton Kernel (Python)
          |
          | calls extern_elementwise
          v
+-----------------------------------+
|    Python Frontend Wrapper        |  <- _torch_symm_triton.py
|    (symm_lsa_ptr)                 |     Handles backend_hint routing
+-----------------------------------+
          |
          | Static dispatch (compile-time) if backend_hint != DEFAULT
          | Otherwise calls unified frontend
          v
+-----------------------------------+
|    CUDA Frontend Function         |  <- torch_symm.cu
|    (symm_lsa_ptr)                 |     extern "C" __device__
+-----------------------------------+
          |
          | Dynamic dispatch based on ctx->type
          v
     +----+----+
     |         |
     v         v
+----------+  +----------+
|  NVSHMEM |  |   NCCL   |      <- nvshmem_symm.cuh, nccl_symm.cuh
|  Backend |  |  Backend |         Backend implementations
+----------+  +----------+
     |             |
     v             v
+----------+  +----------+
| nvshmem_ |  | nccl*()  |      <- Actual backend library calls
| *() APIs |  | LSA APIs |
+----------+  +----------+
```

### 3.3 Dispatch Mechanisms

#### 3.3.1 Dynamic Dispatch (Runtime) in Backend Kernel

The frontend CUDA functions use **runtime dispatch** based on the `SymmContext::type`
field. This is implemented via a `switch` statement:

```cpp
// torch_symm.cu - Dynamic dispatch example
__device__ int64_t symm_lsa_ptr(int64_t ctx_ptr, int64_t local_ptr, int32_t peer) {
    SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
    TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

    switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
        case SymmContext::Type::NCCL: {
            NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
            return nccl_lsa_ptr_impl(nccl_ctx, local_ptr, peer);
        }
#endif
        case SymmContext::Type::NVSHMEM: {
            NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
            return nvshmem_lsa_ptr_impl(nvshmem_ctx, local_ptr, peer);
        }
        default:
            TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
            return 0;
    }
}
```

**Trade-offs:**
- ✅ Single kernel binary works with any backend
- ✅ Backend selection at runtime based on available infrastructure
- ❌ All backend bitcode libraries must be linked
- ❌ Small runtime overhead from switch/dispatch

#### 3.3.2 Static Dispatch (Compile-Time) in Frontend Kernel

The Python frontend supports **compile-time dispatch** via a `backend_hint` parameter.
When a specific backend is chosen, only that backend's function is called:

```python
# _torch_symm_triton.py - Static dispatch routing
def symm_lsa_ptr(ctx_ptr, local_ptr, peer, backend=BACKEND_DEFAULT):
    if backend == BACKEND_NVSHMEM:
        return _nvshmem_symm_lsa_ptr(ctx_ptr, local_ptr, peer)  # Direct call
    elif backend == BACKEND_NCCL:
        return _nccl_symm_lsa_ptr(ctx_ptr, local_ptr, peer)     # Direct call
    else:
        return _symm_lsa_ptr_frontend(ctx_ptr, local_ptr, peer)  # Dynamic dispatch
```

**Trade-offs:**
- ✅ Dead code elimination removes unused backends
- ✅ Only required backend bitcode needs to be linked
- ✅ Zero runtime dispatch overhead
- ❌ Must specify backend at kernel definition time
- ❌ Separate kernel compilations for different backends

### 3.4 Bitcode Library Requirements

Triton uses LLVM bitcode (`.bc` files) to link extern device functions. Each backend
must provide a device bitcode library:

```
+-------------------------------------------------------------------------+
|                      Bitcode Library Hierarchy                          |
+-------------------------------------------------------------------------+

torch_symm.bc                    <- Unified frontend + backend implementations
    |
    +-- Includes nvshmem_symm.cuh    (NVSHMEM backend, compiled in)
    +-- Includes nccl_symm.cuh       (NCCL backend, compiled in)
    |
    |   Requires external libraries at link time:
    v
+-------------------------+     +-------------------------+
| libnvshmem_device.bc    |     | libnccl_device.bc       |
| (Provided by NVSHMEM)   |     | (NOT provided by NCCL)  |
| ✅ Available            |     | ❌ Not available        |
+-------------------------+     +-------------------------+
```

#### 3.4.1 Library Lookup Mechanism

The `TorchSymmLibFinder` class locates bitcode libraries at runtime:

```python
class TorchSymmLibFinder:
    """Utility class for finding torch symmetric memory bitcode library."""

    @classmethod
    def find_device_library(cls) -> str:
        """
        Search order:
        1. TORCH_SYMM_LIB_PATH environment variable
        2. torch/lib/torch_symm.bc (installed location)
        3. torch/csrc/_extern_triton/torch_symm.bc (source location)
        """
        ...

    @classmethod
    def find_nvshmem_device_library(cls) -> str:
        """Finds libnvshmem_device.bc via NvshmemLibFinder."""
        ...
```

The `@requires_torch_symm` decorator automatically links the required libraries:

```python
@requires_torch_symm(backend=BACKEND_NVSHMEM)
@triton.jit
def my_kernel(...):
    ...  # Uses symm_* primitives

# Under the hood:
# extern_libs = {
#     "torch_symm": "/path/to/torch_symm.bc",
#     "libnvshmem_device": "/path/to/libnvshmem_device.bc"
# }
```

### 3.5 Complete Data Flow

```
+-------------------------------------------------------------------------+
|                         COMPLETE DATA FLOW                              |
+-------------------------------------------------------------------------+

[1] HOST SETUP
    +------------------+
    | Create context:  |
    | NVSHMEMSymmContext or NCCLSymmContext
    | - Register windows (NCCL) or get symmetric addresses (NVSHMEM)
    | - Setup signal pads
    | - Copy to device memory
    +------------------+
            |
            v
[2] KERNEL DEFINITION (Python)
    +------------------+
    | @requires_torch_symm(backend=BACKEND_NVSHMEM)
    | @triton.jit
    | def kernel(ctx_ptr, data_ptr, ...):
    |     peer_ptr = symm_lsa_ptr(ctx_ptr, data_ptr, peer, BACKEND_NVSHMEM)
    |     ...
    +------------------+
            |
            | Triton compiles kernel:
            | - Resolves extern_elementwise to bitcode symbols
            | - Links torch_symm.bc + libnvshmem_device.bc
            | - Generates PTX → CUBIN
            v
[3] KERNEL COMPILATION
    +------------------+
    | LLVM IR from Triton
    |       +
    | torch_symm.bc (unified frontend)
    |       +
    | libnvshmem_device.bc (backend)
    |       =
    | Final linked LLVM module
    |       → PTX → CUBIN
    +------------------+
            |
            v
[4] KERNEL LAUNCH
    +------------------+
    | kernel[(grid,)](ctx_ptr, data_ptr, ...)
    | - ctx_ptr points to device-resident SymmContext
    | - Backend dispatch occurs on device
    +------------------+
            |
            v
[5] DEVICE EXECUTION
    +------------------+
    | symm_lsa_ptr(ctx_ptr, local_ptr, peer)
    |     -> nvshmem_lsa_ptr_impl()
    |         -> nvshmem_ptr(local_ptr, peer)
    |             -> Returns peer's GPU memory address
    +------------------+
```

---

## 4. Extending with Additional Backends

### 4.1 General Extension Process

To add a new backend (e.g., NCCLx), follow these steps:

```
+-------------------------------------------------------------------------+
|                    BACKEND EXTENSION CHECKLIST                          |
+-------------------------------------------------------------------------+

[1] Define Backend Context Structure
    +---> Create <Backend>SymmContext inheriting from SymmContext
    +---> Add backend-specific handles (device communicator, windows, etc.)
    +---> Define in symm_comm.cuh

[2] Implement Backend Kernel File
    +---> Create <backend>_symm.cuh with all _impl functions
    +---> Implement each primitive using backend's device API
    +---> Mark all functions extern "C" __device__

[3] Update Frontend Dispatch
    +---> Add new case to SymmContext::Type enum
    +---> Add dispatch case in each frontend function in torch_symm.cu
    +---> Guard with #if BACKEND_HAS_DEVICE_BITCODE

[4] Provide Bitcode Library
    +---> Backend vendor must provide lib<backend>_device.bc
    +---> Or compile backend-specific .cu to .bc yourself

[5] Update Python Frontend
    +---> Add BACKEND_<NAME> constant
    +---> Add library finder for backend's bitcode
    +---> Add static dispatch path in primitive wrappers
    +---> Update @requires_torch_symm decorator

[6] Host-Side Integration
    +---> Create Python class to build <Backend>SymmContext
    +---> Handle communicator creation, window registration, etc.
```

### 4.2 Case Study: Extending with NCCLx

NCCLx (extended NCCL) provides enhanced collective operations and debugging
capabilities. Understanding how NCCLx relates to NCCL is important for planning
future backend extensions.

#### 4.2.1 NCCLx Characteristics

NCCLx is **not a separate library** but rather NCCL built with extended features
enabled via the `IS_NCCLX` compile flag. It uses the **same device APIs** as
standard NCCL but enables additional host-side collective operations.

| Feature              | NCCL (Standard)           | NCCLx (Extended)              |
|----------------------|---------------------------|-------------------------------|
| Sparse AllReduce     | ❌ Not supported          | ✅ `ncclAllReduceSparseBlock` |
| Device Bitcode       | ❌ Not shipped            | ❌ Same as NCCL               |
| Comm Dump/Debug      | ❌ Limited                | ✅ `ncclCommDump`             |
| Symmetric Memory     | ✅ NCCL 2.27+ (host-side) | ✅ Same APIs                  |
| Device APIs          | ✅ See below              | ✅ Same APIs                  |
| Build Flag           | Default                   | `IS_NCCLX` compile flag       |

**Key Insight**: NCCLx does NOT introduce new device-side APIs. The device APIs
are provided by NCCL itself (when/if `libnccl_device.bc` becomes available).

#### 4.2.2 NCCL Device APIs (Used by Both NCCL and NCCLx)

The actual NCCL device APIs that would be used:

```cpp
// =============================================================================
// NCCL LSA (Local Symmetric Access) APIs
// =============================================================================

// Get pointer to peer's memory via LSA window
extern "C" __device__ void* ncclGetLsaPointer(
    void* window,       // ncclWindow_t handle
    size_t offset,      // Byte offset within the window
    int peer);          // Peer rank

// Get multicast pointer for broadcasting (NVSwitch required)
extern "C" __device__ void* ncclGetLsaMultimemPointer(
    void* window,       // ncclWindow_t handle
    size_t offset,      // Byte offset within the window
    void* devComm);     // ncclDevComm* device communicator

// =============================================================================
// NCCL LSA Barrier (RAII Template Class)
// =============================================================================

// Barrier synchronization using RAII pattern
template<typename Coop>  // ncclCoopCta, ncclCoopWarp, or ncclCoopThread
class ncclLsaBarrierSession {
public:
    ncclLsaBarrierSession(
        Coop coop_group,
        ncclDevComm& comm,
        ncclTeamTagLsa tag,
        uint32_t barrier_index,
        bool use_multimem);

    void arrive();                                      // Signal arrival
    void wait();                                        // Wait for all peers
    void sync(Coop coop, cuda::memory_order order);     // Atomic arrive+wait
};

// Usage example:
// ncclLsaBarrierSession<ncclCoopCta> barrier(
//     ncclCoopCta{}, *dev_comm, ncclTeamTagLsa{}, 0, false);
// barrier.sync(ncclCoopCta{}, cuda::memory_order_seq_cst);

// =============================================================================
// NCCL GIN (GPU-Initiated Networking) APIs
// =============================================================================

class ncclGin {
public:
    ncclGin(ncclDevComm& comm, int context_id);

    // Signal operations (point-to-point notification)
    void signal(ncclTeam team, int dest_rank,
                ncclGin_SignalInc{uint32_t idx});        // Increment by 1
    void signal(ncclTeam team, int dest_rank,
                ncclGin_SignalAdd{uint32_t idx, uint64_t val}); // Add value

    // Wait for signal condition
    template<typename Coop>
    void waitSignal(Coop coop, uint32_t signal_index, uint64_t cmp_value);

    // Reset signal to zero
    void resetSignal(uint32_t signal_index);

    // One-sided put operations
    void put(ncclTeam team, int dest_rank,
             void* dest_window, size_t dest_offset,
             void* src_window, size_t src_offset,
             size_t byte_count);

    // Put with signal (fused data transfer + notification)
    void put(ncclTeam team, int dest_rank,
             void* dest_window, size_t dest_offset,
             void* src_window, size_t src_offset,
             size_t byte_count,
             ncclGin_SignalInc{uint32_t idx});          // Signal after put
};

// =============================================================================
// NCCL Team APIs
// =============================================================================

ncclTeam ncclTeamWorld(ncclDevComm& comm);  // All ranks
ncclTeam ncclTeamLsa(ncclDevComm& comm);    // LSA domain only
```

#### 4.2.3 Integration Approach for NCCLx

Since NCCLx uses the same device APIs as NCCL, the integration approach is:

```
+-------------------------------------------------------------------------+
|              NCCLx Integration Architecture                             |
+-------------------------------------------------------------------------+

                    symm_comm.cuh
                         |
    +--------------------+--------------------+
    |                    |                    |
    v                    v                    v
SymmContext       NCCLSymmContext      NCCLxSymmContext (NEW)
    |                    |                    |
    |                    +--------+-----------+
    |                             |
    v                             v
torch_symm.cu              nccl_symm.cuh  <-- Shared backend implementation
    |                             |
    v                             v
(dispatch)                 libnccl_device.bc (when available from NCCL)
```

**Option A: Reuse NCCL Backend (Recommended)**

Since NCCLx uses identical device APIs, NCCLx can reuse the existing NCCL backend:

```cpp
// In symm_comm.cuh - NCCLx context extends NCCL context
struct NCCLxSymmContext : public NCCLSymmContext {
    // Additional NCCLx-specific fields for host-side features
    void* sparse_metadata;        // For ncclAllReduceSparseBlock
    int32_t ncclx_features;       // Feature flags

    __host__ __device__ NCCLxSymmContext()
        : NCCLSymmContext(),
          sparse_metadata(nullptr),
          ncclx_features(0) {
        type = Type::NCCLX;  // Override type for identification
    }
};
```

```cpp
// In torch_symm.cu - NCCLx dispatches to NCCL implementation
__device__ int64_t symm_lsa_ptr(int64_t ctx_ptr, int64_t local_ptr, int32_t peer) {
    SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);

    switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
        case SymmContext::Type::NCCL:
        case SymmContext::Type::NCCLX: {  // NCCLx uses same device APIs
            NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
            return nccl_lsa_ptr_impl(nccl_ctx, local_ptr, peer);
        }
#endif
        case SymmContext::Type::NVSHMEM: {
            NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
            return nvshmem_lsa_ptr_impl(nvshmem_ctx, local_ptr, peer);
        }
        default:
            return 0;
    }
}
```

**Option B: Separate NCCLx Backend (Future)**

If NCCLx later introduces distinct device APIs, create a separate backend:

```cpp
// ncclx_symm.cuh - Only if NCCLx introduces new device APIs

#pragma once
#include "symm_comm.cuh"

// Forward declarations for hypothetical NCCLx-specific device APIs
// (Currently these do NOT exist - NCCLx uses standard NCCL APIs)
#ifdef NCCLX_HAS_UNIQUE_DEVICE_APIS
extern "C" {
extern __device__ void* ncclxEnhancedLsaPointer(...);
extern __device__ void ncclxFastBarrier(...);
}
#endif

// Implementation would mirror nccl_symm.cuh using NCCLx-specific APIs
```

#### 4.2.4 Python Frontend Updates

```python
# _torch_symm_triton.py - Add NCCLx support

# Backend constants
_BACKEND_DEFAULT = 0
_BACKEND_NCCL = 1
_BACKEND_NVSHMEM = 2
_BACKEND_NCCLX = 3  # NEW: Treated as NCCL variant

BACKEND_NCCLX = _BACKEND_NCCLX

def requires_torch_symm(
    jit_func_or_backend: JITFunction | int | None = None,
    backend: int = _BACKEND_DEFAULT,
) -> GridCallableWithExtern | Any:

    def _apply_decorator(jit_func: JITFunction, backend_hint: int):
        extern_libs = {}
        extern_libs["torch_symm"] = TorchSymmLibFinder.find_device_library()

        # NCCL and NCCLx share the same device bitcode library
        if backend_hint in (_BACKEND_DEFAULT, _BACKEND_NCCL, _BACKEND_NCCLX):
            try:
                # Both NCCL and NCCLx use libnccl_device.bc
                extern_libs["libnccl_device"] = TorchSymmLibFinder.find_nccl_device_library()
            except RuntimeError:
                if backend_hint in (_BACKEND_NCCL, _BACKEND_NCCLX):
                    raise  # Fail if explicitly requested

        if backend_hint in (_BACKEND_DEFAULT, _BACKEND_NVSHMEM):
            extern_libs["libnvshmem_device"] = TorchSymmLibFinder.find_nvshmem_device_library()

        # ... rest of decorator logic
```

#### 4.2.5 Host-Side Context Creation

The key difference for NCCLx is on the **host side**, where additional features
like sparse allreduce are available:

```python
# _ncclx_symm_comm.py - Host-side NCCLx context management

class NCCLxSymmContextBuilder(NCCLSymmContextBuilder):
    """
    Builds NCCLxSymmContext for device kernel use.

    Extends NCCLSymmContextBuilder with NCCLx-specific features.
    The device-side APIs are identical to NCCL.
    """

    def __init__(self, comm: "NCCLCommunicator"):
        super().__init__(comm)
        self._check_ncclx_available()

    def _check_ncclx_available(self) -> None:
        """Verify NCCLx features are available (IS_NCCLX build)."""
        # Check if NCCL was built with IS_NCCLX flag
        import torch.cuda.nccl as nccl
        version = nccl.version()
        if not version[-1] == "x":
            raise RuntimeError(
                "NCCLx features require NCCL built with IS_NCCLX flag. "
                f"Current version: {version}"
            )

    def build(self, buffer: torch.Tensor) -> "NCCLxSymmContext":
        """
        Create an NCCLxSymmContext.

        Device-side operations use standard NCCL APIs.
        NCCLx-specific features (sparse ops) are host-side only.
        """
        # Use parent class to build base NCCL context
        base_ctx = super().build(buffer)

        # Wrap in NCCLx context with additional metadata
        return NCCLxSymmContext(
            base_ctx=base_ctx,
            sparse_metadata=None,  # Set if using sparse operations
            ncclx_features=self._detect_ncclx_features(),
        )

    def _detect_ncclx_features(self) -> int:
        """Detect available NCCLx features."""
        features = 0
        # Feature flags for sparse allreduce, comm dump, etc.
        return features
```

---

## 5. Summary

The `_extern_triton` abstraction layer provides:

1. **Backend Independence**: Kernel authors write against unified primitives
2. **Flexible Dispatch**: Both compile-time (static) and runtime (dynamic) dispatch
3. **Minimal Dependencies**: Only link the bitcode libraries you actually use
4. **Clear Extension Path**: Well-defined process for adding new backends

**Current Status:**

| Backend  | Context Defined | Backend Impl | Bitcode Available | Functional |
|----------|-----------------|--------------|-------------------|------------|
| NVSHMEM  | ✅              | ✅           | ✅ (shipped)      | ✅         |
| NCCL     | ✅              | ✅           | ❌ (not shipped)  | ❌         |
| NCCLx    | ❌ (proposed)   | ⚠️ Reuses NCCL | ❌ (same as NCCL) | ❌         |

**Note on NCCLx**: NCCLx uses the same device APIs as NCCL (via `libnccl_device.bc`).
The distinction is primarily in host-side features (sparse allreduce, comm dump) enabled
by the `IS_NCCLX` compile flag. NCCLx contexts can dispatch to the existing NCCL backend.

**Key Files:**

```
torch/csrc/_extern_triton/
├── symm_comm.cuh          # Context definitions (SymmContext, SymmTeam)
├── torch_symm.cu          # Unified frontend with dispatch
├── nvshmem_symm.cuh       # NVSHMEM backend implementations
├── nccl_symm.cuh          # NCCL backend implementations
└── torch_symm.bc          # Compiled bitcode library

torch/_extern_triton/
├── _torch_symm_triton.py  # Python frontend, decorators, library finder
├── _nvshmem_symm_comm.py  # NVSHMEM context builder
└── _nccl_symm_comm.py     # NCCL context builder
```

---

## 6. Integration with TorchComms

### 6.1 TorchComms Overview

**TorchComms** is a unified communication API library designed to provide robust,
fault-tolerant distributed communication at scale. It abstracts over multiple
backend implementations (NCCL, NCCLx, RCCL, etc.) and provides:

- High-level collective operations
- Communicator management with caching and lifecycle control
- DeviceMesh integration for multi-dimensional parallelism
- Backend-agnostic API for heterogeneous hardware platforms

```
+-------------------------------------------------------------------------+
|                    TorchComms Architecture                              |
+-------------------------------------------------------------------------+

                    Application Layer
                          |
                          v
+-----------------------------------------------+
|              torchcomms.new_comm()            |  <- Unified API
|              torchcomms.init_device_mesh()    |
+-----------------------------------------------+
          |               |               |
          v               v               v
    +---------+     +---------+     +---------+
    |  NCCL   |     |  NCCLx  |     |  RCCL   |     <- Backend Plugins
    | Backend |     | Backend |     | Backend |
    +---------+     +---------+     +---------+
          |               |               |
          v               v               v
    +---------+     +---------+     +---------+
    | ncclComm|     | ncclComm|     | rcclComm|     <- Native Communicators
    +---------+     +---------+     +---------+
```

### 6.2 Integration Opportunity: Communicator Reuse

Currently, `_extern_triton` creates its own communicators independently via
`NCCLSymmComm` or leverages `ProcessGroupNCCL::getCommPtr()`. This leads to:

- **Resource duplication**: Multiple communicators for the same rank topology
- **Initialization overhead**: Redundant NCCL initialization calls
- **Lifecycle complexity**: Manual synchronization of communicator lifetimes

**Proposed Integration**: Reuse TorchComms communicators to create SymmContext
objects for NCCL and NCCLx backends, eliminating duplication and simplifying
resource management.

```
+-------------------------------------------------------------------------+
|              CURRENT ARCHITECTURE (Duplicated Resources)                |
+-------------------------------------------------------------------------+

    TorchComms                          _extern_triton
        |                                     |
        v                                     v
  torchcomms.new_comm()              NCCLSymmComm.__init__()
        |                                     |
        v                                     v
  +-------------+                      +-------------+
  | ncclComm #1 |                      | ncclComm #2 |  <- DUPLICATED!
  +-------------+                      +-------------+
        |                                     |
        v                                     v
  Collectives                          SymmContext
  (allreduce, etc.)                    (Triton kernels)


+-------------------------------------------------------------------------+
|              PROPOSED ARCHITECTURE (Shared Resources)                   |
+-------------------------------------------------------------------------+

                    TorchComms
                         |
                         v
               torchcomms.new_comm()
                         |
                         v
               +------------------+
               |    ncclComm      |  <- SINGLE COMMUNICATOR
               +------------------+
                    |         |
         +----------+         +----------+
         |                               |
         v                               v
    Collectives                   SymmContextBuilder
    (allreduce, etc.)             .from_torchcomms(comm)
                                         |
                                         v
                                  +---------------+
                                  | NCCLSymmContext|
                                  +---------------+
                                         |
                                         v
                                  Triton Kernels
```

### 6.3 Implementation Approach

#### 6.3.1 TorchComms Communicator Interface (Verified APIs)

Based on actual usage in TorchTitan, TorchComms provides the following **verified** APIs:

```python
import torchcomms
import torch

# Create a communicator
# Verified: torchtitan/experiments/torchcomms/parallel_dims.py:75-79
comm = torchcomms.new_comm(
    backend,                   # Backend name from env: "nccl", "ncclx", "rccl", etc.
    device,                    # torch.device("cuda")
    name="my_comm",            # Communicator name (keyword argument)
)

# Get this process's rank
# Verified: torchtitan/experiments/torchcomms/parallel_dims.py:81
rank = comm.get_rank()

# Split communicator for sub-groups
# Verified: torchtitan/experiments/torchcomms/parallel_dims.py:92
sub_comm = comm.split(ranks, name)  # ranks: List[int], name: str

# Finalize (cleanup) - must be called on sub-comms before root
# Verified: torchtitan/experiments/torchcomms/parallel_dims.py:106-107
sub_comm.finalize()
comm.finalize()
```

**Note**: The following APIs are observed in C++ headers but require verification for Python bindings:
- `get_size()` - Get world size (C++: `getSize()`)
- `get_backend()` - Get backend name string (C++: `getBackendName()`)

#### 6.3.2 Proposed API Extensions for SymmContext Integration

To enable SymmContext creation from TorchComms communicators, the following
**proposed API extensions** would be needed in TorchComms:

```python
# PROPOSED EXTENSION (not yet implemented in TorchComms)
class Communicator:
    # ... existing methods ...

    def get_native_handle(self) -> int:
        """
        [PROPOSED] Get the underlying native communicator handle.

        Returns:
            int64: Pointer to native communicator (e.g., ncclComm_t for NCCL)

        This method would enable integration with symmetric memory contexts
        by allowing direct access to the NCCL communicator for window
        registration and device communicator creation.
        """
        pass

    def get_size(self) -> int:
        """
        [NEEDS VERIFICATION] Get the total number of ranks.

        Returns:
            int: World size of this communicator
        """
        pass

    def get_backend(self) -> str:
        """
        [NEEDS VERIFICATION] Get the backend name.

        Returns:
            str: Backend name ("nccl", "ncclx", "rccl", "nvshmem", etc.)
        """
        pass
```

#### 6.3.3 SymmContext Factory from TorchComms

Add a factory method to create SymmContext from an existing TorchComms communicator.
This requires the proposed `get_native_handle()` extension:

```python
# torch/_extern_triton/_torchcomms_integration.py

from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    import torchcomms

class TorchCommsSymmContextFactory:
    """
    Factory for creating SymmContext objects from TorchComms communicators.

    This enables reuse of communicators that are already created and managed
    by TorchComms, avoiding resource duplication and ensuring consistent
    lifecycle management.

    REQUIREMENTS:
    - TorchComms must expose get_native_handle() to access underlying ncclComm_t
    - TorchComms must expose get_size() for world size
    - TorchComms must expose get_backend() for backend detection
    """

    @staticmethod
    def create_nccl_context(
        comm: "torchcomms.Communicator",
        buffer: torch.Tensor,
        signal_pad_size: int = 1024,
    ) -> int:
        """
        Create an NCCLSymmContext from a TorchComms communicator.

        Args:
            comm: TorchComms communicator (must be NCCL or NCCLx backend)
            buffer: Tensor to use as symmetric memory buffer
            signal_pad_size: Size of signal pad in bytes

        Returns:
            int64: Device pointer to NCCLSymmContext

        Raises:
            RuntimeError: If communicator backend is not NCCL/NCCLx
            AttributeError: If required TorchComms APIs are not available
        """
        # Validate backend (requires get_backend() extension)
        if not hasattr(comm, 'get_backend'):
            raise AttributeError(
                "TorchComms communicator does not expose get_backend(). "
                "This integration requires TorchComms API extensions."
            )

        backend = comm.get_backend()
        if backend not in ("nccl", "ncclx"):
            raise RuntimeError(
                f"NCCL SymmContext requires NCCL/NCCLx backend, got: {backend}"
            )

        # Extract the underlying ncclComm_t handle (requires extension)
        if not hasattr(comm, 'get_native_handle'):
            raise AttributeError(
                "TorchComms communicator does not expose get_native_handle(). "
                "This integration requires TorchComms API extensions."
            )

        nccl_comm_ptr = comm.get_native_handle()

        # Get world size (requires get_size() extension)
        if not hasattr(comm, 'get_size'):
            raise AttributeError(
                "TorchComms communicator does not expose get_size(). "
                "This integration requires TorchComms API extensions."
            )

        # Use existing infrastructure to build context
        from torch._C._distributed_c10d import (
            _create_nccl_symm_context_from_comm,
        )

        return _create_nccl_symm_context_from_comm(
            nccl_comm_ptr,
            buffer.data_ptr(),
            buffer.numel() * buffer.element_size(),
            signal_pad_size,
            buffer.device.index,
            comm.get_rank(),       # Verified API
            comm.get_size(),       # Requires verification
        )

    @staticmethod
    def create_context(
        comm: "torchcomms.Communicator",
        buffer: torch.Tensor,
        signal_pad_size: int = 1024,
    ) -> int:
        """
        Create a SymmContext from a TorchComms communicator.

        Automatically selects the appropriate context type based on backend.

        Args:
            comm: TorchComms communicator
            buffer: Tensor to use as symmetric memory buffer
            signal_pad_size: Size of signal pad in bytes

        Returns:
            int64: Device pointer to SymmContext (NCCL or NVSHMEM)
        """
        if not hasattr(comm, 'get_backend'):
            raise AttributeError(
                "TorchComms communicator does not expose get_backend(). "
                "Cannot auto-detect backend for SymmContext creation."
            )

        backend = comm.get_backend()

        if backend in ("nccl", "ncclx"):
            return TorchCommsSymmContextFactory.create_nccl_context(
                comm, buffer, signal_pad_size
            )
        elif backend == "nvshmem":
            return TorchCommsSymmContextFactory.create_nvshmem_context(
                comm, buffer, signal_pad_size
            )
        else:
            raise RuntimeError(f"Unsupported backend for SymmContext: {backend}")
```

#### 6.3.3 C++ Backend: Context Creation from Existing Communicator

Add C++ functions to create SymmContext from an existing `ncclComm_t`:

```cpp
// torch/csrc/_extern_triton/torchcomms_integration.cpp

#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>
#include "symm_comm.cuh"

namespace torch_symm {

/**
 * Create NCCLSymmContext from an existing ncclComm_t.
 *
 * This enables reuse of communicators from TorchComms or ProcessGroupNCCL
 * without creating duplicate communicators.
 *
 * @param nccl_comm_ptr Pointer to existing ncclComm_t
 * @param buffer_ptr Device pointer to symmetric buffer
 * @param buffer_size Size of buffer in bytes
 * @param signal_pad_size Size of signal pad in bytes
 * @param device_idx CUDA device index
 * @param rank This process's rank
 * @param world_size Total number of ranks
 * @return Device pointer to NCCLSymmContext
 */
int64_t create_nccl_symm_context_from_comm(
    int64_t nccl_comm_ptr,
    int64_t buffer_ptr,
    size_t buffer_size,
    size_t signal_pad_size,
    int device_idx,
    int rank,
    int world_size) {

  ncclComm_t comm = reinterpret_cast<ncclComm_t>(nccl_comm_ptr);
  void* buffer = reinterpret_cast<void*>(buffer_ptr);

  // Register buffer as symmetric window
  ncclWindow_t buffer_window;
  C10D_NCCL_CHECK(
      ncclCommWindowRegister(
          comm, buffer, buffer_size, &buffer_window, NCCL_WIN_COLL_SYMMETRIC),
      "Failed to register buffer window");

  // Allocate and register signal pad
  void* signal_pad_ptr;
  ncclWindow_t signal_window;
  C10D_NCCL_CHECK(ncclMemAlloc(&signal_pad_ptr, signal_pad_size),
                  "Failed to allocate signal pad");
  C10D_NCCL_CHECK(
      ncclCommWindowRegister(
          comm, signal_pad_ptr, signal_pad_size, &signal_window,
          NCCL_WIN_COLL_SYMMETRIC),
      "Failed to register signal window");

  // Create device communicator (reuses if already exists)
#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
  auto& devcomm_manager = c10d::symmetric_memory::NCCLDevCommManager::get(
      c10::Device(c10::DeviceType::CUDA, device_idx));

  // Use a unique group name based on communicator pointer
  std::string group_name = "torchcomms_" + std::to_string(nccl_comm_ptr);
  devcomm_manager.try_emplace_devcomm(group_name, comm);
  ncclDevComm* dev_comm = &devcomm_manager.get_devcomm(group_name);
#else
  ncclDevComm* dev_comm = nullptr;
#endif

  // Build peer pointer arrays (same as NCCLPeerAllocInfo)
  uint64_t** signal_pad_ptrs = build_peer_signal_ptrs(
      signal_window, world_size, device_idx);

  // Allocate context on device
  NCCLSymmContext* ctx;
  cudaMalloc(&ctx, sizeof(NCCLSymmContext));

  NCCLSymmContext host_ctx(
      rank, world_size,
      buffer_window, signal_window,
      dev_comm, buffer, buffer_size,
      signal_pad_ptrs, device_idx);

  cudaMemcpy(ctx, &host_ctx, sizeof(NCCLSymmContext), cudaMemcpyHostToDevice);

  return reinterpret_cast<int64_t>(ctx);
}

} // namespace torch_symm
```

### 6.4 Usage Pattern with TorchComms

#### 6.4.1 Basic Integration

```python
import torch
import torchcomms
import triton
from torch._extern_triton import (
    BACKEND_NCCL,
    requires_torch_symm,
    symm_lsa_ptr,
    symm_barrier,
)
from torch._extern_triton._torchcomms_integration import (
    TorchCommsSymmContextFactory,
)


def run_with_torchcomms():
    # 1. Create TorchComms communicator (shared resource)
    comm = torchcomms.new_comm(
        backend="ncclx",
        device=torch.device("cuda"),
        name="my_distributed_job",
    )

    # 2. Allocate symmetric buffer
    buffer = torch.zeros(1024 * 1024, dtype=torch.float32, device="cuda")

    # 3. Create SymmContext from TorchComms communicator
    ctx_ptr = TorchCommsSymmContextFactory.create_context(comm, buffer)

    # 4. Use in Triton kernel
    @requires_torch_symm(backend=BACKEND_NCCL)
    @triton.jit
    def my_kernel(ctx_ptr, buffer_ptr, peer: tl.constexpr):
        # Get pointer to peer's buffer
        peer_ptr = symm_lsa_ptr(ctx_ptr, buffer_ptr, peer, BACKEND_NCCL)
        # ... kernel logic ...
        symm_barrier(ctx_ptr, BACKEND_NCCL)

    my_kernel[(1,)](ctx_ptr, buffer.data_ptr(), peer=1)

    # 5. TorchComms manages communicator lifecycle
    # Context is automatically cleaned up when comm is finalized
    comm.finalize()
```

#### 6.4.2 Integration with DeviceMesh

```python
import torch
import torchcomms
from torchcomms.device_mesh import init_device_mesh
from torch._extern_triton._torchcomms_integration import (
    TorchCommsSymmContextFactory,
)


def setup_distributed_training():
    # Create root communicator
    comm = torchcomms.new_comm(
        backend="ncclx",
        device=torch.device("cuda"),
        name="training_job",
    )

    # Split for different parallelism dimensions
    dp_comm = comm.split(ranks=dp_ranks, name="dp")
    tp_comm = comm.split(ranks=tp_ranks, name="tp")

    # Initialize DeviceMesh with TorchComms
    device_mesh = init_device_mesh(
        mesh_dim_comms=(dp_comm, tp_comm),
        mesh_dim_names=("dp", "tp"),
        _global_comm=comm,
    )

    # Create SymmContext for tensor-parallel communication
    tp_buffer = torch.zeros(1024 * 1024, dtype=torch.float32, device="cuda")
    tp_symm_ctx = TorchCommsSymmContextFactory.create_context(
        tp_comm, tp_buffer
    )

    # Now tp_symm_ctx can be used in Triton kernels for TP collectives
    return device_mesh, tp_symm_ctx, comm
```

### 6.5 Benefits of Integration

| Aspect                | Without Integration          | With Integration              |
|-----------------------|------------------------------|-------------------------------|
| **Communicators**     | Duplicated per use case      | Shared across use cases       |
| **Initialization**    | Multiple NCCL init calls     | Single init via TorchComms    |
| **Lifecycle**         | Manual per-component         | Unified via TorchComms        |
| **Backend Selection** | Hardcoded per component      | Centralized in TorchComms     |
| **DeviceMesh**        | Separate from Triton kernels | Unified topology management   |
| **Resource Usage**    | Higher memory overhead       | Minimal overhead              |

### 6.6 Implementation Roadmap

```
+-------------------------------------------------------------------------+
|                    TorchComms Integration Roadmap                       |
+-------------------------------------------------------------------------+

Phase 1: Foundation (Current Focus)
├── [ ] Add TorchCommsSymmContextFactory Python class
├── [ ] Implement C++ create_nccl_symm_context_from_comm()
├── [ ] Add Python bindings for context creation
└── [ ] Unit tests for basic integration

Phase 2: Window Management
├── [ ] Support multiple buffers per communicator
├── [ ] Implement window caching to avoid re-registration
├── [ ] Handle buffer lifecycle (register/deregister)
└── [ ] Integration tests with DeviceMesh

Phase 3: Full Integration
├── [ ] Automatic backend detection from TorchComms
├── [ ] Support for NVSHMEM backend via TorchComms
├── [ ] Signal pad sharing across contexts
└── [ ] Performance benchmarks vs standalone

Phase 4: Production Hardening
├── [ ] Error handling and recovery
├── [ ] Thread safety audit
├── [ ] Documentation and examples
└── [ ] Integration with TorchTitan experiments
```

### 6.7 API Reference Summary

```python
# Primary integration API
class TorchCommsSymmContextFactory:
    @staticmethod
    def create_context(comm, buffer, signal_pad_size=1024) -> int:
        """Create SymmContext from TorchComms communicator."""

    @staticmethod
    def create_nccl_context(comm, buffer, signal_pad_size=1024) -> int:
        """Create NCCLSymmContext specifically."""

    @staticmethod
    def create_nvshmem_context(comm, buffer, signal_pad_size=1024) -> int:
        """Create NVSHMEMSymmContext specifically."""


# C++ backend (exposed via pybind11)
int64_t _create_nccl_symm_context_from_comm(
    int64_t nccl_comm_ptr,
    int64_t buffer_ptr,
    size_t buffer_size,
    size_t signal_pad_size,
    int device_idx,
    int rank,
    int world_size
);
```
