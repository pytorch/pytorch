# NCCL EP (Expert Parallelism) API

NCCL EP is a high-performance NCCL API extension for efficient Mixture-of-Experts (MoE) communication.
It provides optimized dispatch and combine primitives for Expert Parallelism (EP) across distributed GPU systems
implemented on top of NCCL Device API: Load-Store Accessible (LSA) and GPU-Initiated Networking (GIN) operations.

# Maintainers

| GitHub | Areas |
|--------|------|
| @artpol84 | APIs, new features, layouts |
| @kwen2501 | APIs, integration |
| @sb17v | Kernels, build systems |
| @nv-lschneider | Kernels, mnnvl |
| @kgioioso | GIN, NCCL |

# Table of Contents

- [Overview](#overview)
- [Usage](#usage)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
- [Execution Modes](#execution-modes)
- [Usage Examples](#usage-examples)

# Overview

The NCCL EP API extends NCCL with native support for efficient MoE communication patterns. It provides optimized implementations for token "dispatch" and expert output "combine" operations, which are an important components of modern large language models employing sparse MoE architectures.

NCCL EP brings the performance benefits of modern device-initiated MoE libraries into the NCCL ecosystem with a unified API. NCCL EP provides unified `ncclEpDispatch` and `ncclEpCombine` primitives that allow selecting the appropriate algorithm based on workload characteristics.

Currently, two distinct communication algorithms tailored for different workload characteristics are supported:

* **Low-Latency (LL)** - optimized for small batch sizes and latency-sensitive workloads (i.e., LLM inference). To minimize latency, it uses direct point-to-point all-to-all communication with experts.

* **High-Throughput (HT)** - optimized for training and inference prefilling with large batch sizes. HT mode implements hierarchical communication patterns, relying on NVLink for intra-node aggregation, and on RDMA for inter-node communication. The implementation leverages Hopper architecture features, including warp-specialized pipelines, and TMA (Tensor Memory Accelerator) operations.

NCCL EP relies on NCCL Device API, using GIN `put`/`signal` operations for RDMA and LSA load/store operations for NVLink communication, eliminating CPU involvement in the critical path while inheriting NCCL topology detection and plugin architecture.

## Key Features

- **Staged Execution** (LL mode only): Enable computation-communication overlap through a `send-only` flag.
- **Automatic Tuning**: Let the API auto-tune buffer sizes, queue pairs, and channels.

## Quick Start

### C API

```c
// Group management. Custom allocator (if any) is set via config.alloc
// (ncclEpAllocConfig_t); zero-init uses cudaMalloc/cudaFree.
ncclEpCreateGroup(&ep_group, comm, &config);
ncclEpGroupDestroy(ep_group);

// Handle management. `topk_idx` is a pointer to a caller-owned tensor
// descriptor; the routing it carries is cached in the handle and reused by
// all dispatches until ncclEpUpdateHandle is called with new routing.
// `layout_info` is an optional ncclEpLayoutInfo_t* whose fields advertise
// device-side metadata tensors (expert_counters, src_rank_counters,
// expert_offsets, recv_total_counter).
ncclEpCreateHandle(&handle, ep_group, layout, &topk_idx, layout_info, handle_cfg, stream);
ncclEpUpdateHandle(handle, &new_topk_idx, layout_info, stream);  // optional: refresh routing
ncclEpHandleDestroy(handle);

// Communication operations. inputs / outputs are named-struct pointers
// (ncclEpDispatchInputs_t / ncclEpDispatchOutputs_t /
// ncclEpCombineInputs_t / ncclEpCombineOutputs_t); each cross-boundary
// tensor lives in a named field as a `ncclEpTensor_t*`.
ncclEpDispatch(handle, &dispatch_in, &dispatch_out, layout_info, &dispatch_cfg, stream);
ncclEpCombine(handle, &combine_in, &combine_out, &combine_cfg, stream);
ncclEpComplete(handle, config, stream);  // LL mode only
```

### Python API

Install nccl4py, which includes the NCCL EP Python bindings as `nccl.ep`. Only CUDA 13 is supported as of now.

```bash
$ pip install nccl4py[cu13]
```

Import and use NCCL EP in a python application
```python
from nccl.ep import NCCLLibrary, NCCL_EP_ALGO_LOW_LATENCY

nccl_lib = NCCLLibrary()
# Use nccl_lib.ncclEpDispatch, ncclEpCombine, etc.
```

### Reference performance numbers

For microbenchmarking, NCCL EP provides the performance evaluation tool [`ep_bench`](ep_bench.cu).

Below, the reference performance numbers collected
on NVIDIA H100 platform for Low-Latency mode
using **BF16 dispatch and combine** (same data type) are presented.
Note, the data was obtained for NCCL v2.29u1 release.

#### Test Configuration
- Hidden: 7168
- Top-k: 8
- Experts: 256
- Tokens: 128 per rank
- 8 GPUs per node
- Up to 8 nodes

#### Performance

| Number Of GPUs | Node Count | Dispatch BW (GB/s) | Combine BW (GB/s)  |
|:--------------:|:----------:|:------------------:|:------------------:|
| 8              | 1          | 224.3              | 185.2              |
| 16             | 2          | 76.7               | 73.0               |
| 32             | 4          | 53.6               | 50.0               |
| 64             | 8          | 48.8               | 43.8               |

### Common scenarios

This section provides a high-level overview of the input, output, and layout
metadata tensors expected by the API for common scenarios. Each cross-boundary
tensor lives in a named field of one of the API's struct types
(`ncclEpDispatchInputs_t`, `ncclEpDispatchOutputs_t`, `ncclEpLayoutInfo_t`,
`ncclEpCombineInputs_t`, `ncclEpCombineOutputs_t`); the **Struct** column
names the struct and the **Field** column names the field within it.

#### Used notation

**Dimensions:**
* B = batch size
* H = hidden dimension
* S = scales dimension
* L = number of local experts
* K = top K
* R = number of ranks (nRanks)
* N(r) = number of tokens targeting rank r


#### LL mode (same data type)

| Operation | Struct             | Field             | Dims             |
|:---------:|:-------------------|:------------------|:----------------:|
| Dispatch  | dispatch_inputs    | tokens            | [B x H]          |
|           | dispatch_outputs   | tokens            | [L x R x B x H]  |
|           | layout_info        | expert_counters   | [L]              |
| Combine   | combine_inputs     | tokens            | [L x R x B x H]  |
|           | combine_outputs    | tokens            | [B x H]          |
|           | combine_outputs    | topk_weights      | [B x K]          |


#### HT mode (same data type)

HT mode supports `NCCL_EP_LAYOUT_FLAT` and `NCCL_EP_LAYOUT_EXPERT_MAJOR`.
With `NCCL_EP_LAYOUT_FLAT`, dispatch output is a contiguous 2D sequence of N(r) received tokens with no rank-major or expert-major structure.
With `NCCL_EP_LAYOUT_EXPERT_MAJOR`, dispatch output is grouped by local expert, optionally padded via `dispatch_output_per_expert_alignment`.

**Handle creation**

| Operation | Struct        | Field           | Dims |
|-----------|:--------------|:----------------|:----:|
| Create    | layout_info   | expert_counters | [L]  |


**Forward pass**

`topk_idx` is supplied once via `ncclEpCreateHandle` (or refreshed via
`ncclEpUpdateHandle`) and cached on the handle; subsequent dispatches reuse it.

| Operation | Struct             | Field             | Dims       |
|:---------:|:-------------------|:------------------|:----------:|
| Dispatch  | dispatch_inputs    | tokens            | [B x H]    |
|           | dispatch_inputs    | topk_weights      | [B x K]    |
|           | dispatch_outputs   | tokens            | [N(r) x H] |
|           | dispatch_outputs   | topk_weights      | [N(r) x K] |
|           | dispatch_outputs   | topk_idx          | [N(r) x K] |
| Combine   | combine_inputs     | tokens            | [N(r) x H] |
|           | combine_outputs    | tokens            | [B x H]    |

**Backward pass**

Compared to the Forward pass, the Backward pass requires per-token routing
weights to be passed as `combine_inputs.topk_weights` and returned via
`combine_outputs.topk_weights`.

| Operation | Struct             | Field             | Dims       |
|:---------:|:-------------------|:------------------|:----------:|
| Dispatch  | dispatch_inputs    | tokens            | [B x H]    |
|           | dispatch_inputs    | topk_weights      | [B x K]    |
|           | dispatch_outputs   | tokens            | [N(r) x H] |
|           | dispatch_outputs   | topk_weights      | [N(r) x K] |
|           | dispatch_outputs   | topk_idx          | [N(r) x K] |
| Combine   | combine_inputs     | tokens            | [N(r) x H] |
|           | combine_inputs     | **topk_weights**  | [N(r) x K] |
|           | combine_outputs    | tokens            | [B x H]    |
|           | combine_outputs    | **topk_weights**  | [B x K]    |

#### LL mode (FP16 -> FP8 conversion - NOT SUPPORTED)

In the token data type conversion scenario, in LL mode the Dispatch operation
will perform precision reduction and return lower-precision tokens via
`dispatch_outputs.tokens`. In addition, `dispatch_outputs.scales` must be
provided to return the scaling information.

| Operation | Struct             | Field             | Dims             |
|:---------:|:-------------------|:------------------|:----------------:|
| Dispatch  | dispatch_inputs    | tokens            | [B x H]          |
|           | dispatch_outputs   | tokens            | [L x R x B x H]  |
|           | dispatch_outputs   | **scales**        | [L x R x B x S]  |
|           | layout_info        | expert_counters   | [L]              |
| Combine   | combine_inputs     | tokens            | [L x R x B x H]  |
|           | combine_outputs    | tokens            | [B x H]          |
|           | combine_outputs    | topk_weights      | [B x K]          |

#### HT mode (FP16 -> FP8 conversion - NOT SUPPORTED)

**Forward pass**

In the HT case, scales are expected to be calculated by the user and passed
via `dispatch_inputs.scales`. NCCL EP communicates them alongside the token
data and returns them via `dispatch_outputs.scales`.

| Operation | Struct             | Field             | Dims       |
|:---------:|:-------------------|:------------------|:----------:|
| Dispatch  | dispatch_inputs    | tokens            | [B x H]    |
|           | dispatch_inputs    | topk_weights      | [B x K]    |
|           | dispatch_inputs    | **scales**        | [B x S]    |
|           | dispatch_outputs   | tokens            | [N(r) x H] |
|           | dispatch_outputs   | topk_weights      | [N(r) x K] |
|           | dispatch_outputs   | topk_idx          | [N(r) x K] |
|           | dispatch_outputs   | **scales**        | [N(r) x S] |
| Combine   | combine_inputs     | tokens            | [N(r) x H] |
|           | combine_inputs     | topk_weights      | [N(r) x K] |
|           | combine_outputs    | tokens            | [B x H]    |

# Usage

## Prerequisites

### Dependencies

| Component | Version | Notes |
|-----------|---------|-------|
| CUDA | 13+ | Required |
| NCCL | 2.29+ | With Device API and GIN support |
| MPI | Any (OpenMPI, MPICH, etc.) | Required for multi-process launch |
| GPU | Hopper (H100) or Blackwell | Tested configurations |

### Discover compute capabilities

Use `nvidia-smi` command to detect the compute capabilities of your NVIDIA GPU.

For example, on Hopper system with `compute_cap` of `90`, the output looks like below:
```bash
$ nvidia-smi --query-gpu=compute_cap --format=csv
compute_cap
9.0
...
```

### Set the environment

The following paths are required for building and running the

```
export COMPUTE_CAP=<discovered compute_cap>
export CUDA_HOME=/path/to/cuda
export MPI_HOME=/path/to/openmpi
export NCCL_HOME=/path/to/nccl/build # The desired NCCL build
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${NCCL_HOME}/lib:$LD_LIBRARY_PATH"
export PATH="${CUDA_HOME}/bin:${NCCL_HOME}/bin:${MPI_HOME}/bin:$PATH"
```

## Building

### Step 1: Build NCCL with Device API Support

```bash
cd /path/to/nccl-source
make -j src.build BUILDDIR=${NCCL_HOME}
```

This creates the NCCL build artifacts in `BUILDDIR` (`./build` by default):
- `${NCCL_HOME}/lib/libnccl.so` - NCCL library with EP support
- `${NCCL_HOME}/include/` - Header files

### Step 2: Build NCCL EP Library and Test


```bash
make -C contrib/nccl_ep MPI=1 BUILDDIR=${NCCL_HOME} \
       NVCC_GENCODE="-gencode=arch=compute_${COMPUTE_CAP},code=sm_${COMPUTE_CAP}"
```

Once `make` command is successfuly completed, the following files will be created:
- `${NCCL_HOME}/lib/libnccl_ep.a` - Static library
- `${NCCL_HOME}/lib/libnccl_ep.so` - Shared library (for Python bindings)
- `${NCCL_HOME}/include/nccl_ep.h` - C API header
- `${NCCL_HOME}/test/nccl_ep/ep_test` - Test application for both Low-Latency and High-Throughput modes
- `${NCCL_HOME}/test/nccl_ep/ep_bench` - Benchmark application for both Low-Latency and High-Throughput modes

## Running

### Environment Setup

Make sure to set the generic environment according to the
[Set the environment](#set-the-environment) section.

```bash
# NCCL GIN configuration recommended for multi-node RDMA:
export NCCL_GIN_TYPE=3  # GDAKI - GPU Direct Async Kernel-Initiated
```

For debugging, the following variables can be set

```bash
export NCCL_DEBUG=INFO        # Enable NCCL debug output
export NCCL_DEBUG_SUBSYS=ALL  # All subsystems
```


# Core Concepts

## Key Data Structures

### `ncclEpTensor_t` - Multi-dimensional Tensor Descriptor

A lightweight value-type struct that encapsulates tensor metadata and data layout.
Declare on the stack, as a struct member, or statically — no heap allocation needed for
the descriptor itself. Always zero-initialise with `NCCL_EP_TENSOR_INIT` and then fill
the fields directly:

```c
size_t dims[2] = { num_tokens, hidden };
ncclEpTensor_t t = NCCL_EP_TENSOR_INIT;
t.ndim = 2;
t.datatype = ncclFloat16;
t.data = my_device_ptr;   // device pointer (or set win_hdl/win_offset for windows)
t.sizes = dims;           // caller-owned array of length `ndim`

// No destroy needed — the descriptor holds no resources, but `dims` must
// outlive the descriptor for the duration of any library call that uses it.

// Access tensor properties directly:
data  = t.data;    // data pointer
ndim  = t.ndim;    // number of dimensions
sizes = t.sizes;   // sizes pointer (points to caller-owned storage)
```

### `ncclEpGroup_t` - EP Group Configuration

Created from an NCCL communicator, manages the distributed EP configuration across all ranks in the group:

```c
typedef struct {
    unsigned int size;                          // = sizeof(struct); ABI-size check
    unsigned int version;                       // = NCCL_EP_API_VERSION; feature-set version
    ncclEpAlgorithm_t algorithm;                // HT or LL mode
    unsigned int num_experts;                   // Total experts across all ranks
    unsigned int max_dispatch_tokens_per_rank;  // Max tokens any single rank dispatches
    unsigned int max_recv_tokens_per_rank;      // Max tokens any single rank receives
                                                //   HT: required (must be >= max_dispatch_tokens_per_rank)
                                                //   LL: NCCL_EP_AUTO → nRanks * max_dispatch_tokens_per_rank
    unsigned int max_token_bytes;               // Upper bound on per-token bytes
    unsigned long int rdma_buffer_size;         // RDMA buffer size for LL mode.
                                                //   NCCL_EP_AUTO  → lazy: allocate on first ncclEpInitHandle, sized to that
                                                //                  handle's actual (layout, num_topk); collective re-grow
                                                //                  if a later handle needs more. See LL section for caveats.
                                                //   explicit > 0  → allocate exactly that many bytes at group time;
                                                //                  ncclEpInitHandle returns ncclInvalidUsage if a layout
                                                //                  doesn't fit. No reallocation ever performed.
    unsigned int num_qp_per_rank;               // Queue pairs per rank (NCCL_EP_AUTO for auto)
    unsigned int num_channels;                  // Channels per rank (NCCL_EP_AUTO for auto)
    unsigned int max_num_sms;                   // SM cap for EP kernels (NCCL_EP_AUTO for auto)
    ncclEpAllocConfig_t alloc;                  // Custom device-memory allocator (zero-init → cudaMalloc/cudaFree)
    unsigned int enable_mask;                   // Enable active-mask fault tolerance (LL only)
    uint64_t timeout_ns;                        // GPU-side wait-loop timeout (0 = default)
} ncclEpGroupConfig_t;

// Use NCCL_EP_GROUP_CONFIG_INIT to pre-fill `size` and `version` correctly.
```

### `ncclEpHandle_t` - Operation Handle

Maintains state for a sequence of related MoE operations, i.e. dispatch and combine pairs for forward and (optionally) backward passes. The handle encapsulates routing metadata and communication buffers.

## Algorithm-related configurations

**High Throughput (HT)**:
- Uses flat layout (`NCCL_EP_LAYOUT_FLAT`).
- Dispatch output tokens are a contiguous flat sequence: `[N(r) x hidden]` where `N(r)` is the total number of tokens targeting this rank.
  - Static allocation: `N(r) = num_ranks * max_dispatch_tokens_per_rank`.
  - Dynamic allocation (`max_dispatch_tokens_per_rank = NCCL_EP_AUTO`): `N(r)` is the actual received count, written by the metadata kernel into the optional `ncclEpLayoutInfo_t.recv_total_counter` scalar tensor.
- `dispatch_outputs.topk_idx` and `dispatch_outputs.topk_weights` carry per-slot routing metadata alongside the received tokens.
- The caller uses `topk_idx` to route each slot to the correct local expert(s), applies the weighted reduction using `topk_weights`, and passes the pre-reduced `[N(r) x hidden]` tensor as `combine_inputs.tokens` to `ncclEpCombine`.
- Supports dynamic `max_dispatch_tokens_per_rank` (set to `NCCL_EP_AUTO`)

**Low Latency (LL)**:
- Supports `NCCL_EP_LAYOUT_EXPERT_MAJOR` and `NCCL_EP_LAYOUT_RANK_MAJOR` layouts.
- Output tokens are 3D:
  - expert-major: `[num_local_experts, N(e), hidden]`, where `N(e)` is the number of tokens routed to the specific expert `e`.
  - rank-major:   `[num_ranks, max_dispatch_tokens_per_rank, hidden]`.
- Supports `send_only` (in `ncclEpDispatchConfig_t` / `ncclEpCombineConfig_t`) to enable computation/communication overlapping.
- Does not support dynamic `max_dispatch_tokens_per_rank` detection.

***`rdma_buffer_size` and lazy allocation (LL only)***

The group's RDMA buffer is sized by `config.rdma_buffer_size`:

- **`NCCL_EP_AUTO`** (recommended for most users): the buffer is **not** allocated at `ncclEpCreateGroup` time. 
The first `ncclEpInitHandle` allocates it sized to that handle's actual `(layout, num_topk)`.
A later `ncclEpInitHandle` whose layout needs a larger buffer (for example, the first `EXPERT_MAJOR`
handle on a group that previously only hosted `RANK_MAJOR` handles, or a handle with a larger `num_topk`)
**collectively reallocates** the buffer: deregister window → free → `ncclMemAlloc` → register.
The recorded layout offsets on every live handle are pure offsets relative to the group's `rdma_buffer` and resolve correctly against the new base at use time, so existing handles remain valid.

  Two constraints follow from a reallocation:
  1. **All ranks must call `ncclEpInitHandle` in lockstep with the same `(layout, num_topk)`**. With AUTO sizing, `ncclEpInitHandle` is a conditionally collective call.
  2. **Reallocation drops the contents of the old RDMA buffer.** Any operation issued with `send_only = 1` that has staged data but is still awaiting its receive half via `ncclEpComplete` will lose its in-flight data. Drain all such operations before triggering a reallocation.
  3. **CUDA graph capture bakes the RDMA base pointer into the captured kernel parameters.** `ncclEpInitHandle` (in AUTO mode) must not be called between `cudaStreamBeginCapture` and `cudaStreamEndCapture`. Any previously captured graph containing EP kernels must be destroyed and re-captured after a reallocation.

- **Explicit `> 0`**: the buffer is allocated to exactly that size at `ncclEpCreateGroup` time. `ncclEpInitHandle` is purely local; it returns `ncclInvalidUsage` if the requested `(layout, num_topk)` does not fit. Use this mode if you need to avoid collective handle creation, mid-stream reallocation, or graph invalidation. Use `nccl_ep::get_low_latency_rdma_size_hint(...)` to compute a worst-case upper bound across all layouts and `num_topk ≤ MAX_NUM_TOPK`.

## Custom Allocators

The API supports custom memory allocators for custom buffer management. This enables integration with memory pools, custom allocation strategies, or framework-specific allocators.

### Function Signatures

```c
typedef cudaError_t (*ncclEpAllocFn_t)(void** ptr, size_t size, void* context);
typedef cudaError_t (*ncclEpFreeFn_t)(void* ptr, void* context);

typedef struct {
    ncclEpAllocFn_t alloc_fn;  // NULL → default cudaMalloc
    ncclEpFreeFn_t  free_fn;   // NULL → default cudaFree
    void*           context;   // opaque pointer forwarded to every alloc_fn/free_fn call
} ncclEpAllocConfig_t;
```

The `context` value is set once in `ncclEpAllocConfig_t::context` and forwarded unchanged on every call, giving the allocator a stable handle to its backing pool / arena.

### Example

```c
// Custom allocator using a memory pool
cudaError_t my_alloc(void** ptr, size_t size, void* context) {
    MyPool* pool = static_cast<MyPool*>(context);
    *ptr = pool->allocate(size);
    return (*ptr != nullptr) ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t my_free(void* ptr, void* context) {
    MyPool* pool = static_cast<MyPool*>(context);
    pool->deallocate(ptr);
    return cudaSuccess;
}

// Wire the allocator into the group config.
ncclEpGroupConfig_t config = NCCL_EP_GROUP_CONFIG_INIT;
config.alloc.alloc_fn = my_alloc;
config.alloc.free_fn  = my_free;
config.alloc.context  = &my_pool;
ncclEpCreateGroup(&ep_group, comm, &config);
```

If `alloc_fn`/`free_fn` are left NULL (the default after `NCCL_EP_GROUP_CONFIG_INIT`), the library uses `cudaMalloc`/`cudaFree`.


## API Reference

### Group Management

#### `ncclEpCreateGroup()`

```c
// Create an EP group from an NCCL communicator
//   This call is collective and must be invoked by all ranks in the group.
//   Any custom device-memory allocator is supplied via config->alloc
//   (see ncclEpAllocConfig_t); zero-init falls back to cudaMalloc/cudaFree.
//
// Arguments:
//   ep_group   - [OUT] Pointer to newly created EP group
//   comm       - [IN]  Existing NCCL communicator
//   config     - [IN]  Pointer to EP configuration structure
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpCreateGroup(
    ncclEpGroup_t* ep_group,
    ncclComm_t comm,
    const ncclEpGroupConfig_t* config
);
```

#### `ncclEpGroupDestroy()`

```c
// Destroy an EP group and release associated resources.
//
// Arguments:
//   ep_group     - [IN]  EP group to destroy
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpGroupDestroy(
    ncclEpGroup_t ep_group
);
```

### Tensor Descriptors

`ncclEpTensor_t` is a plain value struct.  The common path is to allocate on the
stack (or as a struct member), zero-initialise with `NCCL_EP_TENSOR_INIT`, and
fill fields directly:

```c
size_t dims[2] = { N, H };
ncclEpTensor_t t = NCCL_EP_TENSOR_INIT;
t.ndim = 2; t.datatype = ncclFloat16; t.data = data_ptr;
t.sizes = dims;   // caller owns; must outlive the descriptor's use
```

For window-backed tensors set `win_hdl` / `win_offset` instead of `data`.

When a heap-allocated descriptor with a library-owned `sizes` copy is more
convenient, use `ncclEpTensorAlloc` to obtain one and `ncclEpTensorDestroy`
to release it (the backing data buffer remains caller-owned).

All user-facing tensor fields (`data`, `ndim`, `datatype`, `sizes`, `win_hdl`,
`win_offset`) are directly accessible as struct members.  Public structs
(`ncclEpDispatchInputs_t`, `ncclEpLayoutInfo_t`, …) hold `ncclEpTensor_t*`
pointers, so callers can mix stack-, static-, and heap-allocated descriptors.

### Handle Management

#### `ncclEpCreateHandle()`

```c
// Create and initialize an EP handle.
//   Performs dispatch setup and (in HT mode only) metadata exchange.
//   This call is collective and must be invoked by all ranks in the group.
//   The routing carried by `topk_idx` is cached on the handle; subsequent
//   ncclEpDispatch / ncclEpCombine calls reuse it until ncclEpUpdateHandle
//   replaces it with new routing.
//
// Arguments:
//   handle              - [OUT] Pointer to newly created and initialized EP handle
//   ep_group            - [IN]  A valid EP group
//   layout              - [IN]  Receive buffer layout. Required; must not be NCCL_EP_LAYOUT_UNSET.
//                                HT supports FLAT / EXPERT_MAJOR; LL supports EXPERT_MAJOR / RANK_MAJOR.
//   topk_idx            - [IN]  Pointer to a caller-owned tensor descriptor holding
//                               top-K expert indices (2D [num_tokens, top_k] int64).
//   layout_info         - [IN/OUT, optional] Named-struct pointer carrying device-side
//                         metadata tensor pointers. Set `expert_counters` (1D
//                         ncclInt32/ncclInt64, size = num_local_experts) to receive
//                         per-expert recv counts; set `recv_total_counter` (scalar)
//                         when max_dispatch_tokens_per_rank is NCCL_EP_AUTO. NULL = no metadata.
//   config              - [IN]  Optional handle configuration (e.g. expert-major
//                               alignment via dispatch_output_per_expert_alignment);
//                               NULL = defaults.
//   stream              - [IN]  CUDA stream
//
// Notes:
//   - If max_dispatch_tokens_per_rank in ncclEpGroupConfig_t was set to NCCL_EP_AUTO,
//     this call may block as the host allocates memory for the actual number
//     of received tokens.
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpCreateHandle(
    ncclEpHandle_t* handle,
    ncclEpGroup_t ep_group,
    ncclEpLayout_t layout,
    const ncclEpTensor_t* topk_idx,
    const ncclEpLayoutInfo_t* layout_info,
    const ncclEpHandleConfig_t* config,
    cudaStream_t stream
);
```

#### `ncclEpHandleDestroy()`

```c
// Destroy an EP handle and release all associated resources.
//
// Arguments:
//   handle         - [IN]  EP handle to destroy
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpHandleDestroy(
    ncclEpHandle_t handle
);
```

### Communication Operations

#### `ncclEpDispatch()`

Perform EP dispatch: send tokens to experts according to routing decisions.

```c
// Perform EP dispatch
//   * Sends tokens and metadata to the experts according to routing decisions.
//   * This call is collective and must be invoked by all ranks in the group.
//   * Routing (topk_idx) is taken from the handle — supply it once via
//     ncclEpCreateHandle / ncclEpUpdateHandle.
//   * Cross-boundary tensors are carried in named-struct fields
//     (ncclEpDispatchInputs_t / ncclEpDispatchOutputs_t / ncclEpLayoutInfo_t),
//     each field a `ncclEpTensor_t*` to a caller-owned descriptor.
//
// Arguments:
//   handle        - [IN,OUT] EP handle
//   inputs        - [IN]     Named preallocated input tensors. `inputs->tokens` is required;
//                            other fields (topk_weights, scales) are optional and depend on
//                            algorithm/layout.
//   outputs       - [IN,OUT] Named preallocated output tensors. `outputs->tokens` is required;
//                            other fields (topk_weights, topk_idx, scales) are optional and
//                            depend on algorithm/layout. For HT, outputs are 2D [N(r), data_size].
//                            For LL expert-major, outputs are 3D [num_local_experts, N(r), data_size].
//                            For LL rank-major,   outputs->tokens is 3D
//                            [num_ranks, max_dispatch_tokens_per_rank, data_size].
//   layout_info   - [IN,OUT, optional] Named-struct pointer for device-side metadata tensors.
//                            LL mode: optional `expert_counters` (1D ncclInt32 / ncclInt64,
//                            size = num_local_experts) receives per-expert recv counts.
//                            NULL = no metadata.
//   config        - [IN]     Dispatch configuration.
//   stream        - [IN]     CUDA stream. If ncclEpDispatch is called on a different stream than
//                            the stream used in `ncclEpCreateHandle`, it is the responsibility
//                            of the user to synchronize between streams to ensure correctness.
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpDispatch(
    ncclEpHandle_t handle,
    const ncclEpDispatchInputs_t* inputs,
    const ncclEpDispatchOutputs_t* outputs,
    const ncclEpLayoutInfo_t* layout_info,
    const ncclEpDispatchConfig_t* config,
    cudaStream_t stream
);
```

#### `ncclEpCombine()`

Perform EP combine: gather expert outputs and return in original token order.

```c
// Perform EP combine
//   * Gathers outputs from experts and returns them to their source in original token order.
//   * This call is collective and must be invoked by all ranks in the group.
//   * Cross-boundary tensors are carried in named-struct fields
//     (ncclEpCombineInputs_t / ncclEpCombineOutputs_t).
//
// Arguments:
//   handle           - [IN,OUT] EP handle that was used for `ncclEpDispatch()` operation.
//   inputs           - [IN]     Named preallocated input tensors. `inputs->tokens` is required;
//                               LL expert-major: 3D [num_local_experts, N(r), data_size].
//                               LL rank-major:   3D [num_ranks, max_dispatch_tokens_per_rank, data_size].
//                               HT:              2D [N(r), data_size].
//                               Backward combine: also set `inputs->topk_weights`
//                               [N(r), top_k] (HT only).
//   outputs          - [IN,OUT] Named preallocated output tensors. `outputs->tokens` is required;
//                               2D [num_tokens, data_size] in original token order.
//                               LL expert-major: `outputs->topk_weights` [num_tokens, top_k]
//                               is used by the receive-side reduction.
//                               HT backward combine: `outputs->topk_weights` [num_tokens, top_k]
//                               receives per-token routing weights.
//   config           - [IN]     Combine configuration (e.g. `send_only` for LL staged mode).
//   stream           - [IN]     CUDA stream. If `ncclEpCombine()` is called on a different stream than
//                               the stream used in `ncclEpCreateHandle()`, it is the responsibility
//                               of the user to synchronize between streams to ensure correctness.
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpCombine(
    ncclEpHandle_t handle,
    const ncclEpCombineInputs_t* inputs,
    const ncclEpCombineOutputs_t* outputs,
    const ncclEpCombineConfig_t* config,
    cudaStream_t stream
);
```

#### `ncclEpComplete()` (LL mode only)

```c
// Continues a staged EP operation to completion.
//   * This should be called after a prior `ncclEpDispatch()` or `ncclEpCombine()` call with `send_only` flag set.
//
// Arguments:
//   handle     - [IN,OUT] EP handle used in the preceding staged operation
//   config     - [IN]     Completion configuration (see ncclEpCompleteConfig_t). NULL = defaults.
//   stream     - [IN]     CUDA stream
//
// Notes:
//   - If `ncclEpComplete()` is called on a different stream than the operation initiation call
//     (i.e., `ncclEpDispatch()` or `ncclEpCombine()`), it is the responsibility of the user to
//     synchronize between streams to ensure correctness.
//   - Only LL mode is supported.
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpComplete(
    ncclEpHandle_t handle,
    const ncclEpCompleteConfig_t* config,
    cudaStream_t stream
);
```

# Execution Modes

Both `ncclEpDispatch()` and `ncclEpCombine()` operations support synchronous and staged semantics.

## Synchronous Mode (Default)

Corresponds to a synchronous version of an operation where GPU resources are allocated during the whole operation execution
including the time to wait for the data to be received. This mode doesn't allow for
computation/communication overlap.

```c
ncclEpDispatchConfig_t dispatch_cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
ncclEpDispatch(handle, &dispatch_in, &dispatch_out,
               /*layout_info=*/NULL, &dispatch_cfg, stream);
// Dispatch is complete when this returns
```

## Staged Mode (Low Latency Only)

In this mode, the operation is split into the send and receive phases enabling computation computation/communication overlap:
  * The operation is invoked with `send_only = true` flag
  * In this case, after all data transfers are initiated, the corresponding GPU resources are released
  * This enables an application to utilize all avaialble GPU resources for computation while offloaded data transfers are performed.
  * To complete the operation, the `ncclEpComplete()` primitive is used.

This mode is particularly beneficial for inference with multiple micro-batches.

```c
// Stage 1: Post send requests without waiting for completion
ncclEpDispatchConfig_t dispatch_cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
dispatch_cfg.send_only = 1;
ncclEpDispatch(handle, &dispatch_in, &dispatch_out,
               /*layout_info=*/NULL, &dispatch_cfg, stream);
// Returns after initiating the operations

// Stage 2: Continue other computations...

// Stage 3: Wait for actual completion
ncclEpComplete(handle, /*config=*/NULL, stream);
// Now all data is actually sent/received
```

# Usage Examples

> **Note:** For a complete working example, see `ep_test.cu` which demonstrates both LL and HT modes with all API calls.

## Example 1: High Throughput Mode - Forward and Backward Pass

```c
#include "nccl.h"
#include "nccl_ep.h"
#include "cuda_runtime.h"

// The library does not own tensor memory. The caller can mix two descriptor
// shapes — value-type "static" descriptors (stack / struct member /
// global, populated in place via NCCL_EP_TENSOR_INIT_INLINE) and heap
// "dynamic" descriptors obtained from ncclEpTensorAlloc (library-owned
// `sizes` copy, released by ncclEpTensorDestroy). Public structs hold
// `ncclEpTensor_t*` either way, so the two are interchangeable at use sites.

// Initialize NCCL communicator
ncclComm_t comm;
ncclCommInitRank(&comm, nRanks, id, myRank);

cudaStream_t stream;
cudaStreamCreate(&stream);

unsigned int top_k = 8;
unsigned int hidden = 4096;

// Configure for High Throughput mode
ncclEpGroupConfig_t config = NCCL_EP_GROUP_CONFIG_INIT;
config.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
config.num_experts = 256;
config.max_dispatch_tokens_per_rank = 4;
config.max_recv_tokens_per_rank = nRanks * config.max_dispatch_tokens_per_rank;
config.max_token_bytes = hidden * 2;  // bfloat16
config.rdma_buffer_size = NCCL_EP_AUTO;     // Auto-size
config.num_qp_per_rank = NCCL_EP_AUTO;      // Auto-size
config.num_channels = NCCL_EP_AUTO;         // Auto-size
// Optional: wire a custom device-memory allocator via config.alloc.
// config.alloc.alloc_fn = my_alloc; config.alloc.free_fn = my_free; config.alloc.context = &my_pool;

ncclEpGroup_t ep_group;
ncclEpCreateGroup(&ep_group, comm, &config);

unsigned int num_local_experts = config.num_experts / nRanks;
unsigned int num_recv_tokens   = config.max_recv_tokens_per_rank;

// --- topk_idx: dynamic descriptor (heap, library-owned sizes copy) ---
// ncclEpTensorAlloc copies `dims` into its own storage, so the local
// array can go out of scope safely after the call.
ncclEpTensor_t* topk_idx = nullptr;
{
    size_t dims[2] = { num_tokens, top_k };
    ncclEpTensorAlloc(&topk_idx, 2, ncclInt64, dims, /*config=*/NULL);
    cudaMalloc(&topk_idx->data, num_tokens * top_k * sizeof(int64_t));
}

// --- expert_counters: static descriptor with in-place initialization ---
// The caller owns both the device buffer and the `sizes` array; both must
// outlive any library call that observes the descriptor.
size_t expert_counters_dims[1] = { num_local_experts };
void*  expert_counters_data    = nullptr;
cudaMalloc(&expert_counters_data, num_local_experts * sizeof(int32_t));
ncclEpTensor_t expert_counters = { NCCL_EP_TENSOR_INIT_INLINE,
                                   .ndim = 1, .datatype = ncclInt32,
                                   .data = expert_counters_data,
                                   .sizes = expert_counters_dims };

// Mix dynamic (topk_idx) and static (expert_counters) in the same call.
ncclEpLayoutInfo_t handle_layout = NCCL_EP_LAYOUT_INFO_INIT;
handle_layout.expert_counters = &expert_counters;   // address-of stack descriptor

ncclEpHandle_t handle;
ncclEpCreateHandle(&handle, ep_group, NCCL_EP_LAYOUT_FLAT, topk_idx, &handle_layout,
                   /*config=*/NULL, stream);

// === FORWARD PASS ===
//
// Static (stack, in-place init): in/out tokens, out topk_weights, out topk_idx.
// Dynamic (heap, ncclEpTensorAlloc):  in topk_weights.

// Static input/output token descriptors.
size_t in_tokens_dims[2]        = { num_tokens,      hidden };
size_t out_tokens_dims[2]       = { num_recv_tokens, hidden };
size_t out_topk_weights_dims[2] = { num_recv_tokens, top_k };
size_t out_topk_idx_dims[2]     = { num_recv_tokens, top_k };
void*  in_tokens_data           = nullptr;
void*  out_tokens_data          = nullptr;
void*  out_topk_weights_data    = nullptr;
void*  out_topk_idx_data        = nullptr;
cudaMalloc(&in_tokens_data,        num_tokens      * hidden * sizeof(uint16_t));
cudaMalloc(&out_tokens_data,       num_recv_tokens * hidden * sizeof(uint16_t));
cudaMalloc(&out_topk_weights_data, num_recv_tokens * top_k  * sizeof(float));
cudaMalloc(&out_topk_idx_data,     num_recv_tokens * top_k  * sizeof(int64_t));

// Fast in-place initialization of static tensors
// go away after Dispatch invocation
ncclEpTensor_t in_tokens        = {                          
    NCCL_EP_TENSOR_INIT_INLINE,
    .ndim = 2,
    .datatype = ncclBfloat16,
    .data = in_tokens_data,
    .sizes = in_tokens_dims };
ncclEpTensor_t out_tokens       = { 
    NCCL_EP_TENSOR_INIT_INLINE,
    .ndim = 2,
    .datatype = ncclBfloat16,
    .data = out_tokens_data,
    .sizes = out_tokens_dims };
ncclEpTensor_t out_topk_weights = {
    NCCL_EP_TENSOR_INIT_INLINE,
    .ndim = 2,
    .datatype = ncclFloat32,
     .data = out_topk_weights_data,
    .sizes = out_topk_weights_dims };
ncclEpTensor_t out_topk_idx     = {
    NCCL_EP_TENSOR_INIT_INLINE,
    .ndim = 2,
    .datatype = ncclInt64,
    .data = out_topk_idx_data,
    .sizes = out_topk_idx_dims };

// Dynamic input topk_weights.
ncclEpTensor_t* in_topk_weights = nullptr;
{
    size_t dims[2] = { num_tokens, top_k };
    ncclEpTensorAlloc(&in_topk_weights, 2, ncclFloat32, dims, /*config=*/NULL);
    cudaMalloc(&in_topk_weights->data, num_tokens * top_k * sizeof(float));
}

// Pointer assignments mix `&` (address of stack descriptor) and the bare
// pointer returned by ncclEpTensorAlloc.
ncclEpDispatchInputs_t  dispatch_in  = NCCL_EP_DISPATCH_INPUTS_INIT;
ncclEpDispatchOutputs_t dispatch_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
dispatch_in.tokens        = &in_tokens;          // static
dispatch_in.topk_weights  = in_topk_weights;     // dynamic
dispatch_out.tokens       = &out_tokens;         // static
dispatch_out.topk_weights = &out_topk_weights;   // static
dispatch_out.topk_idx     = &out_topk_idx;       // static

ncclEpDispatchConfig_t dispatch_cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
ncclEpDispatch(handle, &dispatch_in, &dispatch_out,
               &handle_layout, &dispatch_cfg, stream);

// Expert forward computation
// ... process out_tokens using expert_counters to size each expert's slab ...

// Combine expert outputs back to original token order (static descriptors).
size_t combine_in_dims[2]  = { num_recv_tokens, hidden };
size_t combine_out_dims[2] = { num_tokens,      hidden };
void*  combine_in_data     = nullptr;
void*  combine_out_data    = nullptr;
cudaMalloc(&combine_in_data,  num_recv_tokens * hidden * sizeof(uint16_t));
cudaMalloc(&combine_out_data, num_tokens      * hidden * sizeof(uint16_t));
ncclEpTensor_t combine_in_tokens  = { 
    NCCL_EP_TENSOR_INIT_INLINE,
    .ndim = 2, 
    .datatype = ncclBfloat16,
    .data = combine_in_data,
    .sizes = combine_in_dims };
ncclEpTensor_t combine_out_tokens = { 
    NCCL_EP_TENSOR_INIT_INLINE,
    .ndim = 2,
    .datatype = ncclBfloat16,
    .data = combine_out_data,
    .sizes = combine_out_dims };

ncclEpCombineInputs_t  combine_in  = NCCL_EP_COMBINE_INPUTS_INIT;
ncclEpCombineOutputs_t combine_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
combine_in.tokens  = &combine_in_tokens;
combine_out.tokens = &combine_out_tokens;

ncclEpCombineConfig_t combine_cfg = NCCL_EP_COMBINE_CONFIG_INIT;
ncclEpCombine(handle, &combine_in, &combine_out, &combine_cfg, stream);

// === BACKWARD PASS ===
// Reuse the same handle — routing information stays the same. Backward
// descriptors not shown here in detail; assume `grad_*` are caller-prepared
// ncclEpTensor_t values (either pattern works).
//
// IMPORTANT (HT only): the forward/backward direction is selected by the
// `pass_direction` field in the dispatch/combine config. The default is
// NCCL_EP_FWD_PASS, so the backward pass MUST set it to NCCL_EP_BWD_PASS.
// This field impacts the set of required fields:
//   - dispatch: FWD requires inputs->topk_weights and forbids it on BWD
//     (BWD also forbids outputs->topk_weights / outputs->topk_idx);
//   - combine:  FWD forbids inputs->topk_weights, BWD requires both
//     inputs->topk_weights and outputs->topk_weights.

ncclEpDispatchInputs_t  bwd_dispatch_in  = NCCL_EP_DISPATCH_INPUTS_INIT;
ncclEpDispatchOutputs_t bwd_dispatch_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
bwd_dispatch_in.tokens   = &grad_combined;     // user-supplied grad descriptor
bwd_dispatch_out.tokens  = &grad_at_experts;   // preallocated buffer descriptor

ncclEpDispatchConfig_t bwd_dispatch_cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
bwd_dispatch_cfg.pass_direction = NCCL_EP_BWD_PASS;   // toggle the selector
ncclEpDispatch(handle, &bwd_dispatch_in, &bwd_dispatch_out,
               &handle_layout, &bwd_dispatch_cfg, stream);

// Expert backward computation
// ... compute gradients for each expert ...

// Combine gradients (backward combine: also carry per-token routing weights).
ncclEpCombineInputs_t  bwd_combine_in  = NCCL_EP_COMBINE_INPUTS_INIT;
ncclEpCombineOutputs_t bwd_combine_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
bwd_combine_in.tokens        = &grad_expert_outputs;
bwd_combine_in.topk_weights  = &combine_topk_weights_input;
bwd_combine_out.tokens       = &grad_tokens;
bwd_combine_out.topk_weights = &combine_topk_weights_output;

ncclEpCombineConfig_t bwd_combine_cfg = NCCL_EP_COMBINE_CONFIG_INIT;
bwd_combine_cfg.pass_direction = NCCL_EP_BWD_PASS;    // toggle the selector
ncclEpCombine(handle, &bwd_combine_in, &bwd_combine_out, &bwd_combine_cfg, stream);

// Cleanup
ncclEpHandleDestroy(handle);
ncclEpGroupDestroy(ep_group);
// Dynamic descriptors: free the caller-owned device buffer first, then the
// descriptor (which also releases the library-owned `sizes` copy).
cudaFree(topk_idx->data);
ncclEpTensorDestroy(topk_idx);
cudaFree(in_topk_weights->data);
ncclEpTensorDestroy(in_topk_weights);
// Static descriptors: free the device buffer; the descriptor itself lives on
// the stack and needs no release. The `sizes` arrays are stack locals too.
cudaFree(in_tokens_data);
cudaFree(out_tokens_data);
cudaFree(out_topk_weights_data);
cudaFree(out_topk_idx_data);
cudaFree(expert_counters_data);
cudaFree(combine_in_data);
cudaFree(combine_out_data);
ncclCommDestroy(comm);
cudaStreamDestroy(stream);
```

## Example 2: Low Latency Mode - Forward Pass (Expert-Major and Rank-Major)

```c
#include "nccl.h"
#include "nccl_ep.h"
#include "cuda_runtime.h"

// Mirrors Example 1's pattern: mix value-type "static" descriptors
// (NCCL_EP_TENSOR_INIT_INLINE) with heap "dynamic" descriptors
// (ncclEpTensorAlloc). In LL forward, only topk_idx is dynamic.

// Initialize NCCL communicator
ncclComm_t comm;
ncclCommInitRank(&comm, nRanks, id, myRank);

cudaStream_t stream;
cudaStreamCreate(&stream);

unsigned int top_k = 8;
unsigned int hidden = 4096;

// Configure for Low Latency mode
ncclEpGroupConfig_t config = NCCL_EP_GROUP_CONFIG_INIT;
config.algorithm = NCCL_EP_ALGO_LOW_LATENCY;
config.num_experts = 256;
config.max_dispatch_tokens_per_rank = 128;  // Required for LL mode
config.max_token_bytes = hidden * 2;    // bfloat16
config.rdma_buffer_size = NCCL_EP_AUTO; // Auto-size
config.num_qp_per_rank = NCCL_EP_AUTO;  // Auto-size (or specify for LL)
config.num_channels = NCCL_EP_AUTO;     // Auto-size

ncclEpGroup_t ep_group;
ncclEpCreateGroup(&ep_group, comm, &config);

unsigned int num_local_experts = config.num_experts / nRanks;

// --- topk_idx: dynamic descriptor (required at handle creation in LL mode) ---
ncclEpTensor_t* topk_idx = nullptr;
{
    size_t dims[2] = { num_tokens, top_k };
    ncclEpTensorAlloc(&topk_idx, 2, ncclInt64, dims, /*config=*/NULL);
    cudaMalloc(&topk_idx->data, num_tokens * top_k * sizeof(int64_t));
}

// Choose the LL receive-buffer layout. Both EXPERT_MAJOR and RANK_MAJOR are
// supported in LL mode; this example branches on `layout` because the dispatch
// output / combine input tensors and the required metadata differ between them.
ncclEpLayout_t layout = NCCL_EP_LAYOUT_EXPERT_MAJOR;  // or NCCL_EP_LAYOUT_RANK_MAJOR
const bool expert_major = (layout == NCCL_EP_LAYOUT_EXPERT_MAJOR);

// Create EP handle. LL mode does not consume layout_info at handle-create time.
ncclEpHandle_t handle;
ncclEpCreateHandle(&handle, ep_group, layout, topk_idx,
                   /*layout_info=*/NULL, /*config=*/NULL, stream);

// === FORWARD PASS ===
//
// Static (stack, in-place init): input tokens, output tokens, and the count
// metadata (expert_counters for expert-major / src_rank_counters for rank-major,
// plus the rank-major-only per-slot topk outputs and per-token input weights).

// Input: tokens [B x H].
size_t in_tokens_dims[2] = { num_tokens, hidden };
void*  in_tokens_data    = nullptr;
cudaMalloc(&in_tokens_data, num_tokens * hidden * sizeof(uint16_t));
ncclEpTensor_t in_tokens = { NCCL_EP_TENSOR_INIT_INLINE,
                             .ndim = 2, .datatype = ncclBfloat16,
                             .data = in_tokens_data,
                             .sizes = in_tokens_dims };

// Dispatch-output tokens are 3D in both layouts, but the leading dimension differs:
//   expert-major: [num_local_experts, nRanks * max_dispatch_tokens_per_rank, hidden]
//   rank-major:   [nRanks,            max_dispatch_tokens_per_rank,          hidden]
size_t out_tokens_dims[3];
if (expert_major) {
    out_tokens_dims[0] = num_local_experts;
    out_tokens_dims[1] = (size_t)nRanks * config.max_dispatch_tokens_per_rank;
    out_tokens_dims[2] = hidden;
} else {  // rank-major
    out_tokens_dims[0] = (size_t)nRanks;
    out_tokens_dims[1] = config.max_dispatch_tokens_per_rank;
    out_tokens_dims[2] = hidden;
}
void*  out_tokens_data    = nullptr;
cudaMalloc(&out_tokens_data,
           out_tokens_dims[0] * out_tokens_dims[1] * out_tokens_dims[2] * sizeof(uint16_t));
ncclEpTensor_t out_tokens = { NCCL_EP_TENSOR_INIT_INLINE,
                              .ndim = 3, .datatype = ncclBfloat16,
                              .data = out_tokens_data,
                              .sizes = out_tokens_dims };

ncclEpDispatchInputs_t  dispatch_in  = NCCL_EP_DISPATCH_INPUTS_INIT;
ncclEpDispatchOutputs_t dispatch_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
ncclEpLayoutInfo_t      layout_info  = NCCL_EP_LAYOUT_INFO_INIT;
dispatch_in.tokens  = &in_tokens;    // static; always 2D [B x H]
dispatch_out.tokens = &out_tokens;   // static; 3D (shape chosen above)

// Per-rank-or-expert count metadata: per-local-expert for expert-major,
// per-source-rank for rank-major. Declared here so it outlives the dispatch call.
size_t counters_dims[1] = { expert_major ? (size_t)num_local_experts : (size_t)nRanks };
void*  counters_data    = nullptr;
cudaMalloc(&counters_data, counters_dims[0] * sizeof(int32_t));
ncclEpTensor_t counters = { NCCL_EP_TENSOR_INIT_INLINE,
                            .ndim = 1, .datatype = ncclInt32,
                            .data = counters_data, .sizes = counters_dims };

// Rank-major-only dispatch extras (unused, left NULL, for expert-major).
size_t out_w_dims[3]   = { (size_t)nRanks, config.max_dispatch_tokens_per_rank, top_k };
size_t out_idx_dims[3] = { (size_t)nRanks, config.max_dispatch_tokens_per_rank, top_k };
size_t in_w_dims[2]    = { num_tokens, top_k };
void*  out_w_data = nullptr, *out_idx_data = nullptr, *in_w_data = nullptr;
ncclEpTensor_t out_topk_weights, out_topk_idx, in_topk_weights;

if (expert_major) {
    // Expert-major: per-expert recv counts. Routing weights are applied on the
    // receive side during combine (see combine_out.topk_weights below), so the
    // dispatch carries no per-slot topk metadata and no input weights.
    layout_info.expert_counters = &counters;
} else {  // rank-major
    // Rank-major: per-source-rank counts (note: src_rank_counters, NOT expert_counters).
    layout_info.src_rank_counters = &counters;

    // Rank-major REQUIRES dispatch to also return per-slot routing metadata, both
    // 3D [nRanks, max_dispatch_tokens_per_rank, top_k]:
    cudaMalloc(&out_w_data,   out_w_dims[0]   * out_w_dims[1]   * out_w_dims[2]   * sizeof(float));
    cudaMalloc(&out_idx_data, out_idx_dims[0] * out_idx_dims[1] * out_idx_dims[2] * sizeof(int32_t));
    out_topk_weights = (ncclEpTensor_t){ NCCL_EP_TENSOR_INIT_INLINE,
                                         .ndim = 3, .datatype = ncclFloat32,
                                         .data = out_w_data, .sizes = out_w_dims };
    out_topk_idx     = (ncclEpTensor_t){ NCCL_EP_TENSOR_INIT_INLINE,
                                         .ndim = 3, .datatype = ncclInt32,
                                         .data = out_idx_data, .sizes = out_idx_dims };
    dispatch_out.topk_weights = &out_topk_weights;
    dispatch_out.topk_idx     = &out_topk_idx;

    // ... and per-token routing weights on the dispatch INPUT.
    cudaMalloc(&in_w_data, num_tokens * top_k * sizeof(float));
    in_topk_weights = (ncclEpTensor_t){ NCCL_EP_TENSOR_INIT_INLINE,
                                        .ndim = 2, .datatype = ncclFloat32,
                                        .data = in_w_data, .sizes = in_w_dims };
    dispatch_in.topk_weights = &in_topk_weights;
}

// Dispatch tokens to experts (staged execution for overlap)
ncclEpDispatchConfig_t dispatch_cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
dispatch_cfg.send_only = 1;
ncclEpDispatch(handle, &dispatch_in, &dispatch_out,
               &layout_info, &dispatch_cfg, stream);

// Overlap with other computation...
// doOtherWork(stream);

// Wait for dispatch to complete
ncclEpComplete(handle, /*config=*/NULL, stream);
cudaStreamSynchronize(stream);

// Expert forward computation:
//   expert-major: out_tokens is [experts x slots x hidden]; use the per-expert
//                 counts to size each expert's valid range.
//   rank-major:   out_tokens is [ranks x slots x hidden]; use out_topk_idx to route
//                 each slot to its local expert(s) and the per-rank counts for valid
//                 ranges. The rank-major combine kernel applies weight = 1, so the
//                 caller MUST pre-reduce (apply out_topk_weights) before combine.

// Combine input: post-processed activation, SAME 3D shape as dispatch_out.tokens.
void*  combine_in_data = nullptr;
cudaMalloc(&combine_in_data,
           out_tokens_dims[0] * out_tokens_dims[1] * out_tokens_dims[2] * sizeof(uint16_t));
ncclEpTensor_t combine_in_tokens = {
    NCCL_EP_TENSOR_INIT_INLINE,
    .ndim = 3, .datatype = ncclBfloat16,
    .data = combine_in_data,
    .sizes = out_tokens_dims };

// Combine output: [B x H] back to original token order.
size_t combine_out_dims[2] = { num_tokens, hidden };
void*  combine_out_data    = nullptr;
cudaMalloc(&combine_out_data, num_tokens * hidden * sizeof(uint16_t));
ncclEpTensor_t combine_out_tokens = {
    NCCL_EP_TENSOR_INIT_INLINE,
    .ndim = 2,
    .datatype = ncclBfloat16,
    .data = combine_out_data,
    .sizes = combine_out_dims };

ncclEpCombineInputs_t  combine_in  = NCCL_EP_COMBINE_INPUTS_INIT;
ncclEpCombineOutputs_t combine_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
combine_in.tokens  = &combine_in_tokens;
combine_out.tokens = &combine_out_tokens;

// Expert-major applies the per-token routing weights on the receive side, so it
// REQUIRES combine_out.topk_weights [B x top_k]. Rank-major already applied the
// weights in the caller's pre-reduction, so it leaves this field NULL.
size_t combine_out_w_dims[2] = { num_tokens, top_k };
void*  combine_out_w_data    = nullptr;
ncclEpTensor_t combine_out_weights;
if (expert_major) {
    cudaMalloc(&combine_out_w_data, num_tokens * top_k * sizeof(float));
    combine_out_weights = (ncclEpTensor_t){ NCCL_EP_TENSOR_INIT_INLINE,
                                            .ndim = 2, .datatype = ncclFloat32,
                                            .data = combine_out_w_data,
                                            .sizes = combine_out_w_dims };
    combine_out.topk_weights = &combine_out_weights;
}

ncclEpCombineConfig_t combine_cfg = NCCL_EP_COMBINE_CONFIG_INIT;
combine_cfg.send_only = 1;
ncclEpCombine(handle, &combine_in, &combine_out, &combine_cfg, stream);

ncclEpComplete(handle, /*config=*/NULL, stream);
cudaStreamSynchronize(stream);

// Cleanup
ncclEpHandleDestroy(handle);
ncclEpGroupDestroy(ep_group);
// Dynamic descriptor: free device buffer, then descriptor.
cudaFree(topk_idx->data);
ncclEpTensorDestroy(topk_idx);
// Static descriptors: free device buffers only (descriptors and sizes are stack locals).
cudaFree(in_tokens_data);
cudaFree(out_tokens_data);
cudaFree(counters_data);
cudaFree(combine_in_data);
cudaFree(combine_out_data);
if (expert_major) {
    cudaFree(combine_out_w_data);   // expert-major receive-side weights
} else {                            // rank-major dispatch extras
    cudaFree(out_w_data);
    cudaFree(out_idx_data);
    cudaFree(in_w_data);
}
ncclCommDestroy(comm);
cudaStreamDestroy(stream);
```
