```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# PyTorch Symmetric Memory

:::{note}
`torch.distributed._symmetric_memory` is currently in alpha state and under
development. Its Python APIs and every operator in `torch.ops.symm_mem` may
change without notice. A dispatcher-visible operator is not necessarily a
user-facing API. The status table below identifies which operators are intended
for direct use and which are implementation details.
:::

## API status and operator guide

The status labels below describe how an operator is intended to be used *within
the alpha API*. They are not backward-compatibility guarantees:

- **Supported alpha** operators have general-purpose interfaces and regular
  correctness coverage. They are the preferred Symmetric Memory building
  blocks, but are not stable APIs yet.
- **Specialized alpha** operators have narrower hardware, backend, dtype, or
  layout requirements. Validate them on the target system before depending on
  them.
- **Experimental** operators are incomplete or primarily compiler-facing. Use
  them only when you can tolerate interface and support changes.
- **Internal** operators are compiler targets or implementation details. They
  have no compatibility guarantee and should not be called directly.

| Operations | Status | When to use them |
| --- | --- | --- |
| `empty`, `rendezvous`, `get`, and `get_mem_pool` | Supported alpha | Allocate, establish, access, and reuse symmetric memory. |
| `fused_all_gather_matmul` and `fused_matmul_reduce_scatter` | Supported alpha | Overlap an unscaled matrix multiplication with a tensor-parallel collective. Ordinary accelerator tensors are accepted; the operators manage a symmetric workspace. These are also targets of the compiler's tensor-parallel fusion pass. |
| `fused_all_gather_scaled_matmul` and `fused_scaled_matmul_reduce_scatter` | Experimental | FP8 or other scaled matrix multiplication when the accelerator supports `aten._scaled_mm`. Their interfaces closely follow compiler lowering requirements. |
| `one_shot_all_reduce`, `one_shot_all_reduce_out`, and `two_shot_all_reduce_` | Supported alpha | Low-latency, direct-access all-reduce on a symmetric tensor. |
| `multimem_all_reduce_` and `multimem_all_gather_out` | Specialized alpha | NVIDIA systems with multicast and multimem support, such as NVLink SHARP. |
| `multimem_one_shot_all_reduce`, `multimem_one_shot_all_reduce_out`, and `multimem_one_shot_reduce_out` | Experimental | Compiler-oriented multimem variants. Their reduction order is not fixed, so all ranks are not guaranteed to receive bitwise-identical results. |
| `get_remote_tensors` | Specialized alpha | Create local tensor views of peer allocations when every peer is directly addressable. The views require explicit cross-rank synchronization. |
| `nvshmem_*`, `all_to_all_vdev*`, `tile_reduce`, and `multi_root_tile_reduce` | Specialized alpha | NVSHMEM or rocSHMEM transport and device-driven collectives. Availability differs by operation and SHMEM implementation. |
| `reduce_scatter_offset` and `all_to_all_nd` | Specialized alpha | NCCL-backend routing and multidimensional all-to-all. |
| `put_signal` and `wait_signal` | Experimental | One-sided signaling; currently implemented only for the NCCL backend. |
| `_low_contention_*`, `_async_input_mm`, `_rendezvous`, and `_barrier` | Internal | Compiler and dispatcher implementation details. |
| `one_shot_all_reduce_copy`, `one_shot_all_reduce_copy_out`, `two_shot_all_reduce_out`, and `reduce_scatter_out` | Internal | Compiler buffer-planning variants. Use the documented allocating or in-place operator instead. |
| `stream_write_value32_`, `memset32_`, `memcpy_to_multicast_`, and raw `nccl_*` operators | Internal | Kernel composition or backend dispatch. Use a documented Python wrapper instead. |

The scaled fused operators, hardware-specific collectives, and internal
operators are the least portable parts of the surface. In particular, do not
infer support from an operator being present in `torch.ops.symm_mem`: some
kernels are compiled only for particular backends or require runtime hardware
features.

The `torch.ops.symm_mem` namespace also does not provide a stable public
autograd contract. For training graphs, prefer expressing the unfused
collective and computation with higher-level PyTorch APIs and allowing the
compiler to select a Symmetric Memory fusion when applicable.

(symmetric-memory-stream-ordering)=
### Stream ordering

Built-in CUDA-backend barriers, signal operations, and collectives serialize
their signal-pad use across streams for each process group and device. This
serialization also participates in CUDA graph capture. It does not cover
NVSHMEM operations, `ncclPutSignal`/`ncclWaitSignal`, or custom kernels that use
a raw signal-pad tensor. Issue those uncovered operations for a group from one
stream, or explicitly order their streams.

## Why Symmetric Memory?

With rapidly evolving parallelization techniques, existing frameworks and
libraries often struggle to keep up, and developers increasingly rely on custom
implementations directly scheduling communications and computations. In recent
years we’ve witnessed a shift from primarily relying on one-dimensional
data-parallelism techniques to multi-dimensional parallelism ones. The latter
have different latency requirements for different types of communications and
thus require fine-grained overlapping of compute and communications.

To minimize compute interference, they also require the use of copy engines and
network interface cards (NICs) to drive communication. Network transport
protocols such as remote direct memory access (RDMA) enhance the performance by
enabling direct, high-speed, and low-latency communication between processors
and memory. This increase in variety indicates the need for finer-grained
communication primitives than are offered today by high-level collective APIs,
ones that would enable developers to implement specific algorithms tailored for
their use cases, such as low-latency collectives, fine-grained
compute-communications overlap, or custom fusions.

Furthermore, today’s advanced AI systems connect GPUs with high-bandwidth links
(such as NVLinks, InfiniBand or RoCE), making GPU global memory directly
accessible to peers. Such connections present a great opportunity for
programmers to program the system as a single, gigantic GPU with vast accessible
memory, instead of programming singular “GPU islands.”

In this document, we will show how you can use PyTorch Symmetric Memory to
program modern GPU systems as a “single GPU” and achieve fine-grained remote
access.

## What PyTorch Symmetric Memory unlocks?

PyTorch Symmetric Memory unlocks three new capabilities:

- **Customized communication patterns**: Increased flexibility in kernel writing
allows developers to write custom kernels that implement their custom
computations and communications, directly tailored to the need of the
application. It will also be straightforward to add support for new data types
along with the special compute that those data types might require, even if it’s
not present yet in the standard libraries.

- **In-kernel compute-comm fusion**: Device-initiated communication capability
allows developers to write kernels with both computation and communication
instructions, allowing for the fusion of computation and data movement in the
smallest possible granularity.

- **Low-latency remote access**: Network transport protocols like RDMA enhance the
performance of symmetric memory in networked environments by enabling direct,
high-speed, and low-latency communication between processors and memory. RDMA
eliminates the overhead associated with the traditional network stack and CPU
involvement. It also offloads data transfer from the compute to the NICs,
freeing up compute resources for computational tasks.

Next, we will show you how PyTorch Symmetric Memory (SymmMem) enables new
applications with the above capabilities.

## A “Hello World” example

The PyTorch SymmMem programming model involves two key elements:

- creating symmetric tensors
- creating SymmMem kernels

To create symmetric tensors, one can use the
`torch.distributed._symmetric_memory` package:

```python
import torch.distributed._symmetric_memory as symm_mem

t = symm_mem.empty(128, device=torch.device("cuda", rank))
hdl = symm_mem.rendezvous(t, group)
```

The `symm_mem.empty` function creates a tensor that is backed by a symmetric
memory allocation. The `rendezvous` function establishes a rendezvous with peers
in the group, and returns a handle to the symmetric memory allocation. The
handle provides method to access information related to the symmetric memory
allocation, such as pointers to symmetric buffer on peer ranks, multicast
pointer (if supported), and signal pads.

The `empty` and `rendezvous` functions must be called in the same order on all
ranks in the group.

Then, collectives can be called on these tensors. For example, to perform a
one-shot all-reduce:

```python
# Most SymmMem ops are under the torch.ops.symm_mem namespace
torch.ops.symm_mem.one_shot_all_reduce(t, "sum", group)
```

Please note that `torch.ops.symm_mem` is an "op namespace" instead of a python
module. Therefore, you can't import it by `import torch.ops.symm_mem`, neither
can you import an op by `from torch.ops.symm_mem import one_shot_all_reduce`.
You can call the op directly as in the example above.

## Write your own kernel

To write your own kernel doing communications with symmetric memory, you’ll need
access to the addresses of mapped peer buffers and access to signal pads that
are required for synchronization. In the kernel you’ll also need to perform
correct synchronizations to make sure that peers are ready for communication,
and signal to them that this GPU is ready.

PyTorch Symmetric Memory provides CUDA Graph-compatible synchronization
primitives that operate on the signal pad accompanying each symmetric memory
allocation. Kernels using symmetric memory can be written both in CUDA and in
Triton. Here’s an example allocating symmetric tensor and exchanging handles:

```python
import torch.distributed._symmetric_memory as symm_mem

dist.init_process_group()
rank = dist.get_rank()

# Allocate a tensor
t = symm_mem.empty(4096, device=f"cuda:{rank}")
# Establish symmetric memory and obtain the handle
hdl = symm_mem.rendezvous(t, dist.group.WORLD)
```

Access to buffer pointers, multimem pointer, and signal pads is provided via:

```python
hdl.buffer_ptrs
hdl.multicast_ptr
hdl.signal_pad_ptrs
```

Data pointed to by `buffer_ptrs` can be accessed just like regular local data,
and any necessary compute can also be performed in the usual ways. As with local
data, you can and should use vectorized accesses to improve efficiency.

Symmetric memory is especially convenient for writing kernels in Triton. While
previously Triton removed the barriers to writing efficient CUDA code, now
communications can be added easily to Triton kernels. The kernel below
demonstrates a low-latency, all-reduce kernel written in Triton.

```python
@triton.jit
def one_shot_all_reduce_kernel(
    buf_tuple,
    signal_pad_ptrs,
    output_ptr,
    numel: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    ptx_utils.symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasSubsequenceMemAccess=True
    )

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    while block_start < numel:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.bfloat16)

        for i in tl.static_range(world_size):
            buffer_rank = buf_tuple[i]
            x = tl.load(buffer_rank + offsets, mask=mask)
            acc += x

        tl.store(output_ptr + offsets, acc, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    ptx_utils.symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasPreviousMemAccess=True
    )
```

Synchronizations at the beginning and the end of the kernel above guarantee that
all the processes see consistent data. The bulk of the kernel is recognizable
Triton code, and Triton will optimize it behind the scene, making sure memory
accesses are performed in an efficient way with vectorization and unrolling. As
with all Triton kernels, it is easily modifiable to add extra computations or
change the communication algorithm. Visit
https://github.com/meta-pytorch/kraken/blob/main/kraken to see additional
utilities and examples of using symmetric memory to implement common patterns in
Triton.

## One-sided get

Symmetric memory also exposes a small one-sided `get` API for copying data from
a peer's symmetric allocation into a local tensor:

```python
src = symm_mem.empty(1024, device=device)
hdl = symm_mem.rendezvous(src, group)

if dist.get_rank(group) == 0:
    dst = torch.empty((512,), device=device)
    # Copy the last 512 elements of the peer's allocation into dst.
    symm_mem.get(dst, hdl, peer=1, offset=512)
```

`hdl` is the symmetric memory handle returned by `rendezvous`; the remote
source is the peer's allocation backing that handle. The number of elements
copied is inferred from `dst`, so pass a view (e.g. `dst[:n]`) to fill only
part of a tensor; `offset` is given in elements of `dst`'s dtype and defaults
to `0`. `dst` may be a regular CUDA tensor or another symmetric tensor; it must
be on the same device as `hdl` and backed by contiguous memory. The copy is
issued on the current CUDA stream.

## Scale out

Large language models distribute experts onto more than 8 GPUs, hence requiring
multi-node access capability. NICs capable of RDMA come to help. In addition,
software libraries such as NVSHMEM or rocSHMEM abstract away the programming
difference between intra-node access and inter-node access with primitives that
are slightly higher level than pointer access, such as put and get.

PyTorch provides NVSHMEM plugins to augment Triton kernels’ cross-node
capabilities. As shown in the code snippet below, one can initiate a cross-node
put command within the kernel.

```python
import torch.distributed._symmetric_memory._nvshmem_triton as nvshmem
from torch.distributed._symmetric_memory._nvshmem_triton import requires_nvshmem

@requires_nvshmem
@triton.jit
def my_put_kernel(
    dest,
    src,
    nelems,
    pe,
):
    nvshmem.put(dest, src, nelems, pe)
```

The `requires_nvshmem` decorator is used to indicate that the kernel requires
the NVSHMEM device library as an external dependency. When Triton compiles the
kernel, the decorator will search your system paths for the NVSHMEM device
library. If it is available, Triton will include the necessary device assembly
to use the NVSHMEM functions.

## Using Memory Pool

Memory pool allows PyTorch SymmMem to cache memory allocations that have been
rendezvoused, saving time when creating new tensors.  For convenience, PyTorch
SymmMem has added a `get_mem_pool` API to return a symmetric memory pool. Users
can use the returned MemPool with the `torch.cuda.use_mem_pool` context manager.
In the example below, tensor `x` will be created from symmetric memory:

```python
    import torch.distributed._symmetric_memory as symm_mem

    mempool = symm_mem.get_mem_pool(device)

    with torch.cuda.use_mem_pool(mempool):
        x = torch.arange(128, device=device, dtype=torch.float32)

    torch.ops.symm_mem.one_shot_all_reduce(x, "sum", group_name)
```

Similarly, you can put a compute operation under the MemPool context, and the
result tensor will be created from symmetric memory too.

```python
    dim = 1024
    w = torch.ones(dim, dim, device=device)
    x = torch.ones(1, dim, device=device)

    mempool = symm_mem.get_mem_pool(device)
    with torch.cuda.use_mem_pool(mempool):
        # y will be in symmetric memory
        y = torch.mm(x, w)
```

As of torch 2.11, the `CUDA` and `NVSHMEM` backends support MemPool. MemPool
support of the `NCCL` backend is in progress.

:::{note}
The pool returned by `get_mem_pool` feeds `torch.ops.symm_mem.*`
kernels, though it does not register the allocation with NCCL. To drive
`dist.*` collectives onto NCCL's symmetric memory
backed kernels, you can register the mempool for NCCL to auto-select.
For more details, see [NCCL Symmetric Kernels](nccl-symmetric-kernels).
:::

(nccl-symmetric-kernels)=

## NCCL Symmetric Kernels

:::{note}
Requires NCCL 2.27 or later and a single NVLink domain (every rank reachable
over direct NVLink).
:::

NCCL 2.27+ added a family of device kernels — "SymK" internally — written
specifically for symmetric, window-registered buffers. Because each rank knows
every peer's buffer address up front, these kernels skip the generic
proxy/ring machinery and instead use LL (low-latency), multimem/NVLS, and TMA
variants. NCCL picks one per call from message size, so the same
`dist.all_reduce` gets a latency-optimized kernel for small messages and a
bandwidth-optimized one for large ones.

Symmetric kernels are driven through the *standard* collective API —
`dist.all_reduce`, `dist.all_gather_into_tensor`, `dist.reduce_scatter_tensor` —
with no change at the call site. What matters is that the buffers were
registered with NCCL as symmetric windows. There are two ways to arrange that.

### Option 1: register a memory pool with the process group

This route puts NCCL's allocator behind a {class}`torch.cuda.MemPool`, so *any*
tensor allocated inside the pool's context is window-registered, including
tensors produced by compute ops. It is usually the better fit for an existing
model, since allocations do not have to be rewritten as `symm_mem.empty`.

```python
import torch
import torch.distributed as dist

device = torch.device("cuda", rank)

# `device_id` eagerly initializes the NCCL communicator. `register_mem_pool`
# requires a communicator that already exists, and raises otherwise.
dist.init_process_group(backend="nccl", device_id=device)
pg = dist.group.WORLD

backend = dist.get_backend_impl(pg, device)

# A MemPool backed by `ncclMemAlloc` / `ncclMemFree`.
pool = torch.cuda.MemPool(backend.mem_allocator)

# `symm=True` registers each segment with `ncclCommWindowRegister` using
# `NCCL_WIN_COLL_SYMMETRIC`, which is what makes the symmetric kernels
# eligible. The default `symm=False` performs ordinary user-buffer
# registration, which does not.
backend.register_mem_pool(pool, symm=True)

with torch.cuda.use_mem_pool(pool):
    x = torch.ones(1024 * 1024, dtype=torch.bfloat16, device=device)

# Dispatches to a NCCL symmetric kernel.
dist.all_reduce(x, op=dist.ReduceOp.SUM)

# De-register before the pool is torn down.
backend.deregister_mem_pool(pool)
```

`register_mem_pool` registers the segments already in the pool *and* installs an
allocator hook, so later allocations in the pool are registered as well.

### Option 2: allocate through the NCCL symmetric memory backend

If the tensors are already symmetric-memory tensors — for example because
custom kernels need the handle, its peer pointers, or its signal pads — select
the `NCCL` backend and rendezvous as usual. `rendezvous` window-registers the
allocation, so `dist.*` collectives on it become eligible too.

```python
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

symm_mem.set_backend("NCCL")

x = symm_mem.empty(1024 * 1024, dtype=torch.bfloat16, device=device)
symm_mem.rendezvous(x, group=dist.group.WORLD.group_name)

dist.all_reduce(x, op=dist.ReduceOp.SUM)
```

### When NCCL uses a symmetric kernel

Only these collective / reduction / dtype combinations currently have a symmetric
implementation:

| Collective | Reduction ops | Data types |
| --- | --- | --- |
| `all_gather` | n/a | any |
| `all_reduce` | `SUM`, `AVG` | `float32`, `float16`, `bfloat16`, `float8_e4m3fn`, `float8_e5m2` |
| `reduce_scatter` | `SUM`, `AVG` | `float32`, `float16`, `bfloat16`, `float8_e4m3fn`, `float8_e5m2` |

Note in particular that `float64` and the integer dtypes are excluded for the
two reducing collectives, as are `MIN` / `MAX` / `PRODUCT`. Collectives outside
the table (`broadcast`, `reduce`, `all_to_all`, point-to-point) currently have no
symmetric implementation, and fall back to the regular ring/tree path silently.

To confirm, you can use NCCL logs to check kernel names:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING python train.py
```

```
AllReduce [Symmetric]: 2097152 Bytes -> Kernel AllReduce_RSxLDMC_AGxSTMC nchannels 16 nthreads 512 nWorks 1
```

Also, if you are looking at a profiler,
device kernel names should resemble `ncclSymkDevKernel_*`.
For example, `ncclSymkDevKernel_AllReduce_AGxLLMC_R_sum_bf16`, as opposed to
`ncclDevKernel_*` for the generic NCCL path.

### CFT logical-endpoint handles

:::{note}
Requires NCCL 2.31.2+, a Blackwell-class GPU (sm_100 or newer), and a driver
reporting CUDA 13.3 or later (an r610-or-newer driver). NCCL itself must also
be built with CUDA 13.3+.
:::

CFT (Compute Fabric Transport) exposes a peer's window-registered memory as a
*logical endpoint*: an opaque `(le_id, le_offset)` pair that a custom device
kernel can hand to the `ncclCft` put/get/reduce device API to reach that
peer's copy of a symmetric buffer — without constructing a `ncclDevComm`.
When the symmetric-memory backend is `NCCL`, the rendezvous handle exposes
the coordinates:

- `hdl.get_peer_cft_handle(peer)` returns the `(le_id, le_offset)` pair
  addressing `peer`'s copy of the buffer.
- `hdl.get_multimem_cft_handle()` returns the multicast endpoint (requires
  NVLS; the first call may be collective, so all ranks must reach it).

Handles are only meaningful for the group the tensor was rendezvoused with —
each group owns a separate set of logical endpoints over the same allocation.

Two knobs control availability:

- `host_cft_mode` on the communicator config
  (`ProcessGroupNCCL.Options().config.host_cft_mode`) decides whether the
  endpoints are created and what happens when the stack cannot support them:
  `1` (enable — fail communicator init if unsupported), `2` (disable), `3`
  (fallback — create them if possible, silently proceed without otherwise).
  The default is **disable**: host-side CFT is opt-in per communicator, since
  endpoints are a limited per-device resource. The mode must be identical on
  every rank and must be set before the communicator is created — the
  endpoints are made during window registration.
- The `NCCL_CFT_ENABLE` environment variable (default `1`) is NCCL's global
  kill switch; `NCCL_CFT_ENABLE=0` makes NCCL report no CFT support
  regardless of `host_cft_mode`.

Under `host_cft_mode=3` (fallback), an unsupported GPU, driver, or NCCL build
is not an error at init — the handle queries simply raise `RuntimeError`.

(copy-engine-collectives)=

## Copy Engine Collectives

:::{note}
Copy Engine Collectives require NCCL 2.28 or later, and GPUs with peer-to-peer (P2P) access.
:::

Copy Engine (CE) Collectives are an optimization for NCCL collective operations that offload
data movement to the GPU's copy engines (DMA engines) instead of using CUDA streaming
multiprocessors (SMs). This frees up SMs for compute work, enabling better overlap of
communication and computation during distributed training.

To use CE collectives, you need to:

1. Configure the NCCL process group with the zero-CTA policy
2. Set up symmetric memory with the NCCL backend
3. Allocate tensors using symmetric memory
4. Register the tensors with symmetric memory via rendezvous

Once set up, standard collective functions like {func}`all_gather_single` and
{func}`all_to_all_single` will automatically use the copy engines when operating
on symmetric memory tensors.

**Example**

```
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# Initialize process group with zero-CTA policy for CE collectives
opts = dist.ProcessGroupNCCL.Options()
opts.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
device = torch.device("cuda", rank)
dist.init_process_group(backend="nccl", pg_options=opts, device_id=device)

# Set up symmetric memory with NCCL backend
symm_mem.set_backend("NCCL")
group_name = dist.group.WORLD.group_name

# Allocate tensors using symmetric memory
numel = 1024 * 1024
inp = symm_mem.empty(numel, device=device)
out = symm_mem.empty(numel * world_size, device=device)

# Register tensors for symmetric memory operations
symm_mem.rendezvous(inp, group=group_name)
symm_mem.rendezvous(out, group=group_name)

# Perform collective operation using copy engines
# This now runs on DMA engines instead of SMs
work = dist.all_gather_single(out, inp, async_op=True)
work.wait()
```

**Benefits**

- **SM offloading**: Communication runs on copy engines, leaving SMs free for computation
- **Better overlap**: Enables more efficient computation/communication overlap
- **Transparent API**: Uses the same collective API, just with symmetric memory tensors

**Requirements and Limitations**

- NCCL version 2.28 or later
- GPUs must have peer-to-peer (P2P) access enabled
- Tensors must be allocated using {func}`torch.distributed._symmetric_memory` and rendezvoused
- The NCCL process group must be configured with `NCCL_CTA_POLICY_ZERO` or the
environment variable `NCCL_CTA_POLICY` be set to 2
- As of NCCL 2.28, CE collectives cannot run with the default stream, so you
would need to use the `async_op=True` flag to activate the internal stream of
`ProcessGroupNCCL` or create a side stream yourself

(higher-precision-reduction)=

## Higher-Precision Reduction

When tensors are allocated with symmetric memory, NCCL's symmetric kernel
implementation enables internal reduction with higher precision. For example, with
BF16 inputs, NCCL will automatically accumulate in FP32 internally before producing
BF16 outputs (BF16 in → FP32 accumulate → BF16 out). This improves numerical
accuracy of reduction operations without changing the collective call.

**Scope**

- **Applicable operations**: ``reduce_scatter`` and ``all_reduce`` only
- **Domain**: Within the NVLink domain as of torch 2.9 (NCCL 2.27);
  NVLink + network for ``reduce_scatter`` as of torch 2.11 (NCCL 2.29)
- **Precision**: BF16/FP16 in → FP32 internal accumulation → BF16/FP16 out

**Example**

```python
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# Allocate tensors using NCCL symmetric memory
symm_mem.set_backend("NCCL")
inp = symm_mem.empty(1024, 1024, device=device, dtype=torch.bfloat16)
symm_mem.rendezvous(inp, group_name)

# reduce_scatter and all_reduce on symmetric memory tensors
# automatically benefit from FP32 internal accumulation
dist.all_reduce(inp)
```

:::{note}
This higher-precision accumulation is enabled transparently by NCCL when
using symmetric memory tensors. No additional configuration is required
beyond the symmetric tensor creation and rendezvous described above. This
currently applies to ``reduce_scatter`` and ``all_reduce`` within the
supported domains only; other collectives (e.g., ``all_gather``) and
inter-node communication are not affected.
:::

## Rendezvous at Scale

By default, `rendezvous` exchanges metadata via the TCPStore. Each rank in the
symmetric memory group issues one store set and N-1 store gets (where N is the
group size, typically 8–72 for NVLink domains). At large world sizes the
TCPStore (~200k QPS capacity) becomes a bottleneck: for example, with 72-rank
NVLink groups at 10k total ranks, a single rendezvous takes ~3.6s via TCPStore;
at 100k ranks this grows to ~36s.

To use the process group's NCCL allgather instead, set
`use_pg_for_symm_mem_rendezvous` in the process group options:

```python
opts = dist.ProcessGroupNCCL.Options()
opts.use_pg_for_symm_mem_rendezvous = True
pg = dist.new_group(ranks, pg_options=opts)

t = symm_mem.empty(size, device=device)
hdl = symm_mem.rendezvous(t, group=pg)
```

If the process group is only used for symmetric memory and won't be used for
regular collectives afterwards (e.g., an expert-parallelism group), you can
release the NCCL communicator after rendezvous via ``abort()``. The symmetric
memory handle remains usable since it only depends on the mapped memory, not the
communicator:

```python
opts = dist.ProcessGroupNCCL.Options()
opts.use_pg_for_symm_mem_rendezvous = True
ep_pg = dist.new_group(ep_ranks, pg_options=opts)

t = symm_mem.empty(size, device=device)
hdl = symm_mem.rendezvous(t, group=ep_pg)

# Release the NCCL communicator since ep_pg won't be used for collectives.
# The symm_mem handle is still usable — it only needs the mapped memory.
ep_pg.abort()
```

:::{note}
Enabling `use_pg_for_symm_mem_rendezvous` will lazily create the NCCL
communicator for the process group if it doesn't already exist.
:::

## API Reference

```{eval-rst}
.. currentmodule:: torch.distributed._symmetric_memory
```

```{eval-rst}
.. autofunction:: empty
```

```{eval-rst}
.. autofunction:: rendezvous
```

```{eval-rst}
.. autofunction:: get
```

```{eval-rst}
.. autofunction:: put_signal
```

```{eval-rst}
.. autofunction:: wait_signal
```

```{eval-rst}
.. autofunction:: is_nvshmem_available
```

```{eval-rst}
.. autofunction:: set_backend
```

```{eval-rst}
.. autofunction:: get_backend
```

```{eval-rst}
.. autofunction:: get_mem_pool
```

```{eval-rst}
.. autofunction:: is_symm_mem_tensor
```

```{eval-rst}
.. autofunction:: set_signal_pad_size
```

```{eval-rst}
.. autofunction:: get_signal_pad_size
```

```{eval-rst}
.. autofunction:: all_to_all_nd
```

## Op Reference
:::{note}
The following ops are hosted in the `torch.ops.symm_mem` namespace. You can call
them directly via `torch.ops.symm_mem.<op_name>`.
:::

```{eval-rst}
.. currentmodule:: torch.distributed._symmetric_memory
```

```{eval-rst}
.. autofunction:: reduce_scatter_offset
```

```{eval-rst}
.. currentmodule:: torch.ops.symm_mem
```

```{eval-rst}
.. py:function:: fused_all_gather_matmul(A_shard: Tensor, Bs: list[Tensor], gather_dim: int, group_name: str, *, return_A: bool = True) -> tuple[Tensor | None, list[Tensor]]

    Overlaps an all-gather of ``A_shard`` with one or more matrix
    multiplications. It is semantically equivalent to::

        A = all_gather_single(A_shard, gather_dim, group)
        outputs = [torch.matmul(A, B) for B in Bs]

    The tensors in ``Bs`` must be 2-D and compatible with ``A`` for matrix
    multiplication. ``A_shard`` may be an ordinary accelerator tensor; the
    operator allocates and reuses symmetric workspace internally. All ranks
    must call the operator with matching shapes and arguments.

    For best performance, arrange ``A_shard`` so that
    ``A_shard.movedim(gather_dim, 0)`` is contiguous. Set ``return_A=False``
    when only the matrix-multiplication results are needed; this can avoid
    materializing the gathered tensor on supported systems. Run the operator
    once before CUDA graph capture so that its workspace is large enough.

    :param Tensor A_shard: Rank-local shard of the left matrix.
    :param list[Tensor] Bs: Right-hand matrices. Multiple matrices reuse the
        same gathered input.
    :param int gather_dim: Dimension of ``A_shard`` to gather.
    :param str group_name: Name of the process group.
    :param bool return_A: Whether to return the gathered ``A``. Defaults to
        ``True``.
    :returns: The gathered tensor (or ``None``) and the list of matrix-
        multiplication results.


.. py:function:: fused_all_gather_scaled_matmul(A_shard: Tensor, Bs: list[Tensor], A_scale: Tensor, B_scales: list[Tensor], gather_dim: int, group_name: str, biases: list[Tensor | None], result_scales: list[Tensor | None], out_dtypes: list[dtype | None], use_fast_accum: list[bool]) -> tuple[Tensor, list[Tensor]]

    Scaled-matrix-multiplication variant of
    :func:`fused_all_gather_matmul`. Each entry of ``B_scales``, ``biases``,
    ``result_scales``, ``out_dtypes``, and ``use_fast_accum`` corresponds to
    the entry at the same position in ``Bs`` and is forwarded to
    ``aten._scaled_mm``.

    ``A_scale`` may be a tensor-wise scalar, a row-wise scale sharded like
    ``A_shard``, or a row-wise scale for the fully gathered tensor. The exact
    dtype, scale-layout, and accelerator requirements are those of
    ``aten._scaled_mm``. This is a specialized compiler fusion target; support
    for a particular FP8 format or shape must be checked on the target
    accelerator.

    :returns: The gathered ``A`` and the list of scaled-matmul results.


.. py:function:: fused_matmul_reduce_scatter(A: Tensor, B: Tensor, reduce_op: str, scatter_dim: int, group_name: str) -> Tensor

    Overlaps matrix multiplication with a reduce-scatter. It is semantically
    equivalent to::

        C = torch.matmul(A, B)
        out = reduce_scatter_single(C, reduce_op, scatter_dim, group)

    ``A`` must have at least two dimensions, ``B`` must be 2-D, and the size
    of the matrix-multiplication result along ``scatter_dim`` must be divisible
    by the process-group size. ``reduce_op`` may be ``"sum"`` or ``"avg"``.
    ``A`` and ``B`` may be ordinary accelerator tensors; symmetric workspace
    is managed internally.

    For best performance, arrange ``A`` so that
    ``A.movedim(scatter_dim, 0)`` is contiguous. Run the operator once before
    CUDA graph capture so that its workspace is large enough.

    :param Tensor A: Left matrix or batch of matrices.
    :param Tensor B: 2-D right-hand matrix.
    :param str reduce_op: ``"sum"`` or ``"avg"``.
    :param int scatter_dim: Result dimension to scatter.
    :param str group_name: Name of the process group.


.. py:function:: fused_scaled_matmul_reduce_scatter(A: Tensor, B: Tensor, A_scale: Tensor, B_scale: Tensor, reduce_op: str, orig_scatter_dim: int, scatter_dim_after_maybe_reshape: int, group_name: str, output_shape: list[int], bias: Tensor | None = None, result_scale: Tensor | None = None, out_dtype: dtype | None = None, use_fast_accum: bool = False) -> Tensor

    Scaled-matrix-multiplication variant of
    :func:`fused_matmul_reduce_scatter`. The two scatter dimensions and
    ``output_shape`` preserve the shape information from the compiler pattern
    ``reshape -> aten._scaled_mm -> reshape -> reduce_scatter``. For a direct
    call without those reshapes, pass the same value for both scatter
    dimensions and pass ``[*A.shape[:-1], B.shape[1]]`` as ``output_shape``.

    This operator is primarily a compiler target. Its supported dtypes, scale
    layouts, and hardware follow ``aten._scaled_mm`` and are narrower than the
    unscaled operator.


.. py:function:: get_remote_tensors(x: Tensor, group_name: str) -> list[Tensor]

    Returns tensor views of the allocation belonging to every rank in
    ``group_name``, ordered by rank. ``x`` must be a symmetric-memory tensor
    and every peer must be directly addressable, for example within one NVLink
    domain. The operation does not work across a network transport.

    The returned tensors alias peer memory. PyTorch does not automatically
    synchronize peer reads and writes through these views; coordinate access
    with a barrier, signal, or another suitable protocol.

    :param Tensor x: Local tensor naming the symmetric allocation.
    :param str group_name: Name of the process group used to rendezvous the
        allocation.


```

```{eval-rst}
.. py:function:: multimem_all_reduce_(input: Tensor, reduce_op: str, group_name: str) -> Tensor

    Performs a multimem all-reduce operation on the input tensor. This operation
    requires hardware support for multimem operations. On NVIDIA GPUs, NVLink
    SHARP is required.

    The built-in CUDA implementation orders this operation with other built-in
    signal-pad operations for the same group. See
    :ref:`symmetric-memory-stream-ordering` for the operations not covered by
    that ordering.

    :param Tensor input: Input tensor to perform all-reduce on. Must be symmetric.
    :param str reduce_op: Reduction operation to perform. Currently only "sum" is supported.
    :param str group_name: Name of the group to perform all-reduce on.


.. py:function:: multimem_all_gather_out(input: Tensor, group_name: str, out: Tensor) -> Tensor

    Performs a multimem all-gather operation on the input tensor. This operation requires hardware support for multimem operations. On NVIDIA GPUs, NVLink SHARP is required.

    :param Tensor input: Input tensor to perform all-gather on.
    :param str group_name: Name of the group to perform all-gather on.
    :param Tensor out: Output tensor to store the result of the all-gather operation. Must be symmetric.


.. py:function:: one_shot_all_reduce(input: Tensor, reduce_op: str, group_name: str) -> Tensor

    Performs a one-shot all-reduce operation on the input tensor.

    :param Tensor input: Input tensor to perform all-reduce on. Must be symmetric.
    :param str reduce_op: Reduction operation to perform. Currently only "sum" is supported.
    :param str group_name: Name of the group to perform all-reduce on.


.. py:function:: one_shot_all_reduce_out(input: Tensor, reduce_op: str, group_name: str, out: Tensor) -> Tensor

    Performs a one-shot all-reduce operation based on the input tensor and writes the result to the output tensor.

    :param Tensor input: Input tensor to perform all-reduce on. Must be symmetric.
    :param str reduce_op: Reduction operation to perform. Currently only "sum" is supported.
    :param str group_name: Name of the group to perform all-reduce on.
    :param Tensor out: Output tensor to store the result of the all-reduce operation. Can be a regular tensor.


.. py:function:: two_shot_all_reduce_(input: Tensor, reduce_op: str, group_name: str) -> Tensor

    Performs a two-shot all-reduce operation on the input tensor.

    :param Tensor input: Input tensor to perform all-reduce on. Must be symmetric.
    :param str reduce_op: Reduction operation to perform. Currently only "sum" is supported.
    :param str group_name: Name of the group to perform all-reduce on.


.. py:function:: nvshmem_broadcast(input: Tensor, root: int, group_name: str) -> Tensor

    Broadcasts the `input` tensor from the `root` rank to all ranks in the group
    using NVSHMEM, in place. This op is host/stream-initiated and works both
    intra-node and across nodes. On non-root ranks the contents of `input` are
    overwritten with the data from the root; on the root rank they are unchanged.
    The operation is issued on the current CUDA stream and returns `input`.

    :param Tensor input: Tensor to broadcast (on the root) or receive into (on other ranks). Must be symmetric.
    :param int root: Rank within the group that holds the source data. Must be smaller than the group size.
    :param str group_name: Name of the group to perform the broadcast on.


.. py:function:: nvshmem_put(tensor: Tensor, peer: int) -> None

    Performs a one-sided, host/stream-initiated put over NVSHMEM: copies the
    local `tensor` into the same symmetric allocation on `peer`. Works both
    intra-node and across nodes (e.g. over RDMA/RoCE). The op only issues the
    transfer on the current CUDA stream; it does not wait for remote completion,
    so you must provide your own synchronization (e.g. `nvshmem_put_with_signal`
    / `nvshmem_wait_for_signal`) before consuming the data on the peer.

    :param Tensor tensor: Symmetric, contiguous tensor whose data is sent, and which also names the destination allocation on the peer.
    :param int peer: Rank to send the data to. Must be smaller than the world size.


.. py:function:: nvshmem_get(tensor: Tensor, peer: int) -> None

    Performs a one-sided, host/stream-initiated get over NVSHMEM: copies the data
    from the same symmetric allocation on `peer` into the local `tensor`. Works
    both intra-node and across nodes. The transfer is issued on the current CUDA
    stream.

    :param Tensor tensor: Symmetric, contiguous tensor that receives the data, and which also names the source allocation on the peer.
    :param int peer: Rank to read the data from. Must be smaller than the world size.


.. py:function:: nvshmem_get_out(dst: Tensor, hdl: SymmetricMemory, offset: int, size: int, peer: int) -> None

    Low-level, host/stream-initiated get that copies `size` elements starting at
    element `offset` from the peer's symmetric allocation (the one backing `hdl`)
    into `dst`. This is the primitive backing the higher-level
    :func:`~torch.distributed._symmetric_memory.get` helper; most users should
    prefer that helper. The copy is issued on the current CUDA stream.

    :param Tensor dst: Local CUDA tensor to receive the data. Must be contiguous, on the same device as `hdl`, and hold at least `size` elements.
    :param SymmetricMemory hdl: Handle returned by `rendezvous`, identifying the peer's symmetric allocation to read from.
    :param int offset: Starting element (in `dst`'s dtype) within the peer allocation. Must be non-negative.
    :param int size: Number of elements to copy. Must be non-negative.
    :param int peer: Rank to read the data from. Must be a valid rank in the group.


.. py:function:: nvshmem_put_with_signal(tensor: Tensor, sigpad: Tensor, signal: int, peer: int) -> None

    Performs a one-sided put of `tensor` to the same symmetric allocation on
    `peer`, and atomically sets the peer's signal location `sigpad` to `signal`
    once the data transfer has completed. This lets the peer detect arrival of
    the data via `nvshmem_wait_for_signal`. Issued on the current CUDA stream.

    :param Tensor tensor: Symmetric tensor whose data is sent, and which also names the destination allocation on the peer.
    :param Tensor sigpad: Symmetric signal pad on the peer to set once the transfer completes.
    :param int signal: Value to set the peer's `sigpad` to.
    :param int peer: Rank to send the data to.


.. py:function:: nvshmem_wait_for_signal(sigpad: Tensor, signal: int, peer: int) -> None

    Blocks the current CUDA stream until the local signal location `sigpad`
    equals `signal`. Typically paired with `nvshmem_put_with_signal` on the
    sender side to wait for incoming data.

    :param Tensor sigpad: Local signal pad to poll.
    :param int signal: Value to wait for.
    :param int peer: Reserved for future use.


.. py:function:: nvshmem_all_to_all(input: Tensor, out: Tensor, group_name: str) -> Tensor

    Performs an equal-split all-to-all operation using NVSHMEM. Unlike the
    pointer-based collectives, this op is host/stream-initiated and runs over the
    NVSHMEM transport, so it works both intra-node and across nodes (e.g. over
    RDMA/RoCE) without requiring peer buffers to be directly addressable from the
    GPU.

    The input is divided into `group_size` equal-sized chunks; chunk `i` is sent
    to rank `i`, and the chunk received from rank `i` is placed at position `i` in
    the output.

    :param Tensor input: Input tensor to perform all-to-all on. Must be symmetric and contiguous. Its number of elements must be divisible by the group size.
    :param Tensor out: Output tensor to store the result of the all-to-all operation. Must be symmetric and contiguous, and have the same number of elements and dtype as `input`.
    :param str group_name: Name of the group to perform all-to-all on.


.. py:function:: all_to_all_vdev(input: Tensor, out: Tensor, in_splits: Tensor, out_splits_offsets: Tensor, group_name: str) -> None

    Performs an all-to-all-v operation using NVSHMEM, with split information provided on device.

    :param Tensor input: Input tensor to perform all-to-all on. Must be symmetric.
    :param Tensor out: Output tensor to store the result of the all-to-all operation. Must be symmetric.
    :param Tensor in_splits: Tensor containing splits of data to send to each peer. Must be symmetric. Must be of size (group_size,). The splits are in the unit of elements in the 1st dimension.
    :param Tensor out_splits_offsets: Tensor containing the splits and offsets of data received from each peer. Must be symmetric. Must be of size (2, group_size). The rows are (in order): output splits and output offsets.
    :param str group_name: Name of the group to perform all-to-all on.


.. py:function:: all_to_all_vdev_2d(input: Tensor, out: Tensor, in_splits: Tensor, out_splits_offsets: Tensor, group_name: str, [major_align: int = None]) -> None

    Perform a 2D all-to-all-v operation using NVSHMEM, with split information provided on device. In Mixture of Experts models, this operation can be used to dispatch tokens.

    :param Tensor input: Input tensor to perform all-to-all on. Must be symmetric.
    :param Tensor out: Output tensor to store the result of the all-to-all operation. Must be symmetric.
    :param Tensor in_splits: Tensor containing the splits of data to send to each expert. Must be symmetric. Must be of size (group_size * ne,), where ne is the number of experts per rank. The splits are in the unit of elements in the 1st dimension.
    :param Tensor out_splits_offsets: Tensor containing the splits and offsets of data received from each peer. Must be symmetric. Must be of size (2, group_size * ne). The rows are (in order): output splits and output offsets.
    :param str group_name: Name of the group to perform all-to-all on.
    :param int major_align: Optional alignment for the major dimension of the output chunk for each expert. If not provided, the alignment is assumed to be 1. Any alignment adjustment will be reflected in the output offsets.

    A 2D AllToAllv shuffle is illustrated below:
    (world_size = 2, ne = 2, total number of experts = 4)::

      Source: |       Rank 0      |       Rank 1      |
              | c0 | c1 | c2 | c3 | d0 | d1 | d2 | d3 |

      Dest  : |       Rank 0      |       Rank 1      |
              | c0 | d0 | c1 | d1 | c2 | d2 | c3 | d3 |

    where each `c_i` / `d_i` are slices of the `input` tensor, targeting expert
    `i`, with length indicated by input splits.  That is, the 2D AllToAllv
    shuffle achieves a transpose from rank-major order at input to expert-major
    order at output.

    If `major_align` is not 1, the output offsets of c1, c2, c3 will be
    up-aligned to this value. For example, if c0 has length 5 and d0 has
    length 7 (making a total of 12), and if the `major_align` is set to 16,
    the output offset of c1 will be 16. Similar for c2 and c3. This value has
    no effect on the offset of the minor dimension, i.e.  d0, d1, d2 and d3.
    Note: since cutlass does not support empty bins, we set the aligned length
    to `major_align` if it is 0. See
    https://github.com/pytorch/pytorch/issues/152668.


.. py:function:: all_to_all_vdev_2d_offset(Tensor input, Tensor out, Tensor in_splits_offsets, Tensor out_splits_offsets, str group_name) -> None

    Perform a 2D AllToAllv shuffle operation, with input split and offset
    information provided on device. The input offsets are not required to be
    exact prefix sum of the input splits, i.e. paddings are allowed between the
    split chunks. The paddings, however, will not be transferred to peer
    ranks.

    In Mixture of Experts models, this operation can be used to combine tokens
    processed by experts on parallel ranks. This operation can be viewed as an
    "reverse" operation to the `all_to_all_vdev_2d` operation (which shuffles
    tokens to experts).

    :param Tensor input: Input tensor to perform all-to-all on. Must be symmetric.
    :param Tensor out: Output tensor to store the result of the all-to-all operation. Must be symmetric.
    :param Tensor in_splits_offsets: Tensor containing the splits and offsets of data to send to each expert. Must be symmetric. Must be of size (2, group_size * ne), where `ne` is the number of experts. The rows are (in order): input splits and input offsets. The splits are in the unit of elements in the 1st dimension.
    :param Tensor out_splits_offsets: Tensor containing the splits and offsets of data received from each peer. Must be symmetric. Must be of size (2, group_size * ne). The rows are (in order): output splits and output offsets.
    :param str group_name: Name of the group to perform all-to-all on.


.. py:function:: tile_reduce(in_tile: Tensor, out_tile: Tensor, root: int, group_name: str, [reduce_op: str = 'sum']) -> None

    Reduces a 2D tile from all ranks to a specified root rank within a process group.

    :param Tensor in_tile: Input 2D tensor to be reduced. Must be symmetrically allocated.
    :param Tensor out_tile: Output 2D tensor to contain the result of the reduction. Must be symmetric and have the same shape, dtype, and device as `in_tile`.
    :param int root: The rank of the process in the specified group that will receive the reduced result.
    :param str group_name: The name of the symmetric memory process group to perform the reduction in.
    :param str reduce_op: The reduction operation to perform. Currently, only ``"sum"`` is supported. Defaults to ``"sum"``.

    This function reduces `in_tile` tensors from all members of the group, writing the result to `out_tile` at the root rank. All ranks must participate and provide the same `group_name` and tensor shapes.

    Example::

        >>> # doctest: +SKIP
        >>> # Reduce the bottom-right quadrant of a tensor
        >>> tile_size = full_size // 2
        >>> full_inp = symm_mem.empty(full_size, full_size)
        >>> full_out = symm_mem.empty(full_size, full_size)
        >>> s = slice(tile_size, 2 * tile_size)
        >>> in_tile = full_inp[s, s]
        >>> out_tile = full_out[s, s]
        >>> torch.ops.symm_mem.tile_reduce(in_tile, out_tile, root=0, group_name)


.. py:function:: multi_root_tile_reduce(in_tiles: list[Tensor], out_tile: Tensor, roots: list[int], group_name: str, [reduce_op: str = 'sum']) -> None

    Perform multiple tile reductions concurrently, with each tile reduced to a separate root.

    : param list[Tensor] in_tiles: A list of input tensors.
    : param Tensor out_tile: Output tensor to contain the reduced tile.
    : param list[int] roots: A list of root ranks each corresponding to an input tile in `in_tiles`, in the same order. A rank cannot be a root more than once.
    : param str group_name: Name of the group to use for the collective operation.
    : param str reduce_op: Reduction operation to perform. Currently only "sum" is supported.

    Example::

        >>> # doctest: +SKIP
        >>> # Reduce four quadrants of a tensor, each to a different root
        >>> tile_size = full_size // 2
        >>> full_inp = symm_mem.empty(full_size, full_size)
        >>> s0 = slice(0, tile_size)
        >>> s1 = slice(tile_size, 2 * tile_size)
        >>> in_tiles = [ full_inp[s0, s0], full_inp[s0, s1], full_inp[s1, s0], full_inp[s1, s1] ]
        >>> out_tile = symm_mem.empty(tile_size, tile_size)
        >>> roots = [0, 1, 2, 3]
        >>> torch.ops.symm_mem.multi_root_tile_reduce(in_tiles, out_tile, roots, group_name)

```
