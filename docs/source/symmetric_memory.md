```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# PyTorch Symmetric Memory

:::{note}
`torch.distributed._symmetric_memory` is currently in alpha state and under
development. API changes may be possible.
:::

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

t = symm_mem.empty(128, device="cuda")
hdl = symm_mem.rendezvous(t, group)
```

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
.. autofunction:: is_nvshmem_available
```

```{eval-rst}
.. autofunction:: set_backend
```

```{eval-rst}
.. autofunction:: get_backend
```
