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
        x = torch.arange(128, device=device)

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

```{eval-rst}
.. autofunction:: get_mem_pool
```

## Op Reference
:::{note}
The following ops are hosted in the `torch.ops.symm_mem` namespace. You can call
them directly via `torch.ops.symm_mem.<op_name>`.
:::

```{eval-rst}
.. currentmodule:: torch.ops.symm_mem
```

```{eval-rst}
.. py:function:: multimem_all_reduce_(input: Tensor, reduce_op: str, group_name: str) -> Tensor

    Performs a multimem all-reduce operation on the input tensor. This operation
    requires hardware support for multimem operations. On NVIDIA GPUs, NVLink
    SHARP is required.

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
