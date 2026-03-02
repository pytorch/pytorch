```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# Distributed communication package - torch.distributed

:::{note}
Please refer to [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
for a brief introduction to all features related to distributed training.
:::

```{eval-rst}
.. automodule:: torch.distributed
```

```{eval-rst}
.. currentmodule:: torch.distributed
```

## Backends

`torch.distributed` supports four built-in backends, each with
different capabilities. The table below shows which functions are available
for use with a CPU or GPU for each backend. For NCCL, GPU refers to CUDA GPU
while for XCCL to XPU GPU.

MPI supports CUDA only if the implementation used to build PyTorch supports it.

```{eval-rst}
+----------------+-----------+-----------+-----------+-----------+
| Backend        | ``gloo``  | ``mpi``   | ``nccl``  | ``xccl``  |
+----------------+-----+-----+-----+-----+-----+-----+-----+-----+
| Device         | CPU | GPU | CPU | GPU | CPU | GPU | CPU | GPU |
+================+=====+=====+=====+=====+=====+=====+=====+=====+
| send           | ✓   | ✘   | ✓   | ?   | ✘   | ✓   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+-----+-----+
| recv           | ✓   | ✘   | ✓   | ?   | ✘   | ✓   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+-----+-----+
| broadcast      | ✓   | ✓   | ✓   | ?   | ✘   | ✓   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+-----+-----+
| all_reduce     | ✓   | ✓   | ✓   | ?   | ✘   | ✓   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+-----+-----+
| reduce         | ✓   | ✓   | ✓   | ?   | ✘   | ✓   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+-----+-----+
| all_gather     | ✓   | ✓   | ✓   | ?   | ✘   | ✓   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+-----+-----+
| gather         | ✓   | ✓   | ✓   | ?   | ✘   | ✓   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+-----+-----+
| scatter        | ✓   | ✓   | ✓   | ?   | ✘   | ✓   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+-----+-----+
| reduce_scatter | ✓   | ✓   | ✘   | ✘   | ✘   | ✓   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+-----+-----+
| all_to_all     | ✘   | ✘   | ✓   | ?   | ✘   | ✓   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+-----+-----+
| barrier        | ✓   | ✘   | ✓   | ?   | ✘   | ✓   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+-----+-----+
```

### Backends that come with PyTorch

PyTorch distributed package supports Linux (stable), macOS (stable), and Windows (prototype).
By default for Linux, the Gloo and NCCL backends are built and included in PyTorch
distributed (NCCL only when building with CUDA). MPI is an optional backend that can only be
included if you build PyTorch from source. (e.g. building PyTorch on a host that has MPI
installed.)

:::{note}
As of PyTorch v1.8, Windows supports all collective communications backends but NCCL,
If the `init_method` argument of {func}`init_process_group` points to a file it must adhere
to the following schema:

- Local file system, `init_method="file:///d:/tmp/some_file"`
- Shared file system, `init_method="file://////{machine_name}/{share_folder_name}/some_file"`

Same as on Linux platform, you can enable TcpStore by setting environment variables,
MASTER_ADDR and MASTER_PORT.
:::

### Which backend to use?

In the past, we were often asked: "which backend should I use?".

- Rule of thumb

  - Use the NCCL backend for distributed training with CUDA **GPU**.
  - Use the XCCL backend for distributed training with XPU **GPU**.
  - Use the Gloo backend for distributed training with **CPU**.

- GPU hosts with InfiniBand interconnect

  - Use NCCL, since it's the only backend that currently supports
    InfiniBand and GPUDirect.

- GPU hosts with Ethernet interconnect

  - Use NCCL, since it currently provides the best distributed GPU
    training performance, especially for multiprocess single-node or
    multi-node distributed training. If you encounter any problem with
    NCCL, use Gloo as the fallback option. (Note that Gloo currently
    runs slower than NCCL for GPUs.)

- CPU hosts with InfiniBand interconnect

  - If your InfiniBand has enabled IP over IB, use Gloo, otherwise,
    use MPI instead. We are planning on adding InfiniBand support for
    Gloo in the upcoming releases.

- CPU hosts with Ethernet interconnect

  - Use Gloo, unless you have specific reasons to use MPI.

### Common environment variables

#### Choosing the network interface to use

By default, both the NCCL and Gloo backends will try to find the right network interface to use.
If the automatically detected interface is not correct, you can override it using the following
environment variables (applicable to the respective backend):

- **NCCL_SOCKET_IFNAME**, for example `export NCCL_SOCKET_IFNAME=eth0`
- **GLOO_SOCKET_IFNAME**, for example `export GLOO_SOCKET_IFNAME=eth0`

If you're using the Gloo backend, you can specify multiple interfaces by separating
them by a comma, like this: `export GLOO_SOCKET_IFNAME=eth0,eth1,eth2,eth3`.
The backend will dispatch operations in a round-robin fashion across these interfaces.
It is imperative that all processes specify the same number of interfaces in this variable.

#### Other NCCL environment variables

**Debugging** - in case of NCCL failure, you can set `NCCL_DEBUG=INFO` to print an explicit
warning message as well as basic NCCL initialization information.

You may also use `NCCL_DEBUG_SUBSYS` to get more details about a specific
aspect of NCCL. For example, `NCCL_DEBUG_SUBSYS=COLL` would print logs of
collective calls, which may be helpful when debugging hangs, especially those
caused by collective type or message size mismatch. In case of topology
detection failure, it would be helpful to set `NCCL_DEBUG_SUBSYS=GRAPH`
to inspect the detailed detection result and save as reference if further help
from NCCL team is needed.

**Performance tuning** - NCCL performs automatic tuning based on its topology detection to save users'
tuning effort. On some socket-based systems, users may still try tuning
`NCCL_SOCKET_NTHREADS` and `NCCL_NSOCKS_PERTHREAD` to increase socket
network bandwidth. These two environment variables have been pre-tuned by NCCL
for some cloud providers, such as AWS or GCP.

For a full list of NCCL environment variables, please refer to
[NVIDIA NCCL's official documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

You can tune NCCL communicators even further using `torch.distributed.ProcessGroupNCCL.NCCLConfig`
and `torch.distributed.ProcessGroupNCCL.Options`. Learn more about them using `help`
(e.g. `help(torch.distributed.ProcessGroupNCCL.NCCLConfig)`) in the interpreter.

(distributed-basics)=

## Basics

The `torch.distributed` package provides PyTorch support and communication primitives
for multiprocess parallelism across several computation nodes running on one or more
machines. The class {func}`torch.nn.parallel.DistributedDataParallel` builds on this
functionality to provide synchronous distributed training as a wrapper around any
PyTorch model. This differs from the kinds of parallelism provided by
{doc}`multiprocessing` and {func}`torch.nn.DataParallel` in that it supports
multiple network-connected machines and in that the user must explicitly launch a separate
copy of the main training script for each process.

In the single-machine synchronous case, `torch.distributed` or the
{func}`torch.nn.parallel.DistributedDataParallel` wrapper may still have advantages over other
approaches to data-parallelism, including {func}`torch.nn.DataParallel`:

- Each process maintains its own optimizer and performs a complete optimization step with each
  iteration. While this may appear redundant, since the gradients have already been gathered
  together and averaged across processes and are thus the same for every process, this means
  that no parameter broadcast step is needed, reducing time spent transferring tensors between
  nodes.
- Each process contains an independent Python interpreter, eliminating the extra interpreter
  overhead and "GIL-thrashing" that comes from driving several execution threads, model
  replicas, or GPUs from a single Python process. This is especially important for models that
  make heavy use of the Python runtime, including models with recurrent layers or many small
  components.

## Initialization

The package needs to be initialized using the {func}`torch.distributed.init_process_group`
or {func}`torch.distributed.device_mesh.init_device_mesh` function before calling any other methods.
Both block until all processes have joined.

:::{warning}
Initialization is not thread-safe. Process group creation should be performed from a single thread, to prevent
inconsistent 'UUID' assignment across ranks, and to prevent races during initialization that can lead to hangs.
:::

```{eval-rst}
.. autofunction:: is_available
```

```{eval-rst}
.. autofunction:: init_process_group
```

```{eval-rst}
.. autofunction:: torch.distributed.device_mesh.init_device_mesh
```

```{eval-rst}
.. autofunction:: is_initialized
```

```{eval-rst}
.. autofunction:: is_mpi_available
```

```{eval-rst}
.. autofunction:: is_nccl_available
```

```{eval-rst}
.. autofunction:: is_gloo_available
```

```{eval-rst}
.. autofunction:: torch.distributed.distributed_c10d.is_xccl_available
.. autofunction:: torch.distributed.distributed_c10d.batch_isend_irecv
.. autofunction:: torch.distributed.distributed_c10d.destroy_process_group
.. autofunction:: torch.distributed.distributed_c10d.is_backend_available
.. autofunction:: torch.distributed.distributed_c10d.irecv
.. autofunction:: torch.distributed.distributed_c10d.is_gloo_available
.. autofunction:: torch.distributed.distributed_c10d.is_initialized
.. autofunction:: torch.distributed.distributed_c10d.is_mpi_available
.. autofunction:: torch.distributed.distributed_c10d.is_nccl_available
.. autofunction:: torch.distributed.distributed_c10d.is_torchelastic_launched
.. autofunction:: torch.distributed.distributed_c10d.is_ucc_available
```

```{eval-rst}
.. autofunction:: is_torchelastic_launched
```

```{eval-rst}
.. autofunction:: get_default_backend_for_device
```

______________________________________________________________________

Currently three initialization methods are supported:

### TCP initialization

There are two ways to initialize using TCP, both requiring a network address
reachable from all processes and a desired `world_size`. The first way
requires specifying an address that belongs to the rank 0 process. This
initialization method requires that all processes have manually specified ranks.

Note that multicast address is not supported anymore in the latest distributed
package. `group_name` is deprecated as well.

```
import torch.distributed as dist

# Use address of one of the machines
dist.init_process_group(backend, init_method='tcp://10.1.1.20:23456',
                        rank=args.rank, world_size=4)
```

### Shared file-system initialization

Another initialization method makes use of a file system that is shared and
visible from all machines in a group, along with a desired `world_size`. The URL should start
with `file://` and contain a path to a non-existent file (in an existing
directory) on a shared file system. File-system initialization will automatically
create that file if it doesn't exist, but will not delete the file. Therefore, it
is your responsibility to make sure that the file is cleaned up before the next
{func}`init_process_group` call on the same file path/name.

Note that automatic rank assignment is not supported anymore in the latest
distributed package and `group_name` is deprecated as well.

:::{warning}
This method assumes that the file system supports locking using `fcntl` - most
local systems and NFS support it.
:::

:::{warning}
This method will always create the file and try its best to clean up and remove
the file at the end of the program. In other words, each initialization with
the file init method will need a brand new empty file in order for the initialization
to succeed. If the same file used by the previous initialization (which happens not
to get cleaned up) is used again, this is unexpected behavior and can often cause
deadlocks and failures. Therefore, even though this method will try its best to clean up
the file, if the auto-delete happens to be unsuccessful, it is your responsibility
to ensure that the file is removed at the end of the training to prevent the same
file to be reused again during the next time. This is especially important
if you plan to call {func}`init_process_group` multiple times on the same file name.
In other words, if the file is not removed/cleaned up and you call
{func}`init_process_group` again on that file, failures are expected.
The rule of thumb here is that, make sure that the file is non-existent or
empty every time {func}`init_process_group` is called.
:::

```
import torch.distributed as dist

# rank should always be specified
dist.init_process_group(backend, init_method='file:///mnt/nfs/sharedfile',
                        world_size=4, rank=args.rank)
```

### Environment variable initialization

This method will read the configuration from environment variables, allowing
one to fully customize how the information is obtained. The variables to be set
are:

- `MASTER_PORT` - required; has to be a free port on machine with rank 0
- `MASTER_ADDR` - required (except for rank 0); address of rank 0 node
- `WORLD_SIZE` - required; can be set either here, or in a call to init function
- `RANK` - required; can be set either here, or in a call to init function

The machine with rank 0 will be used to set up all connections.

This is the default method, meaning that `init_method` does not have to be specified (or
can be `env://`).

### Improving initialization time

- `TORCH_GLOO_LAZY_INIT` - establishes connections on demand rather than
  using a full mesh which can greatly improve initialization time for non all2all
  operations.

## Post-Initialization

Once {func}`torch.distributed.init_process_group` was run, the following functions can be used. To
check whether the process group has already been initialized use {func}`torch.distributed.is_initialized`.

```{eval-rst}
.. autoclass:: Backend
    :members:
```

```{eval-rst}
.. autofunction:: get_backend
```

```{eval-rst}
.. autofunction:: get_rank
```

```{eval-rst}
.. autofunction:: get_world_size
```

## Shutdown

It is important to clean up resources on exit by calling {func}`destroy_process_group`.

The simplest pattern to follow is to destroy every process group and backend by calling
{func}`destroy_process_group()` with the default value of None for the `group` argument, at a
point in the training script where communications are no longer needed, usually near the
end of main(). The call should be made once per trainer-process, not at the outer
process-launcher level.

if {func}`destroy_process_group` is not called by all ranks in a pg within the timeout duration,
especially when there are multiple process-groups in the application e.g. for N-D parallelism,
hangs on exit are possible. This is because the destructor for ProcessGroupNCCL calls ncclCommAbort,
which must be called collectively, but the order of calling ProcessGroupNCCL's destructor if called
by python's GC is not deterministic. Calling {func}`destroy_process_group` helps by ensuring
ncclCommAbort is called in a consistent order across ranks, and avoids calling ncclCommAbort
during ProcessGroupNCCL's destructor.

### Reinitialization

`destroy_process_group` can also be used to destroy individual process groups. One use
case could be fault tolerant training, where a process group may be destroyed and then
a new one initialized during runtime. In this case, it's critical to synchronize the trainer
processes using some means other than torch.distributed primitives \_after\_ calling destroy and
before subsequently initializing. This behavior is currently unsupported/untested, due to
the difficulty of achieving this synchronization, and is considered a known issue. Please file
a github issue or RFC if this is a use case that's blocking you.

______________________________________________________________________

## Groups

By default collectives operate on the default group (also called the world) and
require all processes to enter the distributed function call. However, some workloads can benefit
from more fine-grained communication. This is where distributed groups come
into play. {func}`~torch.distributed.new_group` function can be
used to create new groups, with arbitrary subsets of all processes. It returns
an opaque group handle that can be given as a `group` argument to all collectives
(collectives are distributed functions to exchange information in certain well-known programming patterns).

```{eval-rst}
.. autofunction:: new_group
```

```{eval-rst}
.. autofunction:: torch.distributed.distributed_c10d.shrink_group
```

```{eval-rst}
.. autofunction:: get_group_rank
```

```{eval-rst}
.. autofunction:: get_global_rank
```

```{eval-rst}
.. autofunction:: get_process_group_ranks

```

## DeviceMesh

DeviceMesh is a higher level abstraction that manages process groups (or NCCL communicators).
It allows user to easily create inter node and intra node process groups without worrying about
how to set up the ranks correctly for different sub process groups, and it helps manage those
distributed process group easily. {func}`~torch.distributed.device_mesh.init_device_mesh` function can be
used to create new DeviceMesh, with a mesh shape describing the device topology.

```{eval-rst}
.. autoclass:: torch.distributed.device_mesh.DeviceMesh
    :members:
```

## Point-to-point communication

```{eval-rst}
.. autofunction:: send
```

```{eval-rst}
.. autofunction:: recv
```

{func}`~torch.distributed.isend` and {func}`~torch.distributed.irecv`
return distributed request objects when used. In general, the type of this object is unspecified
as they should never be created manually, but they are guaranteed to support two methods:

- `is_completed()` - returns True if the operation has finished
- `wait()` - will block the process until the operation is finished.
  `is_completed()` is guaranteed to return True once it returns.

```{eval-rst}
.. autofunction:: isend
```

```{eval-rst}
.. autofunction:: irecv
```

```{eval-rst}
.. autofunction:: send_object_list
```

```{eval-rst}
.. autofunction:: recv_object_list
```

```{eval-rst}
.. autofunction:: batch_isend_irecv
```

```{eval-rst}
.. autoclass:: P2POp
```

## Synchronous and asynchronous collective operations

Every collective operation function supports the following two kinds of operations,
depending on the setting of the `async_op` flag passed into the collective:

**Synchronous operation** - the default mode, when `async_op` is set to `False`.
When the function returns, it is guaranteed that
the collective operation is performed. In the case of CUDA operations, it is not guaranteed
that the CUDA operation is completed, since CUDA operations are asynchronous. For CPU collectives, any
further function calls utilizing the output of the collective call will behave as expected. For CUDA collectives,
function calls utilizing the output on the same CUDA stream will behave as expected. Users must take care of
synchronization under the scenario of running under different streams. For details on CUDA semantics such as stream
synchronization, see [CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html).
See the below script to see examples of differences in these semantics for CPU and CUDA operations.

**Asynchronous operation** - when `async_op` is set to True. The collective operation function
returns a distributed request object. In general, you don't need to create it manually and it
is guaranteed to support two methods:

- `is_completed()` - in the case of CPU collectives, returns `True` if completed. In the case of CUDA operations,
  returns `True` if the operation has been successfully enqueued onto a CUDA stream and the output can be utilized on the
  default stream without further synchronization.
- `wait()` - in the case of CPU collectives, will block the process until the operation is completed. In the case
  of CUDA collectives, will block the currently active CUDA stream until the operation is completed (but will not block the CPU).
- `get_future()` - returns `torch._C.Future` object. Supported for NCCL, also supported for most operations on GLOO
  and MPI, except for peer to peer operations.
  Note: as we continue adopting Futures and merging APIs, `get_future()` call might become redundant.

**Example**

The following code can serve as a reference regarding semantics for CUDA operations when using distributed collectives.
It shows the explicit need to synchronize when using collective outputs on different CUDA streams:

```
# Code runs on each rank.
dist.init_process_group("nccl", rank=rank, world_size=2)
output = torch.tensor([rank]).cuda(rank)
s = torch.cuda.Stream()
handle = dist.all_reduce(output, async_op=True)
# Wait ensures the operation is enqueued, but not necessarily complete.
handle.wait()
# Using result on non-default stream.
with torch.cuda.stream(s):
    s.wait_stream(torch.cuda.default_stream())
    output.add_(100)
if rank == 0:
    # if the explicit call to wait_stream was omitted, the output below will be
    # non-deterministically 1 or 101, depending on whether the allreduce overwrote
    # the value after the add completed.
    print(output)
```

## Collective functions

```{eval-rst}
.. autofunction:: broadcast
```

```{eval-rst}
.. autofunction:: broadcast_object_list
```

```{eval-rst}
.. autofunction:: all_reduce
```

```{eval-rst}
.. autofunction:: reduce
```

```{eval-rst}
.. autofunction:: all_gather
```

```{eval-rst}
.. autofunction:: all_gather_into_tensor
```

```{eval-rst}
.. autofunction:: all_gather_object
```

```{eval-rst}
.. autofunction:: gather
```

```{eval-rst}
.. autofunction:: gather_object
```

```{eval-rst}
.. autofunction:: scatter
```

```{eval-rst}
.. autofunction:: scatter_object_list
```

```{eval-rst}
.. autofunction:: reduce_scatter
```

```{eval-rst}
.. autofunction:: reduce_scatter_tensor
```

```{eval-rst}
.. autofunction:: all_to_all_single
```

```{eval-rst}
.. autofunction:: all_to_all
```

```{eval-rst}
.. autofunction:: barrier
```

```{eval-rst}
.. autofunction:: monitored_barrier
```

```{eval-rst}
.. autoclass:: Work
    :members:
```

```{eval-rst}
.. autoclass:: ReduceOp
```

```{eval-rst}
.. class:: reduce_op

    Deprecated enum-like class for reduction operations: ``SUM``, ``PRODUCT``,
    ``MIN``, and ``MAX``.

    :class:`~torch.distributed.ReduceOp` is recommended to use instead.

```

## Distributed Key-Value Store

The distributed package comes with a distributed key-value store, which can be
used to share information between processes in the group as well as to
initialize the distributed package in
{func}`torch.distributed.init_process_group` (by explicitly creating the store
as an alternative to specifying `init_method`.) There are 3 choices for
Key-Value Stores: {class}`~torch.distributed.TCPStore`,
{class}`~torch.distributed.FileStore`, and {class}`~torch.distributed.HashStore`.

```{eval-rst}
.. autoclass:: Store
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: TCPStore
    :members:
    :special-members: __init__
```

```{eval-rst}
.. autoclass:: HashStore
    :members:
    :special-members: __init__
```

```{eval-rst}
.. autoclass:: FileStore
    :members:
    :special-members: __init__
```

```{eval-rst}
.. autoclass:: PrefixStore
    :members:
    :special-members: __init__

```

## Profiling Collective Communication

Note that you can use `torch.profiler` (recommended, only available after 1.8.1) or `torch.autograd.profiler` to profile collective communication and point-to-point communication APIs mentioned here. All out-of-the-box backends (`gloo`,
`nccl`, `mpi`) are supported and collective communication usage will be rendered as expected in profiling output/traces. Profiling your code is the same as any regular torch operator:

```
import torch
import torch.distributed as dist
with torch.profiler():
    tensor = torch.randn(20, 10)
    dist.all_reduce(tensor)
```

Please refer to the [profiler documentation](https://pytorch.org/docs/main/profiler.html) for a full overview of profiler features.

## Optimization with Symmetric Memory

### Copy Engine Collectives

When NCCL collective operations are performed on symmetric memory tensors with
the zero-CTA policy, data movement is offloaded to the GPU's copy engines (DMA
engines) instead of using CUDA streaming multiprocessors (SMs). This frees up
SMs for compute work, enabling better overlap of communication and computation.

For setup instructions, requirements, and examples, see
[Copy Engine Collectives](copy-engine-collectives) in the Symmetric Memory documentation.

### Higher-Precision Reduction

When NCCL collectives such as ``reduce_scatter`` and ``all_reduce`` operate on
symmetric memory tensors, NCCL's symmetric kernel implementation automatically
performs internal reduction with higher precision (e.g., BF16/FP16 in → FP32
accumulate → BF16/FP16 out). This improves numerical accuracy without any code
changes to the collective call.

For details on scope, supported domains, and version requirements, see
[Higher-Precision Reduction](higher-precision-reduction) in the Symmetric Memory documentation.

## Multi-GPU collective functions

:::{warning}
The multi-GPU functions (which stand for multiple GPUs per CPU thread) are
deprecated. As of today, PyTorch Distributed's preferred programming model
is one device per thread, as exemplified by the APIs in this document. If
you are a backend developer and want to support multiple devices per thread,
please contact PyTorch Distributed's maintainers.
:::

(object_collectives)=

## Object collectives

:::{warning}
Object collectives have a number of serious limitations. Read further to determine
if they are safe to use for your use case.
:::

Object collectives are a set of collective-like operations that work on arbitrary
Python objects, as long as they can be pickled. There are various collective patterns
implemented (e.g. broadcast, all_gather, ...) but they each roughly follow this pattern:

1. convert the input object into a pickle (raw bytes), then shove it into a byte tensor
2. communicate the size of this byte tensor to peers (first collective operation)
3. allocate appropriately sized tensor to perform the real collective
4. communicate the object data (second collective operation)
5. convert raw data back into Python (unpickle)

Object collectives sometimes have surprising performance or memory characteristics that lead to
long runtimes or OOMs, and thus they should be used with caution. Here are some common issues.

**Asymmetric pickle/unpickle time** - Pickling objects can be slow, depending on the number, type and size of the objects.
When the collective has a fan-in (e.g. gather_object), the receiving rank(s) must unpickle N times more objects than
the sending rank(s) had to pickle, which can cause other ranks to time out on their next collective.

**Inefficient tensor communication** - Tensors should be sent via regular collective APIs, not object collective APIs.
It is possible to send Tensors via object collective APIs, but they will be serialized and deserialized (including a
CPU-sync and device-to-host copy in the case of non-CPU tensors), and in almost every case other than debugging or
troubleshooting code, it would be worth the trouble to refactor the code to use non-object collectives instead.

**Unexpected tensor devices** - If you still want to send tensors via object collectives, there is another aspect
specific to cuda (and possibly other accelerators) tensors. If you pickle a tensor that is currently on `cuda:3`, and
then unpickle it, you will get another tensor on `cuda:3` *regardless of which process you are on, or which CUDA device
is the 'default' device for that process*. With regular tensor collective APIs, 'output tensors' will always be on the
same, local device, which is generally what you'd expect.

Unpickling a tensor will implicitly activate a CUDA context if it is the first
time a GPU is used by the process, which can waste significant amounts of GPU memory. This issue can be avoided by
moving tensors to CPU before passing them as inputs to an object collective.

## Third-party backends

Besides the builtin GLOO/MPI/NCCL backends, PyTorch distributed supports
third-party backends through a run-time register mechanism.
For references on how to develop a third-party backend through C++ Extension,
please refer to [Tutorials - Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html) and
`test/cpp_extensions/cpp_c10d_extension.cpp`. The capability of third-party
backends are decided by their own implementations.

The new backend derives from `c10d::ProcessGroup` and registers the backend
name and the instantiating interface through {func}`torch.distributed.Backend.register_backend`
when imported.

When manually importing this backend and invoking {func}`torch.distributed.init_process_group`
with the corresponding backend name, the `torch.distributed` package runs on
the new backend.

:::{warning}
The support of third-party backend is experimental and subject to change.
:::

(distributed-launch)=

## Launch utility

The `torch.distributed` package also provides a launch utility in
`torch.distributed.launch`. This helper utility can be used to launch
multiple processes per node for distributed training.

```{eval-rst}
.. automodule:: torch.distributed.launch
```

```{eval-rst}
.. currentmodule:: torch.distributed.launch
```

```{eval-rst}
.. autofunction:: launch
```

```{eval-rst}
.. autofunction:: main
```

```{eval-rst}
.. autofunction:: parse_args
```

## Spawn utility

The {ref}`multiprocessing-doc` package also provides a `spawn`
function in {func}`torch.multiprocessing.spawn`. This helper function
can be used to spawn multiple processes. It works by passing in the
function that you want to run and spawns N processes to run it. This
can be used for multiprocess distributed training as well.

For references on how to use it, please refer to [PyTorch example - ImageNet
implementation](https://github.com/pytorch/examples/tree/master/imagenet)

Note that this function requires Python 3.4 or higher.

## Debugging `torch.distributed` applications

Debugging distributed applications can be challenging due to hard to understand hangs, crashes, or inconsistent behavior across ranks. `torch.distributed` provides
a suite of tools to help debug training applications in a self-serve fashion:

### Python Breakpoint

It is extremely convenient to use python's debugger in a distributed environment, but because it does not work out of the box many people do not use it at all.
PyTorch offers a customized wrapper around pdb that streamlines the process.

`torch.distributed.breakpoint` makes this process easy. Internally, it customizes `pdb`'s breakpoint behavior in two ways but otherwise behaves as normal `pdb`.
1. Attaches the debugger only on one rank (specified by the user).
2. Ensures all other ranks stop, by using a `torch.distributed.barrier()` that will release once the debugged rank issues a `continue`
3. Reroutes stdin from the child process such that it connects to your terminal.

To use it, simply issue `torch.distributed.breakpoint(rank)` on all ranks, using the same value for `rank` in each case.

### Monitored Barrier

As of v1.10, {func}`torch.distributed.monitored_barrier` exists as an alternative to {func}`torch.distributed.barrier` which fails with helpful information about which rank may be faulty
when crashing, i.e. not all ranks calling into {func}`torch.distributed.monitored_barrier` within the provided timeout. {func}`torch.distributed.monitored_barrier` implements a host-side
barrier using `send`/`recv` communication primitives in a process similar to acknowledgements, allowing rank 0 to report which rank(s) failed to acknowledge
the barrier in time. As an example, consider the following function where rank 1 fails to call into {func}`torch.distributed.monitored_barrier` (in practice this could be due
to an application bug or hang in a previous collective):

```
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank):
    dist.init_process_group("nccl", rank=rank, world_size=2)
    # monitored barrier requires gloo process group to perform host-side sync.
    group_gloo = dist.new_group(backend="gloo")
    if rank not in [1]:
        dist.monitored_barrier(group=group_gloo, timeout=timedelta(seconds=2))


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    mp.spawn(worker, nprocs=2, args=())
```

The following error message is produced on rank 0, allowing the user to determine which rank(s) may be faulty and investigate further:

```
RuntimeError: Rank 1 failed to pass monitoredBarrier in 2000 ms
 Original exception:
[gloo/transport/tcp/pair.cc:598] Connection closed by peer [2401:db00:eef0:1100:3560:0:1c05:25d]:8594
```

### `TORCH_DISTRIBUTED_DEBUG`

With `TORCH_CPP_LOG_LEVEL=INFO`, the environment variable `TORCH_DISTRIBUTED_DEBUG` can be used to trigger additional useful logging and collective synchronization checks to ensure all ranks
are synchronized appropriately. `TORCH_DISTRIBUTED_DEBUG` can be set to either `OFF` (default), `INFO`, or `DETAIL` depending on the debugging level
required. Please note that the most verbose option, `DETAIL` may impact the application performance and thus should only be used when debugging issues.

Setting `TORCH_DISTRIBUTED_DEBUG=INFO` will result in additional debug logging when models trained with {func}`torch.nn.parallel.DistributedDataParallel` are initialized, and
`TORCH_DISTRIBUTED_DEBUG=DETAIL` will additionally log runtime performance statistics a select number of iterations. These runtime statistics
include data such as forward time, backward time, gradient communication time, etc. As an example, given the following application:

```
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class TwoLinLayerNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(10, 10, bias=False)
        self.b = torch.nn.Linear(10, 1, bias=False)

    def forward(self, x):
        a = self.a(x)
        b = self.b(x)
        return (a, b)


def worker(rank):
    dist.init_process_group("nccl", rank=rank, world_size=2)
    torch.cuda.set_device(rank)
    print("init model")
    model = TwoLinLayerNet().cuda()
    print("init ddp")
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    inp = torch.randn(10, 10).cuda()
    print("train")

    for _ in range(20):
        output = ddp_model(inp)
        loss = output[0] + output[1]
        loss.sum().backward()


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"  # set to DETAIL for runtime logging.
    mp.spawn(worker, nprocs=2, args=())
```

The following logs are rendered at initialization time:

```
I0607 16:10:35.739390 515217 logger.cpp:173] [Rank 0]: DDP Initialized with:
broadcast_buffers: 1
bucket_cap_bytes: 26214400
find_unused_parameters: 0
gradient_as_bucket_view: 0
is_multi_device_module: 0
iteration: 0
num_parameter_tensors: 2
output_device: 0
rank: 0
total_parameter_size_bytes: 440
world_size: 2
backend_name: nccl
bucket_sizes: 440
cuda_visible_devices: N/A
device_ids: 0
dtypes: float
master_addr: localhost
master_port: 29501
module_name: TwoLinLayerNet
nccl_async_error_handling: N/A
nccl_blocking_wait: N/A
nccl_debug: WARN
nccl_ib_timeout: N/A
nccl_nthreads: N/A
nccl_socket_ifname: N/A
torch_distributed_debug: INFO
```

The following logs are rendered during runtime (when `TORCH_DISTRIBUTED_DEBUG=DETAIL` is set):

```
I0607 16:18:58.085681 544067 logger.cpp:344] [Rank 1 / 2] Training TwoLinLayerNet unused_parameter_size=0
 Avg forward compute time: 40838608
 Avg backward compute time: 5983335
Avg backward comm. time: 4326421
 Avg backward comm/comp overlap time: 4207652
I0607 16:18:58.085693 544066 logger.cpp:344] [Rank 0 / 2] Training TwoLinLayerNet unused_parameter_size=0
 Avg forward compute time: 42850427
 Avg backward compute time: 3885553
Avg backward comm. time: 2357981
 Avg backward comm/comp overlap time: 2234674
```

In addition, `TORCH_DISTRIBUTED_DEBUG=INFO` enhances crash logging in {func}`torch.nn.parallel.DistributedDataParallel` due to unused parameters in the model. Currently, `find_unused_parameters=True`
must be passed into {func}`torch.nn.parallel.DistributedDataParallel` initialization if there are parameters that may be unused in the forward pass, and as of v1.10, all model outputs are required
to be used in loss computation as {func}`torch.nn.parallel.DistributedDataParallel` does not support unused parameters in the backwards pass. These constraints are challenging especially for larger
models, thus when crashing with an error, {func}`torch.nn.parallel.DistributedDataParallel` will log the fully qualified name of all parameters that went unused. For example, in the above application,
if we modify `loss` to be instead computed as `loss = output[1]`, then `TwoLinLayerNet.a` does not receive a gradient in the backwards pass, and
thus results in `DDP` failing. On a crash, the user is passed information about parameters which went unused, which may be challenging to manually find for large models:

```
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by passing
 the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by
making sure all `forward` function outputs participate in calculating loss.
If you already have done the above, then the distributed data parallel module wasn't able to locate the output tensors in the return value of your module's `forward` function. Please include the loss function and the structure of the return va
lue of `forward` of your module when reporting this issue (e.g. list, dict, iterable).
Parameters which did not receive grad for rank 0: a.weight
Parameter indices which did not receive grad for rank 0: 0
```

Setting `TORCH_DISTRIBUTED_DEBUG=DETAIL` will trigger additional consistency and synchronization checks on every collective call issued by the user
either directly or indirectly (such as DDP `allreduce`). This is done by creating a wrapper process group that wraps all process groups returned by
{func}`torch.distributed.init_process_group` and {func}`torch.distributed.new_group` APIs. As a result, these APIs will return a wrapper process group that can be used exactly like a regular process
group, but performs consistency checks before dispatching the collective to an underlying process group. Currently, these checks include a {func}`torch.distributed.monitored_barrier`,
which ensures all ranks complete their outstanding collective calls and reports ranks which are stuck. Next, the collective itself is checked for consistency by
ensuring all collective functions match and are called with consistent tensor shapes. If this is not the case, a detailed error report is included when the
application crashes, rather than a hang or uninformative error message. As an example, consider the following function which has mismatched input shapes into
{func}`torch.distributed.all_reduce`:

```
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank):
    dist.init_process_group("nccl", rank=rank, world_size=2)
    torch.cuda.set_device(rank)
    tensor = torch.randn(10 if rank == 0 else 20).cuda()
    dist.all_reduce(tensor)
    torch.cuda.synchronize(device=rank)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    mp.spawn(worker, nprocs=2, args=())
```

With the `NCCL` backend, such an application would likely result in a hang which can be challenging to root-cause in nontrivial scenarios. If the user enables
`TORCH_DISTRIBUTED_DEBUG=DETAIL` and reruns the application, the following error message reveals the root cause:

```
work = default_pg.allreduce([tensor], opts)
RuntimeError: Error when verifying shape tensors for collective ALLREDUCE on rank 0. This likely indicates that input shapes into the collective are mismatched across ranks. Got shapes:  10
[ torch.LongTensor{1} ]
```

:::{note}
For fine-grained control of the debug level during runtime the functions {func}`torch.distributed.set_debug_level`, {func}`torch.distributed.set_debug_level_from_env`, and
{func}`torch.distributed.get_debug_level` can also be used.
:::

In addition, `TORCH_DISTRIBUTED_DEBUG=DETAIL` can be used in conjunction with `TORCH_SHOW_CPP_STACKTRACES=1` to log the entire callstack when a collective desynchronization is detected. These
collective desynchronization checks will work for all applications that use `c10d` collective calls backed by process groups created with the
{func}`torch.distributed.init_process_group` and {func}`torch.distributed.new_group` APIs.


### torch.distributed.debug HTTP Server

The `torch.distributed.debug` module provides a HTTP server that can be used to debug distributed applications. The server can
be started by calling {func}`torch.distributed.debug.start_debug_server`. This
allows users to collect data across all workers at runtime.

```{eval-rst}
.. automodule:: torch.distributed.debug
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
    :member-order: bysource

```


## Logging

In addition to explicit debugging support via {func}`torch.distributed.monitored_barrier` and `TORCH_DISTRIBUTED_DEBUG`, the underlying C++ library of `torch.distributed` also outputs log
messages at various levels. These messages can be helpful to understand the execution state of a distributed training job and to troubleshoot problems such as network connection failures. The
following matrix shows how the log level can be adjusted via the combination of `TORCH_CPP_LOG_LEVEL` and `TORCH_DISTRIBUTED_DEBUG` environment variables.

| `TORCH_CPP_LOG_LEVEL` | `TORCH_DISTRIBUTED_DEBUG` | Effective Log Level |
| --------------------- | ------------------------- | ------------------- |
| `ERROR`               | ignored                   | Error               |
| `WARNING`             | ignored                   | Warning             |
| `INFO`                | ignored                   | Info                |
| `INFO`                | `INFO`                    | Debug               |
| `INFO`                | `DETAIL`                  | Trace (a.k.a. All)  |

Distributed components raise custom Exception types derived from `RuntimeError`:

- `torch.distributed.DistError`: This is the base type of all distributed exceptions.
- `torch.distributed.DistBackendError`: This exception is thrown when a backend-specific error occurs. For example, if
  the `NCCL` backend is used and the user attempts to use a GPU that is not available to the `NCCL` library.
- `torch.distributed.DistNetworkError`: This exception is thrown when networking
  libraries encounter errors (ex: Connection reset by peer)
- `torch.distributed.DistStoreError`: This exception is thrown when the Store encounters
  an error (ex: TCPStore timeout)

```{eval-rst}
.. autoclass:: torch.distributed.DistError
```

```{eval-rst}
.. autoclass:: torch.distributed.DistBackendError
```

```{eval-rst}
.. autoclass:: torch.distributed.DistNetworkError
```

```{eval-rst}
.. autoclass:: torch.distributed.DistStoreError
```

If you are running single node training, it may be convenient to interactively breakpoint your script. We offer a way to conveniently breakpoint a single rank:

```{eval-rst}
.. autofunction:: torch.distributed.breakpoint
```

% Distributed modules that are missing specific entries.

% Adding them here for tracking purposes until they are more permanently fixed.

```{eval-rst}
.. py:module:: torch.distributed.algorithms
```

```{eval-rst}
.. py:module:: torch.distributed.algorithms.ddp_comm_hooks
```

```{eval-rst}
.. py:module:: torch.distributed.algorithms.model_averaging
```

```{eval-rst}
.. py:module:: torch.distributed.elastic
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.utils
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.utils.data
```

```{eval-rst}
.. py:module:: torch.distributed.launcher
```

```{eval-rst}
.. py:module:: torch.distributed.nn
```

```{eval-rst}
.. py:module:: torch.distributed.nn.api
```

```{eval-rst}
.. py:module:: torch.distributed.nn.jit
```

```{eval-rst}
.. py:module:: torch.distributed.nn.jit.templates
```

```{eval-rst}
.. py:module:: torch.distributed.algorithms.ddp_comm_hooks.ddp_zero_hook
```

```{eval-rst}
.. py:module:: torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks
```

```{eval-rst}
.. py:module:: torch.distributed.algorithms.ddp_comm_hooks.default_hooks
```

```{eval-rst}
.. py:module:: torch.distributed.algorithms.ddp_comm_hooks.mixed_precision_hooks
```

```{eval-rst}
.. py:module:: torch.distributed.algorithms.ddp_comm_hooks.optimizer_overlap_hooks
```

```{eval-rst}
.. py:module:: torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook
```

```{eval-rst}
.. py:module:: torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook
```

```{eval-rst}
.. py:module:: torch.distributed.algorithms.ddp_comm_hooks.quantization_hooks
```

```{eval-rst}
.. py:module:: torch.distributed.algorithms.join
```

```{eval-rst}
.. py:module:: torch.distributed.algorithms.model_averaging.averagers
```

```{eval-rst}
.. py:module:: torch.distributed.algorithms.model_averaging.hierarchical_model_averager
```

```{eval-rst}
.. py:module:: torch.distributed.algorithms.model_averaging.utils
```

```{eval-rst}
.. py:module:: torch.distributed.argparse_util
```

```{eval-rst}
.. py:module:: torch.distributed.c10d_logger
```

```{eval-rst}
.. py:module:: torch.distributed.checkpoint.api
```

```{eval-rst}
.. py:module:: torch.distributed.checkpoint.default_planner
```

```{eval-rst}
.. py:module:: torch.distributed.checkpoint.filesystem
```

```{eval-rst}
.. py:module:: torch.distributed.checkpoint.hf_storage
```

```{eval-rst}
.. py:module:: torch.distributed.checkpoint.quantized_hf_storage
```

```{eval-rst}
.. py:module:: torch.distributed.checkpoint.metadata
```

```{eval-rst}
.. py:module:: torch.distributed.checkpoint.optimizer
```

```{eval-rst}
.. py:module:: torch.distributed.checkpoint.planner
```

```{eval-rst}
.. py:module:: torch.distributed.checkpoint.planner_helpers
```

```{eval-rst}
.. py:module:: torch.distributed.checkpoint.resharding
```

```{eval-rst}
.. py:module:: torch.distributed.checkpoint.state_dict_loader
```

```{eval-rst}
.. py:module:: torch.distributed.checkpoint.state_dict_saver
```

```{eval-rst}
.. py:module:: torch.distributed.checkpoint.stateful
```

```{eval-rst}
.. py:module:: torch.distributed.checkpoint.storage
```

```{eval-rst}
.. py:module:: torch.distributed.checkpoint.utils
```

```{eval-rst}
.. py:module:: torch.distributed.collective_utils
```

```{eval-rst}
.. py:module:: torch.distributed.constants
```

```{eval-rst}
.. py:module:: torch.distributed.device_mesh
```

```{eval-rst}
.. py:module:: torch.distributed.distributed_c10d
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.agent.server.api
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.agent.server.local_elastic_agent
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.events.api
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.events.handlers
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.metrics.api
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.multiprocessing.api
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.multiprocessing.errors.error_handler
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.multiprocessing.errors.handlers
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.multiprocessing.redirects
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.multiprocessing.tail_log
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.rendezvous.api
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.rendezvous.c10d_rendezvous_backend
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.rendezvous.dynamic_rendezvous
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.rendezvous.etcd_rendezvous
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.rendezvous.etcd_rendezvous_backend
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.rendezvous.etcd_server
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.rendezvous.etcd_store
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.rendezvous.static_tcp_rendezvous
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.rendezvous.utils
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.timer.api
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.timer.file_based_local_timer
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.timer.local_timer
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.utils.api
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.utils.data.cycling_iterator
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.utils.data.elastic_distributed_sampler
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.utils.distributed
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.utils.log_level
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.utils.logging
```

```{eval-rst}
.. py:module:: torch.distributed.elastic.utils.store
```

```{eval-rst}
.. py:module:: torch.distributed.fsdp.api
```

```{eval-rst}
.. py:module:: torch.distributed.fsdp.fully_sharded_data_parallel
```

```{eval-rst}
.. py:module:: torch.distributed.fsdp.sharded_grad_scaler
```

```{eval-rst}
.. py:module:: torch.distributed.fsdp.wrap
```

```{eval-rst}
.. py:module:: torch.distributed.launcher.api

.. autofunction:: torch.distributed.launcher.api.launch_agent
```

```{eval-rst}
.. py:module:: torch.distributed.logging_handlers
```

```{eval-rst}
.. py:module:: torch.distributed.nn.api.remote_module
```

```{eval-rst}
.. py:module:: torch.distributed.nn.functional
```

```{eval-rst}
.. py:module:: torch.distributed.nn.jit.instantiator
```

```{eval-rst}
.. py:module:: torch.distributed.nn.jit.templates.remote_module_template
```

```{eval-rst}
.. py:module:: torch.distributed.optim.apply_optimizer_in_backward
```

```{eval-rst}
.. py:module:: torch.distributed.optim.functional_adadelta
```

```{eval-rst}
.. py:module:: torch.distributed.optim.functional_adagrad
```

```{eval-rst}
.. py:module:: torch.distributed.optim.functional_adam
```

```{eval-rst}
.. py:module:: torch.distributed.optim.functional_adamax
```

```{eval-rst}
.. py:module:: torch.distributed.optim.functional_adamw
```

```{eval-rst}
.. py:module:: torch.distributed.optim.functional_rmsprop
```

```{eval-rst}
.. py:module:: torch.distributed.optim.functional_rprop
```

```{eval-rst}
.. py:module:: torch.distributed.optim.functional_sgd
```

```{eval-rst}
.. py:module:: torch.distributed.optim.named_optimizer
```

```{eval-rst}
.. py:module:: torch.distributed.optim.optimizer
```

```{eval-rst}
.. py:module:: torch.distributed.optim.post_localSGD_optimizer
```

```{eval-rst}
.. py:module:: torch.distributed.optim.utils
```

```{eval-rst}
.. py:module:: torch.distributed.optim.zero_redundancy_optimizer
```

```{eval-rst}
.. py:module:: torch.distributed.remote_device
```

```{eval-rst}
.. py:module:: torch.distributed.rendezvous
```

```{eval-rst}
.. py:module:: torch.distributed.rpc.api
```

```{eval-rst}
.. py:module:: torch.distributed.rpc.backend_registry
```

```{eval-rst}
.. py:module:: torch.distributed.rpc.constants
```

```{eval-rst}
.. py:module:: torch.distributed.rpc.functions
```

```{eval-rst}
.. py:module:: torch.distributed.rpc.internal
```

```{eval-rst}
.. py:module:: torch.distributed.rpc.options
```

```{eval-rst}
.. py:module:: torch.distributed.rpc.rref_proxy
```

```{eval-rst}
.. py:module:: torch.distributed.rpc.server_process_global_profiler
```

```{eval-rst}
.. py:module:: torch.distributed.tensor.parallel.api
```

```{eval-rst}
.. py:module:: torch.distributed.tensor.parallel.ddp
```

```{eval-rst}
.. py:module:: torch.distributed.tensor.parallel.fsdp
```

```{eval-rst}
.. py:module:: torch.distributed.tensor.parallel.input_reshard
```

```{eval-rst}
.. py:module:: torch.distributed.tensor.parallel.loss
```

```{eval-rst}
.. py:module:: torch.distributed.tensor.parallel.style
```

```{eval-rst}
.. py:module:: torch.distributed.utils
```

```{eval-rst}
.. py:module:: torch.distributed.checkpoint.state_dict
```

```{toctree}
:hidden:

distributed._dist2
```
