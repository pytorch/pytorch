.. role:: hidden
    :class: hidden-section

Distributed communication package - torch.distributed
=====================================================

.. automodule:: torch.distributed
.. currentmodule:: torch.distributed

Currently torch.distributed supports four backends, each with
different capabilities. The table below shows which functions are available
for use with CPU / CUDA tensors.
MPI supports cuda only if the implementation used to build PyTorch supports it.


+------------+-----------+-----------+-----------+-----------+
| Backend    | ``tcp``   | ``gloo``  | ``mpi``   | ``nccl``  |
+------------+-----+-----+-----+-----+-----+-----+-----+-----+
| Device     | CPU | GPU | CPU | GPU | CPU | GPU | CPU | GPU |
+============+=====+=====+=====+=====+=====+=====+=====+=====+
| send       | ✓   | ✘   | ✘   | ✘   | ✓   | ?   | ✘   | ✘   |
+------------+-----+-----+-----+-----+-----+-----+-----+-----+
| recv       | ✓   | ✘   | ✘   | ✘   | ✓   | ?   | ✘   | ✘   |
+------------+-----+-----+-----+-----+-----+-----+-----+-----+
| broadcast  | ✓   | ✘   | ✓   | ✓   | ✓   | ?   | ✘   | ✓   |
+------------+-----+-----+-----+-----+-----+-----+-----+-----+
| all_reduce | ✓   | ✘   | ✓   | ✓   | ✓   | ?   | ✘   | ✓   |
+------------+-----+-----+-----+-----+-----+-----+-----+-----+
| reduce     | ✓   | ✘   | ✘   | ✘   | ✓   | ?   | ✘   | ✓   |
+------------+-----+-----+-----+-----+-----+-----+-----+-----+
| all_gather | ✓   | ✘   | ✘   | ✘   | ✓   | ?   | ✘   | ✓   |
+------------+-----+-----+-----+-----+-----+-----+-----+-----+
| gather     | ✓   | ✘   | ✘   | ✘   | ✓   | ?   | ✘   | ✓   |
+------------+-----+-----+-----+-----+-----+-----+-----+-----+
| scatter    | ✓   | ✘   | ✘   | ✘   | ✓   | ?   | ✘   | ✓   |
+------------+-----+-----+-----+-----+-----+-----+-----+-----+
| barrier    | ✓   | ✘   | ✓   | ✓   | ✓   | ?   | ✘   | ✘   |
+------------+-----+-----+-----+-----+-----+-----+-----+-----+

.. _distributed-basics:

Basics
------

The `torch.distributed` package provides PyTorch support and communication primitives
for multiprocess parallelism across several computation nodes running on one or more
machines. The class :func:`torch.nn.parallel.DistributedDataParallel` builds on this
functionality to provide synchronous distributed training as a wrapper around any
PyTorch model. This differs from the kinds of parallelism provided by
:doc:`multiprocessing` and :func:`torch.nn.DataParallel` in that it supports
multiple network-connected machines and in that the user must explicitly launch a separate
copy of the main training script for each process.

In the single-machine synchronous case, `torch.distributed` or the
:func:`torch.nn.parallel.DistributedDataParallel` wrapper may still have advantages over other
approaches to data-parallelism, including :func:`torch.nn.DataParallel`:

* Each process maintains its own optimizer and performs a complete optimization step with each
  iteration. While this may appear redundant, since the gradients have already been gathered
  together and averaged across processes and are thus the same for every process, this means
  that no parameter broadcast step is needed, reducing time spent transferring tensors between
  nodes.
* Each process contains an independent Python interpreter, eliminating the extra interpreter
  overhead and "GIL-thrashing" that comes from driving several execution threads, model
  replicas, or GPUs from a single Python process. This is especially important for models that
  make heavy use of the Python runtime, including models with recurrent layers or many small
  components.

Initialization
--------------

The package needs to be initialized using the :func:`torch.distributed.init_process_group`
function before calling any other methods. This blocks until all processes have
joined.

.. autofunction:: init_process_group

.. autofunction:: get_rank

.. autofunction:: get_world_size

--------------------------------------------------------------------------------

Currently three initialization methods are supported:

TCP initialization
^^^^^^^^^^^^^^^^^^

There are two ways to initialize using TCP, both requiring a network address
reachable from all processes and a desired ``world_size``. The first way
requires specifying an address that belongs to the rank 0 process. This first way of
initialization requires that all processes have manually specified ranks.

Alternatively, the address has to be a valid IP multicast address, in which case
ranks can be assigned automatically. Multicast initialization also supports
a ``group_name`` argument, which allows you to use the same address for multiple
jobs, as long as they use different group names.

::

    import torch.distributed as dist

    # Use address of one of the machines
    dist.init_process_group(init_method='tcp://10.1.1.20:23456', rank=args.rank, world_size=4)

    # or a multicast address - rank will be assigned automatically if unspecified
    dist.init_process_group(init_method='tcp://[ff15:1e18:5d4c:4cf0:d02d:b659:53ba:b0a7]:23456',
                            world_size=4)

Shared file-system initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Another initialization method makes use of a file system that is shared and
visible from all machines in a group, along with a desired ``world_size``. The URL should start
with ``file://`` and contain a path to a non-existent file (in an existing
directory) on a shared file system. This initialization method also supports a
``group_name`` argument, which allows you to use the same shared file path for
multiple jobs, as long as they use different group names.

.. warning::
    This method assumes that the file system supports locking using ``fcntl`` - most
    local systems and NFS support it.

::

    import torch.distributed as dist

    # Rank will be assigned automatically if unspecified
    dist.init_process_group(init_method='file:///mnt/nfs/sharedfile', world_size=4,
                            group_name=args.group)

Environment variable initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method will read the configuration from environment variables, allowing
one to fully customize how the information is obtained. The variables to be set
are:

* ``MASTER_PORT`` - required; has to be a free port on machine with rank 0
* ``MASTER_ADDR`` - required (except for rank 0); address of rank 0 node
* ``WORLD_SIZE`` - required; can be set either here, or in a call to init function
* ``RANK`` - required; can be set either here, or in a call to init function

The machine with rank 0 will be used to set up all connections.

This is the default method, meaning that ``init_method`` does not have to be specified (or
can be ``env://``).

Groups
------

By default collectives operate on the default group (also called the world) and
require all processes to enter the distributed function call. However, some workloads can benefit
from more fine-grained communication. This is where distributed groups come
into play. :func:`~torch.distributed.new_group` function can be
used to create new groups, with arbitrary subsets of all processes. It returns
an opaque group handle that can be given as a ``group`` argument to all collectives
(collectives are distributed functions to exchange information in certain well-known programming patterns).

.. autofunction:: new_group

Point-to-point communication
----------------------------

.. autofunction:: send

.. autofunction:: recv

:func:`~torch.distributed.isend` and :func:`~torch.distributed.irecv`
return distributed request objects when used. In general, the type of this object is unspecified
as they should never be created manually, but they are guaranteed to support two methods:

* ``is_completed()`` - returns True if the operation has finished
* ``wait()`` - will block the process until the operation is finished.
  ``is_completed()`` is guaranteed to return True once it returns.

When using the MPI backend, :func:`~torch.distributed.isend` and :func:`~torch.distributed.irecv`
support non-overtaking, which has some guarantees on supporting message order. For more detail, see
http://mpi-forum.org/docs/mpi-2.2/mpi22-report/node54.htm#Node54

.. autofunction:: isend

.. autofunction:: irecv

Collective functions
--------------------

.. autofunction:: broadcast

.. autofunction:: all_reduce

.. autofunction:: reduce

.. autofunction:: all_gather

.. autofunction:: gather

.. autofunction:: scatter

.. autofunction:: barrier

Multi-GPU collective functions
------------------------------

If you have more than one GPU on each node, when using the NCCL backend,
:func:`~torch.distributed.broadcast_multigpu`
:func:`~torch.distributed.all_reduce_multigpu`
:func:`~torch.distributed.reduce_multigpu` and
:func:`~torch.distributed.all_gather_multigpu` support distributed collective
operations among multiple GPUs within each node. These functions can potentially
improve the overall distributed training performance and be easily used by
passing a list of tensors. Each Tensor in the passed tensor list needs
to be on a separate GPU device of the host where the function is called. Note
that the length of the tensor list needs to be identical among all the
distributed processes. Also note that currently the multi-GPU collective
functions are only supported by the NCCL backend.

For example, if the system we use for distributed training has 2 nodes, each
of which has 8 GPUs. On each of the 16 GPUs, there is a tensor that we would
like to all-reduce. The following code can serve as a reference:

Code running on Node 0

::

    import torch
    import torch.distributed as dist

    dist.init_process_group(backend="nccl",
                            init_method="file:///distributed_test",
                            world_size=2,
                            rank=0)
    tensor_list = []
    for dev_idx in range(torch.cuda.device_count()):
        tensor_list.append(torch.FloatTensor([1]).cuda(dev_idx))

    dist.all_reduce_multigpu(tensor_list)

Code running on Node 1

::

    import torch
    import torch.distributed as dist

    dist.init_process_group(backend="nccl",
                            init_method="file:///distributed_test",
                            world_size=2,
                            rank=1)
    tensor_list = []
    for dev_idx in range(torch.cuda.device_count()):
        tensor_list.append(torch.FloatTensor([1]).cuda(dev_idx))

    dist.all_reduce_multigpu(tensor_list)

After the call, all 16 tensors on the two nodes will have the all-reduced value
of 16

.. autofunction:: broadcast_multigpu

.. autofunction:: all_reduce_multigpu

.. autofunction:: reduce_multigpu

.. autofunction:: all_gather_multigpu


Launch utility
--------------

The `torch.distributed` package also provides a launch utility in
`torch.distributed.launch`.

.. automodule:: torch.distributed.launch
