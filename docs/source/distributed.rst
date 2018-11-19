.. role:: hidden
    :class: hidden-section

Distributed communication package - torch.distributed
=====================================================

.. automodule:: torch.distributed
.. currentmodule:: torch.distributed

Currently torch.distributed supports three backends, each with
different capabilities. The table below shows which functions are available
for use with CPU / CUDA tensors.
MPI supports cuda only if the implementation used to build PyTorch supports it.


+------------+-----------+-----------+-----------+
| Backend    | ``gloo``  | ``mpi``   | ``nccl``  |
+------------+-----+-----+-----+-----+-----+-----+
| Device     | CPU | GPU | CPU | GPU | CPU | GPU |
+============+=====+=====+=====+=====+=====+=====+
| send       | ✓   | ✘   | ✓   | ?   | ✘   | ✘   |
+------------+-----+-----+-----+-----+-----+-----+
| recv       | ✓   | ✘   | ✓   | ?   | ✘   | ✘   |
+------------+-----+-----+-----+-----+-----+-----+
| broadcast  | ✓   | ✓   | ✓   | ?   | ✘   | ✓   |
+------------+-----+-----+-----+-----+-----+-----+
| all_reduce | ✓   | ✓   | ✓   | ?   | ✘   | ✓   |
+------------+-----+-----+-----+-----+-----+-----+
| reduce     | ✘   | ✘   | ✓   | ?   | ✘   | ✓   |
+------------+-----+-----+-----+-----+-----+-----+
| all_gather | ✘   | ✘   | ✓   | ?   | ✘   | ✓   |
+------------+-----+-----+-----+-----+-----+-----+
| gather     | ✘   | ✘   | ✓   | ?   | ✘   | ✘   |
+------------+-----+-----+-----+-----+-----+-----+
| scatter    | ✘   | ✘   | ✓   | ?   | ✘   | ✘   |
+------------+-----+-----+-----+-----+-----+-----+
| barrier    | ✘   | ✘   | ✓   | ?   | ✘   | ✘   |
+------------+-----+-----+-----+-----+-----+-----+

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

.. autoclass:: Backend

.. autofunction:: get_backend

.. autofunction:: get_rank

.. autofunction:: get_world_size

.. autofunction:: is_initialized

.. autofunction:: get_default_group

.. autofunction:: is_mpi_available

.. autofunction:: is_nccl_available

--------------------------------------------------------------------------------

Currently three initialization methods are supported:

TCP initialization
^^^^^^^^^^^^^^^^^^

There are two ways to initialize using TCP, both requiring a network address
reachable from all processes and a desired ``world_size``. The first way
requires specifying an address that belongs to the rank 0 process. This
initialization method requires that all processes have manually specified ranks.

Note that multicast address is not supported anymore in the latest distributed
package. ``group_name`` is deprecated as well.

::

    import torch.distributed as dist

    # Use address of one of the machines
    dist.init_process_group(backend, init_method='tcp://10.1.1.20:23456',
                            rank=args.rank, world_size=4)

Shared file-system initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Another initialization method makes use of a file system that is shared and
visible from all machines in a group, along with a desired ``world_size``. The URL should start
with ``file://`` and contain a path to a non-existent file (in an existing
directory) on a shared file system. File-system initialization will automatically
create that file if it doesn't exist, but will not delete the file. Therefore, it
is your responsibility to make sure that the file is cleaned up before the next
:func:`init_process_group` call on the same file path/name.

Note that automatic rank assignment is not supported anymore in the latest
distributed package and ``group_name`` is deprecated as well.

.. warning::
    This method assumes that the file system supports locking using ``fcntl`` - most
    local systems and NFS support it.

.. warning::
    This method does not clean up and remove the file and it is your responsibility
    to remove the file at the end of the training. This is especially important
    if you plan to call :func:`init_process_group` multiple times on the same file name.
    In other words, if the file is not removed/cleaned up and you call
    :func:`init_process_group` again on that file, it is unexpected behavior and will cause
    failures. The rule of thumb here is that, make sure that the file is non-existent or
    empty everytime :func:`init_process_group` is called.

::

    import torch.distributed as dist

    # rank should always be specified
    dist.init_process_group(backend, init_method='file:///mnt/nfs/sharedfile',
                            world_size=4, rank=args.rank)

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

.. autofunction:: isend

.. autofunction:: irecv

Synchronous and asynchornous collective operations
--------------------------------------------------
Every collective operation function supports the following two kinds of operations:

synchronous operation - the default mode, when ``async_op`` is set to False.
when the function returns, it is guaranteed that
the collective operation is performed (not necessarily completed if it's a CUDA op since all
CUDA ops are asynchornous), and any further function calls depending on the data of the
collective operation can be called. In the synchronous mode, the collective function does not
return anything

asynchornous operation - when ``async_op`` is set to True. The collective operation function
returns a distributed request object. In general, you don't need to create it manually and it
is guaranteed to support two methods:

* ``is_completed()`` - returns True if the operation has finished
* ``wait()`` - will block the process until the operation is finished.


Collective functions
--------------------

.. autofunction:: broadcast

.. autofunction:: all_reduce

.. autofunction:: reduce

.. autofunction:: all_gather

.. autofunction:: gather

.. autofunction:: scatter

.. autofunction:: barrier

.. autoclass:: ReduceOp

.. class:: reduce_op

    Deprecated enum-like class for reduction operations: ``SUM``, ``PRODUCT``,
    ``MIN``, and ``MAX``.

    :class:`~torch.distributed.ReduceOp` is recommended to use instead.


Multi-GPU collective functions
------------------------------

If you have more than one GPU on each node, when using the NCCL and Gloo backend,
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
