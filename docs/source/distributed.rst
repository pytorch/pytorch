.. role:: hidden
    :class: hidden-section

Distributed communication package - torch.distributed
=====================================================

.. automodule:: torch.distributed
.. currentmodule:: torch.distributed

Backends
--------

``torch.distributed`` supports three built-in backends, each with
different capabilities. The table below shows which functions are available
for use with CPU / CUDA tensors.
MPI supports CUDA only if the implementation used to build PyTorch supports it.


+----------------+-----------+-----------+-----------+
| Backend        | ``gloo``  | ``mpi``   | ``nccl``  |
+----------------+-----+-----+-----+-----+-----+-----+
| Device         | CPU | GPU | CPU | GPU | CPU | GPU |
+================+=====+=====+=====+=====+=====+=====+
| send           | ✓   | ✘   | ✓   | ?   | ✘   | ✘   |
+----------------+-----+-----+-----+-----+-----+-----+
| recv           | ✓   | ✘   | ✓   | ?   | ✘   | ✘   |
+----------------+-----+-----+-----+-----+-----+-----+
| broadcast      | ✓   | ✓   | ✓   | ?   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+
| all_reduce     | ✓   | ✓   | ✓   | ?   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+
| reduce         | ✓   | ✘   | ✓   | ?   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+
| all_gather     | ✓   | ✘   | ✓   | ?   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+
| gather         | ✓   | ✘   | ✓   | ?   | ✘   | ✘   |
+----------------+-----+-----+-----+-----+-----+-----+
| scatter        | ✓   | ✘   | ✓   | ?   | ✘   | ✘   |
+----------------+-----+-----+-----+-----+-----+-----+
| reduce_scatter | ✘   | ✘   | ✘   | ✘   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+
| all_to_all     | ✘   | ✘   | ✓   | ?   | ✘   | ✘   |
+----------------+-----+-----+-----+-----+-----+-----+
| barrier        | ✓   | ✘   | ✓   | ?   | ✘   | ✓   |
+----------------+-----+-----+-----+-----+-----+-----+


Backends that come with PyTorch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch distributed currently only supports Linux. By default, the Gloo and NCCL backends
are built and included in PyTorch distributed (NCCL only when building with CUDA).
MPI is an
optional backend that can only be included if you build PyTorch from source. (e.g.
building PyTorch on a host that has MPI installed.)


Which backend to use?
^^^^^^^^^^^^^^^^^^^^^

In the past, we were often asked: "which backend should I use?".

- Rule of thumb

  - Use the NCCL backend for distributed **GPU** training
  - Use the Gloo backend for distributed **CPU** training.

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

Common environment variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Choosing the network interface to use
"""""""""""""""""""""""""""""""""""""

By default, both the NCCL and Gloo backends will try to find the right network interface to use.
If the automatically detected interface is not correct, you can override it using the following
environment variables (applicable to the respective backend):

* **NCCL_SOCKET_IFNAME**, for example ``export NCCL_SOCKET_IFNAME=eth0``
* **GLOO_SOCKET_IFNAME**, for example ``export GLOO_SOCKET_IFNAME=eth0``

If you're using the Gloo backend, you can specify multiple interfaces by separating
them by a comma, like this: ``export GLOO_SOCKET_IFNAME=eth0,eth1,eth2,eth3``.
The backend will dispatch operations in a round-robin fashion across these interfaces.
It is imperative that all processes specify the same number of interfaces in this variable.

Other NCCL environment variables
""""""""""""""""""""""""""""""""

NCCL has also provided a number of environment variables for fine-tuning purposes.

Commonly used ones include the following for debugging purposes:

- ``export NCCL_DEBUG=INFO``
- ``export NCCL_DEBUG_SUBSYS=ALL``

For the full list of NCCL environment variables, please refer to
`NVIDIA NCCL's official documentation <https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html>`_


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
    This method will always create the file and try its best to clean up and remove
    the file at the end of the program. In other words, each initialization with
    the file init method will need a brand new empty file in order for the initialization
    to succeed. If the same file used by the previous initialization (which happens not
    to get cleaned up) is used again, this is unexpected behavior and can often cause
    deadlocks and failures. Therefore, even though this method will try its best to clean up
    the file, if the auto-delete happens to be unsuccessful, it is your responsibility
    to ensure that the file is removed at the end of the training to prevent the same
    file to be reused again during the next time. This is especially important
    if you plan to call :func:`init_process_group` multiple times on the same file name.
    In other words, if the file is not removed/cleaned up and you call
    :func:`init_process_group` again on that file, failures are expected.
    The rule of thumb here is that, make sure that the file is non-existent or
    empty every time :func:`init_process_group` is called.

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

Synchronous and asynchronous collective operations
--------------------------------------------------
Every collective operation function supports the following two kinds of operations:

synchronous operation - the default mode, when ``async_op`` is set to False.
when the function returns, it is guaranteed that
the collective operation is performed (not necessarily completed if it's a CUDA op since all
CUDA ops are asynchronous), and any further function calls depending on the data of the
collective operation can be called. In the synchronous mode, the collective function does not
return anything

asynchronous operation - when ``async_op`` is set to True. The collective operation function
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

.. autofunction:: reduce_scatter

.. autofunction:: all_to_all

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
:func:`~torch.distributed.reduce_multigpu`
:func:`~torch.distributed.all_gather_multigpu` and
:func:`~torch.distributed.reduce_scatter_multigpu` support distributed collective
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

.. autofunction:: reduce_scatter_multigpu


.. _distributed-launch:

Third-party backends
--------------------

Besides the GLOO/MPI/NCCL backends, PyTorch distributed supports third-party backends
through a run-time register mechanism.
For references on how to develop a third-party backend through C++ Extension,
please refer to `Tutorials - Custom C++ and CUDA Extensions <https://pytorch.org/
tutorials/advanced/cpp_extension.html>`_ and `test/cpp_extensions/cpp_c10d_extension.cpp`.
The capability of third-party backends are decided by their own implementations.

The new backend derives from `c10d.ProcessGroup` and registers the backend name and the
instantiating interface through :func:`torch.distributed.Backend.register_backend` when
imported.

When manually importing this backend and invoking :func:`torch.distributed.init_process_group`
with the corresponding backend name, the `torch.distributed` package runs on the new backend.

.. warning::
    The support of third-party backend is experimental and subject to change.

Launch utility
--------------

The `torch.distributed` package also provides a launch utility in
`torch.distributed.launch`. This helper utility can be used to launch
multiple processes per node for distributed training. This utility also supports
both python2 and python3.


.. automodule:: torch.distributed.launch


Spawn utility
-------------

The :ref:`multiprocessing-doc` package also provides a ``spawn``
function in :func:`torch.multiprocessing.spawn`. This helper function
can be used to spawn multiple processes. It works by passing in the
function that you want to run and spawns N processes to run it. This
can be used for multiprocess distributed training as well.

For references on how to use it, please refer to `PyTorch example - ImageNet
implementation <https://github.com/pytorch/examples/tree/master/imagenet>`_

Note that this function requires Python 3.4 or higher.
