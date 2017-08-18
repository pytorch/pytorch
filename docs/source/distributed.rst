.. role:: hidden
    :class: hidden-section

Distributed communication package - torch.distributed
=====================================================

.. automodule:: torch.distributed
.. currentmodule:: torch.distributed

Currently torch.distributed supports three backends, each with
different capabilities. The table below shows which functions are available
for use with CPU / CUDA tensors.
MPI supports cuda only iff the implementation used to build PyTorch supports it.

+------------+-----------+-----------+-----------+
| Backend    | ``tcp``   | ``gloo``  | ``mpi``   |
+------------+-----+-----+-----+-----+-----+-----+
| Device     | CPU | GPU | CPU | GPU | CPU | GPU |
+============+=====+=====+=====+=====+=====+=====+
| send       | ✓   | ✘   | ✘   | ✘   | ✓   | ?   |
+------------+-----+-----+-----+-----+-----+-----+
| recv       | ✓   | ✘   | ✘   | ✘   | ✓   | ?   |
+------------+-----+-----+-----+-----+-----+-----+
| broadcast  | ✓   | ✘   | ✓   | ✓   | ✓   | ?   |
+------------+-----+-----+-----+-----+-----+-----+
| all_reduce | ✓   | ✘   | ✓   | ✓   | ✓   | ?   |
+------------+-----+-----+-----+-----+-----+-----+
| reduce     | ✓   | ✘   | ✘   | ✘   | ✓   | ?   |
+------------+-----+-----+-----+-----+-----+-----+
| all_gather | ✓   | ✘   | ✘   | ✘   | ✓   | ?   |
+------------+-----+-----+-----+-----+-----+-----+
| gather     | ✓   | ✘   | ✘   | ✘   | ✓   | ?   |
+------------+-----+-----+-----+-----+-----+-----+
| scatter    | ✓   | ✘   | ✘   | ✘   | ✓   | ?   |
+------------+-----+-----+-----+-----+-----+-----+
| barrier    | ✓   | ✘   | ✓   | ✓   | ✓   | ?   |
+------------+-----+-----+-----+-----+-----+-----+

Initialization
--------------

The package needs to be initialized using the :func:`torch.distributed.init_process_group`
function before calling any other methods.

.. autofunction:: init_process_group

.. autofunction:: get_rank

.. autofunction:: get_world_size

--------------------------------------------------------------------------------

Currently three initialization methods are supported:

TCP initialization
^^^^^^^^^^^^^^^^^^

Initialization will utilize a network address reachable from all processes.
If the address belongs to one of the machines, initialization requires that all processes
have manually specified ranks.

Alternatively, the address has to be a valid IP multicast address, in which case,
ranks can be assigned automatically. Multicast initialization also supports
a ``group_name`` argument, which allows you to use the same address for multiple jobs,
as long as they use different group names.

::

    import torch.distributed as dist

    # Use address of one of the machines
    dist.init_process_group(init_method='tcp://10.1.1.20:23456', rank=args.rank, world_size=4)

    # or a multicast address - rank will be assigned automatically if unspecified
    dist.init_process_group(init_method='tcp://[ff15:1e18:5d4c:4cf0:d02d:b659:53ba:b0a7]:23456',
                            world_size=4)

Shared file-system initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Another initialization method makes use of a file system shared and visible from
all machines in a group. The URL should start with ``file://`` and contain a path
to a non-existent file (in an existing directory) on a shared file system.
This initialization method also supports a ``group_name`` argument, which allows you to
use the same shared file path for multiple jobs, as long as they use different
group names.

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

