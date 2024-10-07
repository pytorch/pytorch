.. currentmodule:: torch.distributed.tensor

torch.distributed.tensor
===========================

.. note::
  ``torch.distributed.tensor`` is currently in alpha state and under
  development, we are committing backward compatibility for the most APIs listed
  in the doc, but there might be API changes if necessary.


PyTorch DTensor (Distributed Tensor)
---------------------------------------

PyTorch DTensor offers simple and flexible tensor sharding primitives that transparently handles distributed
logic, including sharded storage, operator computation and collective communications across devices/hosts.
``DTensor`` could be used to build different paralleism solutions and support sharded state_dict representation
when working with multi-dimensional sharding.

Please see examples from the PyTorch native parallelism solutions that are built on top of ``DTensor``:

* `Tensor Parallel <https://pytorch.org/docs/main/distributed.tensor.parallel.html>`__
* `FSDP2 <https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md>`__

.. automodule:: torch.distributed.tensor

:class:`DTensor` follows the SPMD (single program, multiple data) programming model to empower users to
write distributed program as if it's a **single-device program with the same convergence property**. It
provides a uniform tensor sharding layout (DTensor Layout) through specifying the :class:`DeviceMesh`
and :class:`Placement`:

- :class:`DeviceMesh` represents the device topology and the communicators of the cluster using
  an n-dimensional array.

- :class:`Placement` describes the sharding layout of the logical tensor on the :class:`DeviceMesh`.
  DTensor supports three types of placements: :class:`Shard`, :class:`Replicate` and :class:`Partial`.


DTensor Class APIs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: torch.distributed.tensor

:class:`DTensor` is a ``torch.Tensor`` subclass. This means once a :class:`DTensor` is created, it could be
used in very similar way to ``torch.Tensor``, including running different types of PyTorch operators as if
running them in a single device, allowing proper distributed computation for PyTorch operators.

In addition to existing ``torch.Tensor`` methods, it also offers a set of additional methods to interact with
``torch.Tensor``, ``redistribute`` the DTensor Layout to a new DTensor, get the full tensor content
on all devices, etc.

.. autoclass:: DTensor
    :members:
    :member-order: bysource


DeviceMesh as the distributed communicator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: torch.distributed.device_mesh

:class:`DeviceMesh` was built from DTensor as the abstraction to describe cluster's device topology and represent
multi-dimensional communicators (on top of ``ProcessGroup``). To see the details of how to create/use a DeviceMesh,
please refer to the `DeviceMesh recipe <https://pytorch.org/tutorials/recipes/distributed_device_mesh.html>`__.


DTensor Placement Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.distributed.tensor.placement_types
.. currentmodule:: torch.distributed.tensor.placement_types

DTensor supports the following types of :class:`Placement` on each :class:`DeviceMesh` dimension:

.. autoclass:: Shard
  :members:
  :undoc-members:

.. autoclass:: Replicate
  :members:
  :undoc-members:

.. autoclass:: Partial
  :members:
  :undoc-members:

.. autoclass:: Placement
  :members:
  :undoc-members:


Different ways to create a DTensor
---------------------------------------

.. currentmodule:: torch.distributed.tensor

There're three ways to construct a :class:`DTensor`:
  * :meth:`distribute_tensor` creates a :class:`DTensor` from a logical or "global" ``torch.Tensor`` on
    each rank. This could be used to shard the leaf ``torch.Tensor`` s (i.e. model parameters/buffers
    and inputs).
  * :meth:`DTensor.from_local` creates a :class:`DTensor` from a local ``torch.Tensor`` on each rank, which can
    be used to create :class:`DTensor` from a non-leaf ``torch.Tensor`` s (i.e. intermediate activation
    tensors during forward/backward).
  * DTensor provides dedicated tensor factory functions (e.g. :meth:`empty`, :meth:`ones`, :meth:`randn`, etc.)
    to allow different :class:`DTensor` creations by directly specifying the :class:`DeviceMesh` and
    :class:`Placement`. Compare to :meth:`distribute_tensor`, this could directly materializing the sharded memory
    on device, instead of performing sharding after initializing the logical Tensor memory.

Create DTensor from a logical torch.Tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The SPMD (single program, multiple data) programming model in ``torch.distributed`` launches multiple processes
(i.e. via ``torchrun``) to execute the same program, this means that the model inside the program would be
initialized on different processes first (i.e. the model might be initialized on CPU, or meta device, or directly
on GPU if enough memory).

``DTensor`` offers a :meth:`distribute_tensor` API that could shard the model weights or Tensors to ``DTensor`` s,
where it would create a DTensor from the "logical" Tensor on each process. This would empower the created
``DTensor`` s to comply with the single device semantic, which is critical for **numerical correctness**.

.. autofunction::  distribute_tensor

Along with :meth:`distribute_tensor`, DTensor also offers a :meth:`distribute_module` API to allow easier
sharding on the :class:`nn.Module` level

.. autofunction::  distribute_module


DTensor Factory Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DTensor also provides dedicated tensor factory functions to allow creating :class:`DTensor` directly
using torch.Tensor like factory function APIs (i.e. torch.ones, torch.empty, etc), by additionally
specifying the :class:`DeviceMesh` and :class:`Placement` for the :class:`DTensor` created:

.. autofunction:: zeros

.. autofunction:: ones

.. autofunction:: empty

.. autofunction:: full

.. autofunction:: rand

.. autofunction:: randn


Debugging
---------------------------------------

.. automodule:: torch.distributed.tensor.debug
.. currentmodule:: torch.distributed.tensor.debug

Logging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When launching the program, you can turn on additional logging using the `TORCH_LOGS` environment variable from
`torch._logging <https://pytorch.org/docs/main/logging.html#module-torch._logging>`__ :

* `TORCH_LOGS=+dtensor` will display `logging.DEBUG` messages and all levels above it.
* `TORCH_LOGS=dtensor` will display `logging.INFO` messages and above.
* `TORCH_LOGS=-dtensor` will display `logging.WARNING` messages and above.

Debugging Tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To debug the program that applied DTensor, and understand more details about what collectives happened under the
hood, DTensor provides a :class:`CommDebugMode`:

.. autoclass:: CommDebugMode
    :members:
    :undoc-members:

To visualize the sharding of a DTensor that have less than 3 dimensions, DTensor provides :meth:`visualize_sharding`:

.. autofunction:: visualize_sharding


Experimental Features
---------------------------------------

``DTensor`` also provides a set of experimental features. These features are either in prototyping stage, or the basic
functionality is done and but looking for user feedbacks. Please submit a issue to PyTorch if you have feedbacks to
these features.

.. automodule:: torch.distributed.tensor.experimental
.. currentmodule:: torch.distributed.tensor.experimental

.. autofunction:: context_parallel
.. autofunction:: local_map
.. autofunction:: register_sharding


.. modules that are missing docs, add the doc later when necessary
.. py:module:: torch.distributed.tensor.device_mesh
