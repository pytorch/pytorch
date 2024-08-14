.. role:: hidden
    :class: hidden-section

PyTorch DTensor (Distributed Tensor)
======================================================

.. note::
  ``torch.distributed.tensor`` is currently in alpha state and under
  development, we are committing backward compatibility for the most APIs listed
  in the doc, but there might be API changes if necessary.


PyTorch DTensor offers simple and flexible tensor sharding primitives that transparently handles distributed
logic, including sharded storage, operator computation and collective communications across devices/hosts.

.. automodule:: torch.distributed.tensor

.. currentmodule:: torch.distributed.tensor

:class:`DTensor` follows the SPMD (single program, multiple data) programming model to empower users to
write distributed program as if it's a single-device program with the same convergence property. It
provides a uniform tensor sharding layout (DTensor Layout) through specifying the :class:`DeviceMesh`
and :class:`Placement`:

- :class:`DeviceMesh` represents the device topology and the communicators of the cluster using
  an n-dimensional array.

- :class:`Placement` describes the sharding layout of the logical tensor on the :class:`DeviceMesh`.
  DTensor supports three types of placements: :class:`Shard`, :class:`Replicate` and :class:`Partial`.

There're three ways to construct a :class:`DTensor`:
  * :meth:`distribute_tensor` creates a :class:`DTensor` from a logical or "global" ``torch.Tensor`` on
    each rank. This could be used to shard the leaf ``torch.Tensor`` s (i.e. model parameters/buffers
    and inputs).
  * :meth:`DTensor.from_local` creates a :class:`DTensor` from a local ``torch.Tensor`` on each rank, which can
    be used to create :class:`DTensor` from a non-leaf ``torch.Tensor`` s (i.e. intermediate activation
    tensors during forward/backward).
  * DTensor provides dedicated tensor factory methods (e.g. :meth:`empty`, :meth:`ones`, :meth:`randn`, etc.)
    to allow different :class:`DTensor` creations by directly specifying the :class:`DeviceMesh` and
    :class:`Placement`

.. autoclass:: DTensor
    :members:
    :member-order: bysource

.. autofunction::  distribute_tensor


Along with :meth:`distribute_tensor`, DTensor also offers a :meth:`distribute_module` API to allow easier
sharding on the :class:`nn.Module` level

.. autofunction::  distribute_module

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

DTensor provides dedicated tensor factory functions to allow creating :class:`DTensor` directly
using torch.Tensor like factory function APIs (i.e. torch.ones, torch.empty, etc), by additionally
specifying the :class:`DeviceMesh` and :class:`Placement` for the :class:`DTensor` created:

.. autofunction:: zeros

.. autofunction:: ones

.. autofunction:: empty

.. autofunction:: full

.. autofunction:: rand

.. autofunction:: randn
