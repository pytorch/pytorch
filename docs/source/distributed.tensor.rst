.. role:: hidden
    :class: hidden-section

PyTorch DTensor (Distributed Tensor)
======================================================

.. note::
  ``torch.distributed.tensor`` is currently in alpha state and under
  development, we are committing backward compatibility for the most APIs listed
  in the doc, but there might be API changes if necessary.


PyTorch DTensor is a tensor sharding primitive

.. automodule:: torch.distributed.tensor

.. currentmodule:: torch.distributed.tensor

.. autoclass:: DTensor
    :members:

.. autofunction::  distribute_tensor

.. autofunction::  distribute_module

DTensor supports the following placement types:

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
