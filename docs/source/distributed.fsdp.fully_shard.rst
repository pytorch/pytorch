
torch.distributed.fsdp.fully_shard
==================================

PyTorch FSDP2 (``fully_shard``)
-------------------------------

PyTorch FSDP2 provides a fully sharded data parallelism (FSDP) implementation
targeting performant eager-mode while using per-parameter sharding for improved
usability.

.. currentmodule:: torch.distributed.fsdp

The frontend API is ``fully_shard`` that can be called on a ``module``:

.. autofunction:: fully_shard

Calling ``fully_shard(module)`` dynamically constructs a new class that
subclasses ``type(module)`` and an FSDP class ``FSDPModule``. For example, if
we call ``fully_shard(linear)`` on ``linear: nn.Linear``, then FSDP constructs
a new class ``FSDPLinear`` and changes ``linear`` 's type to this. Otherwise,
``fully_shard`` does not change the module structure and parameter
fully-qualified names. The class ``FSDPModule`` allows providing some
FSDP-specific methods on the module.

.. autoclass:: FSDPModule
    :members:
    :member-order: bysource

.. autoclass:: UnshardHandle
    :members:

.. autoclass:: MixedPrecisionPolicy
    :members:

.. autoclass:: OffloadPolicy
    :members:

.. autoclass:: CPUOffloadPolicy
    :members:
