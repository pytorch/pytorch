torch.distributed.fsdp.fully_shard
==================================

PyTorch FSDP2 (``fully_shard``)
-------------------------------

PyTorch FSDP2 provides a fully sharded data parallelism (FSDP) implementation
targeting performant eager-mode while using per-parameter sharding for improved
usability. Compared to PyTorch FSDP1 (``FullyShardedDataParallel``):

- FSDP2 uses ``DTensor``-based dim-0 per-parameter sharding for a simpler
  sharding representation compared to FSDP1's flat-parameter sharding, while
  preserving similar throughput performance. This provides a more intuitive
  user experience, relaxes constraints around frozen parameters, and allows for
  communication-free sharded state dicts.
- FSDP2 implements a different memory management approach to handle the
  multi-stream usages that avoids ``torch.Tensor.record_stream``. This ensures
  deterministic and expected memory usage and does not require blocking the CPU
  like in FSDP1's ``limit_all_gathers=True``.
- FSDP2 exposes APIs for manual control over prefetching and collective
  scheduling, allowing power users more customization. See the methods on
  ``FSDPModule`` below for details.

.. currentmodule:: torch.distributed.fsdp

The frontend API is ``fully_shard`` that can be called on a ``module``:

.. autofunction:: fully_shard

Calling ``fully_shard(module)`` dynamically constructs a new class that
subclasses ``type(module)`` and an FSDP class ``FSDPModule``. For example, if
we call ``fully_shard(linear)`` on a module ``linear: nn.Linear``, then FSDP
constructs a new class ``FSDPLinear`` and changes ``linear`` 's type to this.
Otherwise, ``fully_shard`` does not change the module structure and parameter
fully-qualified names. The class ``FSDPModule`` allows providing some
FSDP-specific methods on the module.

.. autoclass:: FSDPModule
    :members:
    :member-order: bysource

.. autoclass:: UnshardHandle
    :members:

.. autofunction:: register_fsdp_forward_method

.. autoclass:: MixedPrecisionPolicy
    :members:

.. autoclass:: OffloadPolicy
    :members:

.. autoclass:: CPUOffloadPolicy
    :members:
