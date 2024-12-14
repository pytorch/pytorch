torch.distributed.fsdp.fully_shard
==================================

PyTorch FSDP2 (``fully_shard``)
-------------------------------

PyTorch FSDP2 provides a fully sharded data parallelism (FSDP) implementation
targeting performant eager-mode while using per-parameter sharding for improved
usability.

- If you are new to FSDP, we recommend that you start with FSDP2 due to improved
  usability.
- If you are currently using FSDP1, consider evaluating the following
  differences to see if you should switch to FSDP2:

Compared to PyTorch FSDP1 (``FullyShardedDataParallel``):

- FSDP2 uses ``DTensor``-based dim-0 per-parameter sharding for a simpler
  sharding representation compared to FSDP1's flat-parameter sharding, while
  preserving similar throughput performance. More specifically, FSDP2 chunks
  each parameter on dim-0 across the data parallel workers (using
  ``torch.chunk(dim=0)``), whereas FSDP1 flattens, concatenates, and chunks a
  group of tensors together, making reasoning about what data is present on
  each worker and resharding to different parallelisms complex. Per-parameter
  sharding provides a more intuitive user experience, relaxes constraints
  around frozen parameters, and allows for communication-free (sharded) state
  dicts, which otherwise require all-gathers in FSDP1.
- FSDP2 implements a different memory management approach to handle the
  multi-stream usages that avoids ``torch.Tensor.record_stream``. This ensures
  deterministic and expected memory usage and does not require blocking the CPU
  like in FSDP1's ``limit_all_gathers=True``.
- FSDP2 exposes APIs for manual control over prefetching and collective
  scheduling, allowing power users more customization. See the methods on
  ``FSDPModule`` below for details.
- FSDP2 simplifies some of the API surface: e.g. FSDP2 does not directly
  support full state dicts. Instead, users can reshard the sharded state dicts
  containing ``DTensor`` s to full state dicts themselves using ``DTensor``
  APIs like ``DTensor.full_tensor()`` or by using higher-level APIs like
  `PyTorch Distributed Checkpoint <https://pytorch.org/docs/stable/distributed.checkpoint.html>`_ 's
  distributed state dict APIs. Also, some other args have been removed; see
  `here <https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md>`_ for
  details.

If you are onboarding FSDP for the first time or if any of the above appeals to
your use case, we recommend that you consider using FSDP2.

See `this RFC <https://github.com/pytorch/pytorch/issues/114299>`_ for details
on system design and implementation.

.. note::
  ``torch.distributed.fsdp.fully_shard`` is currently in prototype state and
  under development. The core API will likely not change, but we may make some
  API changes if necessary.

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
