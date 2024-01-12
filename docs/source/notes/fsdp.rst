.. _fsdp_notes:

FSDP Notes
==========

.. _fsdp_prefetch:

FSDP Prefetch Nuances
---------------------

For overlapping ``forward`` all-gathers with ``forward`` compute, there are two possible mechanisms:

1. Implicit forward prefetching (always enabled)
2. Explicit forward prefetching (``forward_prefetch=True``)

Implicit ``forward`` prefetching refers to relying on issuing the all-gathers from a separate CUDA
stream to allow for overlapping an all-gather with ``forward`` compute issued before it (from the CPU
perspective). For example, if we have layer 0 all-gather -> layer 0 ``forward`` compute -> layer 1
all-gather -> …, then layer 1 all-gather can overlap with layer 0 ``forward`` compute even though the
CPU thread issued it afterwards. (The 1st all-gather will not be able to overlap with anything.)

Explicit ``forward`` prefetching refers to changing the CPU thread’s issue order: e.g. layer 0
all-gather -> layer 1 all-gather -> layer 0 ``forward`` compute -> …. In eager mode, there is no way to
know in general which layer is the next layer (e.g. layer 1 in the example) when still executing on
layer 0. Therefore, explicit ``forward`` prefetching should only be used for models whose execution
order is fixed from iteration to iteration (which we sometimes call “static graph”). An example of a
model that does not satisfy this constraint is `FLAVA
<https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/>`_).

Explicit ``forward`` prefetching only saves the time taken to issue a layer’s ``forward`` compute kernels at
the cost that the next all-gather’s output tensor must be allocated while the current one is still
in use. By issuing the next all- gather before the current ``forward`` compute kernels, the next
all-gather can start sooner on GPU. For most LLM workloads, this is not the case, so there is no
motivation for enabling ``forward_prefetch=True``.

In contrast, for ``backward``, we must use explicit ``backward`` prefetching or else there will be 0 overlap
of communication and computation. The reason is because we use a single NCCL process group for both
all-gather and reduce-scatter (partially because in earlier NCCL versions, it was not safe to use
multiple concurrently on the same device over the same ranks). A single NCCL process group means a
single internal NCCL stream on which reduce-scatters and all-gathers run serially. As such, unless
we explicitly reorder the CPU issue order to be next all-gather -> current reduce-scatter, then the
current reduce-scatter would block the next all-gather and hence the next ``backward`` computation,
preventing the current reduce-scatter from overlapping.


.. _fsdp_comms_buffer_size:

Communication buffer size
-------------------------

In FSDP the communications are:

1. all-gather on weights in ``forward``
2. all-gather on weights in ``backward``
3. reduce-scatter on gradients in ``backward``

If activation checkpointing (:func:`~torch.utils.checkpoint.checkpoint`) is used there is no
additional communication since the weights are prefetches anyway during ``backward``.

In the FSDP design, the communication buffer size is determined as follows: Each call to
:class:`FullyShardedDataParallel` creates one communication group consisting of the parameters in
``module.parameters()`` except any already assigned to a nested :class:`FullyShardedDataParallel`
instance. For example, for Llama, if you apply :class:`FullyShardedDataParallel` to every
transformer block and also to the root module, then there is one communication group for each
transformer block and finally one communication group with the initial embedding and final linear.
Each communication group corresponds to a single all-gather call and single reduce-scatter call. In
that way, how you apply :class:`FullyShardedDataParallel` determines the communication size. In
general, applying FSDP to each transformer block is a good heuristic for LLMs, and it is hard to do
better than that given the current design.

As explained in :ref:`fsdp_prefetch` in the implicit ``forward`` prefetching case of layer 0
all-gather -> layer 0 forward compute -> layer 1 all-gather there is a need for only one all-gather
buffer because only when layer 0 compute started then layer 1 all-gather is issued. But there will
be a need for 2x buffer size with ``forward_prefetch=True``.
