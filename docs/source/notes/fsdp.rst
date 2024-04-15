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

.. _fsdp_comms_payload_size:

Communication payload size
--------------------------

In FSDP the communications are:

1. all-gather on parameters in ``forward``
2. all-gather on parameters in ``backward``
3. reduce-scatter on gradients in ``backward``

If activation checkpointing (:func:`~torch.utils.checkpoint.checkpoint`) is used there is no
additional communication since the parameters are prefetched anyway during ``backward``.

In the FSDP design, the communication payload per rank is determined as follows: Each call to
:class:`FullyShardedDataParallel` creates one communication group consisting of the parameters in
``module.parameters()`` except any already assigned to a nested :class:`FullyShardedDataParallel`
instance. For example, for Llama, if you apply :class:`FullyShardedDataParallel` to every
transformer block and also to the root module, then there is one communication group for each
transformer block and finally one communication group with the initial embedding and final linear.
Each communication group corresponds to a single all-gather call and single reduce-scatter call. In
that way, how you apply :class:`FullyShardedDataParallel` determines the communication size. In
general, applying FSDP to each transformer block is a good heuristic for LLMs, and it is hard to do
better than that given the current design.

Let's consider an example where we have a Transformer-based model sharded over 8 GPUs, where the
sharding happens at the transformer block-level only, and each transformer block contains 1.6B
parameters and the parameters are in fp32 (4 bytes each). Which means that once sharded, each
transformer block will contain 0.2B parameters on each rank.

* The ``forward`` pass will communicate in chunks of ``0.2*4 = 0.8GB`` in all-gather
* The ``backward`` pass will communicate 2 times ``0.8GB`` each (1x all-gather and 1x reduce-scatter)

In other words there will be 3 communications with a payload of ``0.8GB`` each. If the model was
comprised of 10 transformer blocks there would be a total of 30 communications for a total of
``30*0.8=24GB``.

To formalize the payload size per communication per rank is
``total_transformer_block_params_in_B*dtype_bytes/num_gpus`` (GBs).

Please note that in this example we didn't include the additional communications required for the
embedding, which should be accounted for as well. And the math would depend on whether the input and
output embeddings are tied or not. If they aren't tied there will be 2x more communications.

.. _fsdp_buffers_sizes:

FSDP buffers sizes
------------------

First, let's cover the buffers allocated for communications:

``forward`` currently requires 2x all-gather buffer size. Here is why:

As explained in :ref:`fsdp_prefetch` in the case of explicit ``forward`` prefetching
(``forward_prefetch=True`) case of layer 0 all-gather -> layer 0 forward compute -> layer 1
all-gather there is a need for 2 all-gather-sized buffers, because one buffer is used in the current ``forward`` while the other is used to do the prefetching.

While the implicit ``forward`` prefetching (``forward_prefetch=False``, default) case of the same sequence in theory should need only 1 buffer, in reality it's still 2x all-gather-sized buffers. The reason is that in the flat-parameter FSDP design, we do not copy-out of the all-gather buffer. The parameters used for compute are directly viewed into the all-gather buffer (in fact, the main benefit of the "flat parameter" is exactly this reason). In that case, while 'layer 1 all-gather' is overlapping with 'layer 0 forward compute', the 'layer 0 forward compute' is using the parameters viewed into the 'layer 0 all-gather' buffer.

A natural question then is, when would you want ``forward_prefetch=False``? For static-graph models (like most LLMs), there is a major technical reason. It is more that, practically, we added this option quickly for some CPU-bound internal models and have not tested every code path with it in unit testing, so we are less confident in it. ``forward_prefetching=False`` can be slightly easier to reason about since we do not have to check the recorded forward order as a possible 'failure mode'; a module's all-gather can always be found under its own ``record_function`` label in its profiler trace.

``backward`` currently requires at least 2x all-gather buffer size and potentially a bit more. Here is why:

The current FSDP design uses ``recordStream`` to manage allocations produced in one stream consumed in another, which can lead to more memory usage than expected. How much more can be "non-deterministic" in that it depends on GPU kernel timing relative to the CPU. The ``limit_all_gathers=True`` argument is a mitigation to that - for more details refer to this discussion is `FSDP & CUDACachingAllocator <https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486/1>`_.

The way existing FSDP works with autograd:

* Existing FSDP all-gathers the ``flat_param``, which is the autograd leaf.
* It calls ``torch.split`` to get 1D views into the ``flat_param`` corresponding to its constituent original parameters.
* It calls ``torch.view`` on each 1D split to view back to ND.
* This means that in ``backward``, we end up with ``ViewBackward`` (ND -> 1D) and ``SplitWithSizesBackward`` (which is a concat). In particular, each individual gradient is computed as a separate allocation, and an explicit concat happens to construct the reduce-scatter input buffer. This implies actually a 2x buffer size for reduce-scatter at that peak memory point.

In summary, for ``backward``, it is about 2x buffer size for reduce-scatter plus any ``recordStream`` effects.

Second, let's discuss the additional buffers:

Once the sharded parameters are gathered from all ranks, they require an additional buffer of `total_transformer_block_params_in_B*dtype_bytes` for the full parameters - so continuing the example from earlier if each transformer block is 1.6B parameters and the parameters are in fp32, then it'd be `1.6*4=6.4GB` buffer.

And there is a need for 2 of those buffers, since there is one currently being used and another being prefetched.

To summarize, we have:

1. 2 times communication buffers of ``total_transformer_block_params_in_B*dtype_bytes/num_gpus``
2. 2 times unsharded transformer block parameters buffer ````total_transformer_block_params_in_B*dtype_bytes``

or if you have been following the example:

1. ``2*1.6*4/8=1.6GB``
2. ``2**1.6*4=12.8GB``

and the total of ``14.4GB``.

Now let's briefly discuss what happens to the embeddings as we have left those out from the calculations:

Given the rule we discussed that you included in the note starting with "the communication buffer
size is determined as follows", we can analyze as follows:

* Suppose we apply FSDP to the root module (e.g. the ``Transformer`` class). Suppose we further apply FSDP to each transformer block (e.g. the ``TransformerBlock`` class).
* Most commonly, the embedding and final linear projection are direct children of the root ``Transformer`` class.
* Following our rule, that means that the embedding and final linear projection are assigned to the root ``Transformer``'s flat parameter.
* We have _another_ special rule, which is that the root does not free its parameters after forward because they will be anyways immediately all-gathered in backward.
* Putting this together, this means that the root's flat parameter including the embedding and final projection are all-gathered to begin forward and kept in GPU memory until the end of backward.
* If the embedding and final linear are not weight-tied, then we _could_ further apply FSDP to the embedding and to the final linear. For weight-tied parameters, we require them to be part of the same flat parameter (or else it would get double-counted). That would allow the embedding to be freed after its usage in forward and only all-gathered toward the end of backward.
* Hopefully, this gives a better sense -- each FSDP module gets assigned parameters in its ``module.parameters`` except those already assigned to another nested FSDP module, and the FSDP module's ``forward`` defines the 'live' interval for its parameters. Hence, the nested ``nn.Module`` structure can affect the all-gather/free schedule and hence the memory/throughput performance.
