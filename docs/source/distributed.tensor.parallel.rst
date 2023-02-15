.. role:: hidden
    :class: hidden-section

Tensor Parallelism - torch.distributed.tensor.parallel
======================================================

Tensor Parallelism(TP) is built on top of DistributedTensor(DTensor) and
provides several Parallelism styles: Rowwise, Colwise and Pairwise Parallelism.

.. warning ::
    Tensor Parallelism APIs are experimental and subject to change.

The entrypoint to parallelize your ``nn.Module`` using Tensor Parallelism is:

.. automodule:: torch.distributed.tensor.parallel

.. currentmodule:: torch.distributed.tensor.parallel

.. autofunction::  parallelize_module

Tensor Parallelism supports the following parallel styles:

.. autoclass:: torch.distributed.tensor.parallel.style.RowwiseParallel
  :members:

.. autoclass:: torch.distributed.tensor.parallel.style.ColwiseParallel
  :members:

.. autoclass:: torch.distributed.tensor.parallel.style.PairwiseParallel
  :members:

Since Tensor Parallelism is built on top of DTensor, we need to specify the
input and output placement of the module with DTensors so it can expectedly
interacts with the module before and after. The followings are functions
used for input/output preparation:


.. currentmodule:: torch.distributed.tensor.parallel.style

.. autofunction::  make_input_replicate_1d
.. autofunction::  make_input_shard_1d
.. autofunction::  make_input_shard_1d_last_dim
.. autofunction::  make_output_replicate_1d
.. autofunction::  make_output_tensor
.. autofunction::  make_output_shard_1d

Currently, there are some constraints which makes it hard for the `nn.MultiheadAttention`
module to work out of box for Tensor Parallelism, so we built this multihead_attention
module for Tensor Parallelism users. Also, in ``parallelize_module``, we automatically
swap ``nn.MultiheadAttention`` to this custom module when specifying ``PairwiseParallel``.

.. autoclass:: torch.distributed.tensor.parallel.multihead_attention_tp.TensorParallelMultiheadAttention
  :members:

We also enabled 2D parallelism to integrate with ``FullyShardedDataParallel``.
Users just need to call the following API explicitly:


.. currentmodule:: torch.distributed.tensor.parallel.fsdp
.. autofunction::  enable_2d_with_fsdp
