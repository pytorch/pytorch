.. role:: hidden
    :class: hidden-section

Tensor Parallelism - torch.distributed.tensor.parallel
========================

We built Tensor Parallelism(TP) on top of DistributedTensor(DTensor) and
provide several Parallelis styles: Rowwise, Colwise and Pairwise Parallelism.

The entrypoint to parallelize your module and use tensor parallelism is:

.. automodule:: torch.distributed.tensor.parallel

.. currentmodule:: torch.distributed.tensor.parallel

.. autofunction::  parallelize_module

All parallel styles inherits the following:

.. autoclass:: torch.distributed.tensor.parallel.style.ParallelStyle
  :members:

We have following parallel styles:

.. autoclass:: torch.distributed.tensor.parallel.style.RowwiseParallel
  :members:

.. autoclass:: torch.distributed.tensor.parallel.style.ColwiseParallel
  :members:

.. autoclass:: torch.distributed.tensor.parallel.style.PairwiseParallel
  :members:

Because we use DTensors within the TP module, we need to specify how we want
it to interact with the module before and after. The followings are functions
used for input/output preparation:


.. currentmodule:: torch.distributed.tensor.parallel.style

.. autofunction::  make_input_replicate_1d
.. autofunction::  make_input_shard_1d
.. autofunction::  make_input_shard_1d_dim_last
.. autofunction::  make_output_replicate_1d
.. autofunction::  make_output_tensor
.. autofunction::  make_output_shard_1d

Currently, there are some constraints which makes it hard for attention module
to work out of box for TP, so we built this multihead_attention module for TP:

.. autoclass:: torch.distributed.tensor.parallel.multihead_attention_tp.TensorParallelMultiheadAttention
  :members:

We also enabled 2D parallelism to integrate with ``FullyShardedDataParallel``.
Users just need to call the following API explicitly:


.. currentmodule:: torch.distributed.tensor.parallel.fsdp
.. autofunction::  is_available
