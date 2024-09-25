.. role:: hidden
    :class: hidden-section

Tensor Parallelism - torch.distributed.tensor.parallel
======================================================

Tensor Parallelism(TP) is built on top of the PyTorch DistributedTensor
(`DTensor <https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/README.md>`__)
and provides different parallelism styles: Colwise, Rowwise, and Sequence Parallelism.

.. warning ::
    Tensor Parallelism APIs are experimental and subject to change.

The entrypoint to parallelize your ``nn.Module`` using Tensor Parallelism is:

.. automodule:: torch.distributed.tensor.parallel

.. currentmodule:: torch.distributed.tensor.parallel

.. autofunction::  parallelize_module

Tensor Parallelism supports the following parallel styles:

.. autoclass:: torch.distributed.tensor.parallel.ColwiseParallel
  :members:
  :undoc-members:

.. autoclass:: torch.distributed.tensor.parallel.RowwiseParallel
  :members:
  :undoc-members:

.. autoclass:: torch.distributed.tensor.parallel.SequenceParallel
  :members:
  :undoc-members:

To simply configure the nn.Module's inputs and outputs with DTensor layouts
and perform necessary layout redistributions, without distribute the module
parameters to DTensors, the following ``ParallelStyle`` s can be used in
the ``parallelize_plan`` when calling ``parallelize_module``:

.. autoclass:: torch.distributed.tensor.parallel.PrepareModuleInput
  :members:
  :undoc-members:

.. autoclass:: torch.distributed.tensor.parallel.PrepareModuleOutput
  :members:
  :undoc-members:

.. note:: when using the ``Shard(dim)`` as the input/output layouts for the above
  ``ParallelStyle`` s, we assume the input/output activation tensors are evenly sharded on
  the tensor dimension ``dim`` on the ``DeviceMesh`` that TP operates on. For instance,
  since ``RowwiseParallel`` accepts input that is sharded on the last dimension, it assumes
  the input tensor has already been evenly sharded on the last dimension. For the case of uneven
  sharded activation tensors, one could pass in DTensor directly to the partitioned modules,
  and use ``use_local_output=False`` to return DTensor after each ``ParallelStyle``, where
  DTensor could track the uneven sharding information.

For models like Transformer, we recommend users to use ``ColwiseParallel``
and ``RowwiseParallel`` together in the parallelize_plan for achieve the desired
sharding for the entire model (i.e. Attention and MLP).

Parallelized cross-entropy loss computation (loss parallelism), is supported via the following context manager:

.. autofunction:: torch.distributed.tensor.parallel.loss_parallel

.. warning ::
    The loss_parallel API is experimental and subject to change.
