.. role:: hidden
    :class: hidden-section

Tensor Parallelism - torch.distributed.tensor.parallel
======================================================

Tensor Parallelism(TP) is built on top of the PyTorch DistributedTensor
(`DTensor <https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tensor/README.md>`__)
and provides different parallelism styles: Colwise and Rowwise Parallelism.

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

To simply configure the nn.Module's inputs and outputs with DTensor layouts
and perform necessary layout redistributions, without distribute the module
parameters to DTensors, the following classes can be used in
the ``parallelize_plan``of ``parallelize_module``:

.. autoclass:: torch.distributed.tensor.parallel.PrepareModuleInput
  :members:
  :undoc-members:

.. autoclass:: torch.distributed.tensor.parallel.PrepareModuleOutput
  :members:
  :undoc-members:


For models like Transformer, we recommend users to use ``ColwiseParallel``
and ``RowwiseParallel`` together in the parallelize_plan for achieve the desired
sharding for the entire model (i.e. Attention and MLP).
