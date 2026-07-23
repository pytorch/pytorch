```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# Distributed Optimizers

:::{warning}
Distributed optimizer is not currently supported when using CUDA tensors
:::

```{eval-rst}
.. automodule:: torch.distributed.optim
    :members: DistributedOptimizer, PostLocalSGDOptimizer, ZeroRedundancyOptimizer
```

```{eval-rst}
.. currentmodule:: torch.distributed.optim.utils
.. autofunction:: register_functional_optim
```
