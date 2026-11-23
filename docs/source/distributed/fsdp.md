# FullyShardedDataParallel

:::{note}
For new code, prefer {func}`torch.distributed.fsdp.fully_shard` (FSDP2). See
the [migration guide](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#fsdp1-to-fsdp2-migration-guide).
:::

```{eval-rst}
.. automodule:: torch.distributed.fsdp
```

```{eval-rst}
.. autoclass:: torch.distributed.fsdp.FullyShardedDataParallel
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.fsdp.BackwardPrefetch
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.fsdp.ShardingStrategy
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.fsdp.MixedPrecision
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.fsdp.CPUOffload
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.fsdp.StateDictType
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.fsdp.StateDictConfig
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.fsdp.FullStateDictConfig
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.fsdp.ShardedStateDictConfig
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.fsdp.LocalStateDictConfig
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.fsdp.OptimStateDictConfig
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.fsdp.FullOptimStateDictConfig
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.fsdp.ShardedOptimStateDictConfig
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.fsdp.LocalOptimStateDictConfig
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.fsdp.StateDictSettings
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.fsdp.OptimStateKeyType
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler
  :members:
```

```{eval-rst}
.. currentmodule:: torch.distributed.fsdp.wrap

.. autoclass:: ModuleWrapPolicy

.. autoclass:: CustomPolicy

.. autofunction:: always_wrap_policy

.. autofunction:: enable_wrap

.. autofunction:: lambda_auto_wrap_policy

.. autofunction:: size_based_auto_wrap_policy

.. autofunction:: transformer_auto_wrap_policy

.. autofunction:: wrap
```
