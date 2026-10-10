# Functional Collectives

```{eval-rst}
.. automodule:: torch.distributed.functional_collectives
```

Functional collectives take the group as an argument. Under `torch.compile` they are traced into the graph so the compiler can reorder and overlap communication.

```{eval-rst}
.. note::
    The :mod:`torch.distributed.nn.functional` module is deprecated in favor of this module.
```

```python
import torch.distributed as dist
import torch.distributed.functional_collectives as funcol

# group can be a ProcessGroup, DeviceMesh, (DeviceMesh, dim) or group name.
out = funcol.all_reduce(tensor, "sum", group=dist.group.WORLD)
gathered = funcol.all_gather_single(tensor, gather_dim=0, group=mesh)
```

```{eval-rst}
.. autofunction:: all_reduce
```

```{eval-rst}
.. autofunction:: all_reduce_coalesced
```

```{eval-rst}
.. autofunction:: all_gather_single
```

```{eval-rst}
.. autofunction:: all_gather_single_coalesced
```

```{eval-rst}
.. autofunction:: reduce_scatter_single
```

```{eval-rst}
.. autofunction:: reduce_scatter_single_coalesced
```

```{eval-rst}
.. autofunction:: all_to_all_single
```

```{eval-rst}
.. autofunction:: broadcast
```

```{eval-rst}
.. autofunction:: permute_tensor
```

```{eval-rst}
.. autofunction:: wait_tensor
```

```{eval-rst}
.. autofunction:: wait_tensors
```

```{eval-rst}
.. autoclass:: AsyncCollectiveTensor
    :members: wait
```
