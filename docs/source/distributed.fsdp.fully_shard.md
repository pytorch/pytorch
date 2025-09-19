# torch.distributed.fsdp.fully_shard

## PyTorch FSDP2 (`fully_shard`)

PyTorch FSDP2 ([RFC](<https://github.com/pytorch/pytorch/issues/114299>)) provides
a fully sharded data parallelism (FSDP) implementation targeting performant
eager-mode while using per-parameter sharding for improved usability

- See the [Getting Started with FSDP2](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
  tutorial for more information.

- If you are currently using FSDP1, consider migrating to FSDP2 using our
  [migration guide](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#fsdp1-to-fsdp2-migration-guide).


The user contract for ``fully_shard(model)`` is as follows

- For model initialization, fully_shard converts model.parameters() from
  plain torch.Tensor to DTensor in-place. The parameters are moved to the
  appropriate device according to the device mesh.

- Before forward and backward passes, pre-forward/backward hooks are
  responsible for all-gathering the parameters and converting model.parameters()
  from DTensor to plain torch.Tensor.

- After forward and backward passes, post-forward/backward hooks free
  the unsharded parameters (no communication needed) and convert
  model.parameters() from plain torch.Tensor back to DTensor.

- For the optimizer, it must be initialized with the DTensor model.parameters(),
  and the optimizer step should be performed on DTensor parameters.

- Call ``model(input)`` instead of ``model.forward(input)`` to trigger pre-forward
  hooks to all-gather parameters. To make model.forward(input) work, users must
  either call ``model.unshard()`` explicitly or use ``register_fsdp_forward_method(model, "forward")``
  to register the forward method for hooking.

- fully_shard groups parameters together for a single all-gather. User should apply
  fully_shard in a bottom-up manner. For example, in a Transformer model, fully_shard
  should be applied to each layer before applying it to the root model. When applied
  to the root model, fully_shard excludes model.parameters() from each layer and groups
  the remaining parameters (e.g., embeddings, output projection) into a single
  all-gather group.

- ``type(model)`` is "unioned" with ``FSDPModule`` in-place. For example, if model
  is originally of type nn.Linear, then fully_shard changes ``type(model)`` from
  nn.Linear to ``FSDPLinear`` in-place. ``FSDPLinear`` is an instance of both
  nn.Linear and ``FSDPModule``. It retains all methods of nn.Linear while also
  exposing FSDP2-specific APIs under FSDPModule, such as ``reshard()`` and
  ``unshard()``.

- Fully Qualified Names (FQNs) for parameters remain unchanged. If we call
  ``model.state_dict()``, the FQNs are the same before and after applying
  fully_shard. This is because fully_shard does not wrap the module but only
  registers hooks to the original module.


Compared to PyTorch FSDP1 (`FullyShardedDataParallel`):

- FSDP2 uses `DTensor`-based dim-0 per-parameter sharding for a simpler
  sharding representation compared to FSDP1's flat-parameter sharding, while
  preserving similar throughput performance. More specifically, FSDP2 chunks
  each parameter on dim-0 across the data parallel workers (using
  `torch.chunk(dim=0)`), whereas FSDP1 flattens, concatenates, and chunks a
  group of tensors together, making reasoning about what data is present on
  each worker and resharding to different parallelisms complex. Per-parameter
  sharding provides a more intuitive user experience, relaxes constraints
  around frozen parameters, and allows for communication-free (sharded) state
  dicts, which otherwise require all-gathers in FSDP1.
- FSDP2 implements a different memory management approach to handle the
  multi-stream usages that avoids `torch.Tensor.record_stream`. This ensures
  deterministic and expected memory usage and does not require blocking the CPU
  like in FSDP1's `limit_all_gathers=True`.
- FSDP2 exposes APIs for manual control over prefetching and collective
  scheduling, allowing power users more customization. See the methods on
  `FSDPModule` below for details.
- FSDP2 simplifies some of the API surface: e.g. FSDP2 does not directly
  support full state dicts. Instead, users can reshard the sharded state dicts
  containing `DTensor` s to full state dicts themselves using `DTensor`
  APIs like `DTensor.full_tensor()` or by using higher-level APIs like
  [PyTorch Distributed Checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html) 's
  distributed state dict APIs. Also, some other args have been removed; see
  [here](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md) for
  details.


```{eval-rst}
.. currentmodule:: torch.distributed.fsdp
```

The frontend API is `fully_shard` that can be called on a `module`:

```{eval-rst}
.. autofunction:: fully_shard
```

```{eval-rst}
.. autoclass:: FSDPModule
    :members:
    :member-order: bysource
```

```{eval-rst}
.. autoclass:: UnshardHandle
    :members:
```

```{eval-rst}
.. autofunction:: register_fsdp_forward_method
```

```{eval-rst}
.. autoclass:: MixedPrecisionPolicy
    :members:
```

```{eval-rst}
.. autoclass:: OffloadPolicy
    :members:
```

```{eval-rst}
.. autoclass:: CPUOffloadPolicy
    :members:
```
