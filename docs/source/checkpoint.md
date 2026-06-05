# torch.utils.checkpoint

```{note}
Checkpointing is implemented by rerunning a forward-pass segment for
each checkpointed segment during backward propagation.  This can cause persistent
states like the RNG state to be more advanced than they would without
checkpointing.  By default, checkpointing includes logic to juggle
the RNG state such that checkpointed passes making use of RNG
(through dropout for example) have deterministic output as
compared to non-checkpointed passes.  The logic to stash and restore
RNG states can incur a moderate performance hit depending on the runtime
of checkpointed operations.  If deterministic output compared to
non-checkpointed passes is not required, supply `preserve_rng_state=False`
to `checkpoint` or `checkpoint_sequential` to omit stashing and
restoring the RNG state during each checkpoint.

The stashing logic saves and restores the RNG state for CPU and another
device type (infer the device type from Tensor arguments excluding CPU
tensors by `_infer_device_type`) to the `run_fn`. If there are multiple
devices, device state will only be saved for devices of a single device type,
and the remaining devices will be ignored. Consequently, if any checkpointed
functions involve randomness, this may result in incorrect gradients. (Note
that if CUDA devices are among the devices detected, it will be prioritized;
otherwise, the first device encountered will be selected.) If there are no
CPU-tensors, the default device type state (default value is `cuda`, and it
could be set to other device by `DefaultDeviceType`) will be saved and restored.
However, the logic has no way to anticipate if the user will move
Tensors to a new device within the `run_fn` itself.  Therefore, if you move
Tensors to a new device ("new" meaning not belonging to the set of
[current device + devices of Tensor arguments]) within `run_fn`, deterministic
output compared to non-checkpointed passes is never guaranteed.
```

## Activation Memory Budget with `torch.compile`

`torch.compile` uses AOTAutograd to trace the forward and backward pass
ahead of time. For training graphs, AOTAutograd's partitioner decides which
forward intermediates to save for backward and which intermediates to
recompute. Use `torch._functorch.config.activation_memory_budget` to control
that memory/runtime tradeoff for compiled regions:

```python
import torch
import torch._functorch.config

with torch._functorch.config.patch(activation_memory_budget=0.5):
    compiled_step = torch.compile(train_step)
    # The first call triggers compilation with the patched budget.
    loss = compiled_step(*args)
    loss.backward()
```

The option is in the `torch._functorch.config` namespace, not
`torch._dynamo.config`, because it is consumed by AOTAutograd after TorchDynamo
captures the graph. Set it before the relevant compile trace, or use
`torch._functorch.config.patch(...)` around the `torch.compile` call and first
invocation that triggers compilation.

Valid values are floats in the inclusive range `0.0` to `1.0`. Values outside
that range raise an error. The default is `1.0`, which chooses the
runtime-optimized partitioning strategy. `0.0` corresponds to applying
activation checkpointing to the full compiled region, saving the minimum
eligible activation state and recomputing more during backward. Intermediate
values ask the partitioner to choose the fastest plan that fits within the
normalized activation memory budget. Lower budgets can reduce saved activation
memory, but may increase backward compute.

Use `torch.autograd.graph.region_activation_memory_budget(...)` to override the
global budget for a compiled region.

Related advanced knobs live in the same namespace:

- `torch._functorch.config.activation_memory_budget_solver` selects the
  knapsack solver used by the partitioner. The default is `"dp"`; other
  built-in choices include `"greedy"`, `"ilp"` (requires SciPy), and
  `"dp_knapsack_sliding_hirschberg"`.
- `torch._functorch.config.activation_memory_budget_runtime_estimator` controls
  how recomputation cost is estimated. The default is `"flops"`; `"profile"`
  benchmarks operators, and `"testing"` is intended for tests.
- Setting `torch._functorch.config.visualize_memory_budget_pareto` to `True`
  causes the partitioner to write an SVG Pareto frontier for memory budget
  versus recomputation runtime. Use
  `torch._functorch.config.memory_budget_pareto_dir` to choose the output
  directory.

```{eval-rst}
.. currentmodule:: torch.utils.checkpoint
.. autofunction:: checkpoint
.. autofunction:: checkpoint_sequential
.. autofunction:: set_checkpoint_debug_enabled
.. autoclass:: CheckpointPolicy
.. autoclass:: SelectiveCheckpointContext
.. autofunction:: create_selective_checkpoint_contexts
.. autoclass:: GraphExecGroup
.. autofunction:: set_checkpoint_early_stop
.. autofunction:: set_device_states
```
