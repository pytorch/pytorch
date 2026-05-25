(compiler_backward)=

``torch.compile`` has different autograd semantics
==================================================

When you apply ``torch.compile`` to a function in your model's forward pass,
it will automatically generate a backward pass for the compiled function.
During compilation, it will trace out a graph for the backward pass that
is used whenever autograd is invoked. We refer to the component inside
``torch.compile`` that is responsible for this as ``AOTDispatcher``
(sometimes known as ``AOTAutograd``).

As so, ``torch.compile`` bakes in details of the computation into the
traced-out backward graph during compilation of the function
in the forward pass.
However, in eager-mode PyTorch, the backward computation is dynamic:
outside of the forward pass, you can wrap the call to
``tensor.backward()`` or ``torch.autograd.grad(...)``
in a context manager that may change its behavior.

This page documents how ``torch.compile``'s autograd semantics differ from
eager-mode PyTorch and how to work around it.

Activation checkpointing memory budget
--------------------------------------

``torch.compile`` uses AOTAutograd to trace the forward and backward pass
ahead of time. For training graphs, AOTAutograd's partitioner decides which
forward intermediates to save for backward and which intermediates to recompute.
Use ``torch._functorch.config.activation_memory_budget`` to control that
memory/runtime tradeoff for compiled regions:

```py
import torch
import torch._functorch.config

with torch._functorch.config.patch(activation_memory_budget=0.5):
    compiled_step = torch.compile(train_step)
    loss = compiled_step(*args)
    loss.backward()
```

The option is in the ``torch._functorch.config`` namespace, not
``torch._dynamo.config``, because it is consumed by AOTAutograd after
TorchDynamo captures the graph. Set it before the relevant compile trace, or
use ``torch._functorch.config.patch(...)`` around the ``torch.compile`` call
and first invocation that triggers compilation.

Valid values are floats in the inclusive range ``0.0`` to ``1.0``. Values
outside that range raise an error. The default is ``1.0``, which chooses the
runtime-optimized partitioning strategy. ``0.0`` corresponds to applying
activation checkpointing to the full compiled region, saving the minimum
eligible activation state and recomputing more during backward. Intermediate
values ask the partitioner to choose the fastest plan that fits within the
normalized activation memory budget. Lower budgets can reduce saved activation
memory, but may increase backward compute.

Related advanced knobs live in the same namespace:

- ``torch._functorch.config.activation_memory_budget_solver`` selects the
  knapsack solver used by the partitioner. The default is ``"dp"``; other
  built-in choices include ``"greedy"``, ``"ilp"`` (requires SciPy), and
  ``"dp_knapsack_sliding_hirschberg"``.
- ``torch._functorch.config.activation_memory_budget_runtime_estimator``
  controls how recomputation cost is estimated. The default is ``"flops"``;
  ``"profile"`` benchmarks operators, and ``"testing"`` is intended for tests.
- ``torch._functorch.config.visualize_memory_budget_pareto`` writes an SVG
  Pareto frontier for memory budget versus recomputation runtime when enabled.
  Use ``torch._functorch.config.memory_budget_pareto_dir`` to choose the output
  directory.

``Autocast`` behavior
---------------------

``torch.compile`` bakes in an assumption on if the backward pass will be
run under an ambient autocast context manager. By default,
Use ``torch._functorch.config.backward_pass_autocast``
to control that assumption; an incorrect assumption may lead to silent
incorrectness.

The options are either:
- `"same_as_forward"` (default).
  We assume that the backward of the ``torch.compile``'ed region
  will be run under the same autocast context manager that the region was run
  under (if any). Use this if your code looks like the following:
  ```py
  with torch.amp.autocast(...):
      y = torch.compile(region)(x)
      ...
      # backward pass run under the same autocast context as the compiled region
      z.backward()
  ```
- `"off"`. We assume that the backward of the torch.compile'd region will
  not be run under any autocast context managers.
  Use this if your code looks like the following:
  ```py
  with torch.amp.autocast(...):
      y = torch.compile(region)(x)
      ...
  # Backward pass runs under no autocast.
  z.backward()
  ```
- There is a third option. If you set ``torch._functorch.config.backward_pass_autocast``
  to a list of kwargs, we will assume the backward pass runs under an autocast context
  constructed by those kwargs.

  For example, if your code looks like the following:
  ```py
  y = torch.compile(region)(x)
  ...
  # Backward pass runs under special context manager
  with torch.amp.autocast(**kwargs):
      z.backward()
  ```
  then set ``torch._functorch.config.backward_pass_autocast = kwargs``.

Use ``patch`` to apply the option to a specific ``torch.compile`` call:
```py
with torch.amp.autocast(...):
    with torch._functorch.config.patch(backward_pass_autocast="same_as_forward")
    y = torch.compile(region)(x)
    ...
    # backward pass run under the same autocast context as the compiled region
    z.backward()
```
