# torch.vm

```{eval-rst}
.. automodule:: torch.vm
```

## Graphs

The default passes make each tensor operation of a block a direct call of its
kernel. `torch.vm.passes.Graphs()` looks one level up: in every block, once
the other passes are done with it, it finds the runs of such calls together
with what sits between them (reads of globals and attributes, constants,
arithmetic on sizes) and takes each run down as a graph. A run ends in front
of everything else: a call of a Python function, an operation that writes to
one of its arguments or draws random numbers, a number read from a tensor. A
run of fewer than `min_ops` operations (4 unless the pass is made with
another number) is no graph.

With `Graphs()` in the list the function runs as it runs without it. The
engine runs a graph as it runs the rest of the block: the same kernels, on
the same values, in the same order, on the autograd tape of eager mode. The
results are those of eager mode bit for bit, and `torch.vm.is_exact` is true
for such a list. What the pass adds is what `torch.vm.regions` reports:

```python
import torch

P = torch.vm.passes


def f(x, w):
    return torch.tanh(x @ w).sum() + (x * 2).mean()


g = torch.vm(f, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs()))
x, w = torch.randn(4, 8), torch.randn(8, 8, requires_grad=True)
for _ in range(3):
    out = g(x, w)
torch.equal(out, f(x, w))  # True
type(out.grad_fn).__name__  # 'AddBackward0': the last operation, on eager's tape
torch.vm.is_exact((*torch.vm.DEFAULT_PASSES, P.Graphs()))  # True
torch.vm.regions(g)
# [{'where': 'f (block 0)[2]', 'state': 'built', 'ops': 6,
#   'operators': ['aten::matmul', 'aten::tanh', 'aten::sum',
#   'aten::mul.Tensor', 'aten::mean', 'aten::add.Tensor'], 'version': 1}]
```

`where` names the function, the block, and the instruction of the first
operation of the graph (`g.dump()` shows them). The state
`"built"` says that the engine runs the graph itself. What was left out of
the graphs of a block that has tensor work on both sides of it is listed
too, with the state `"vm"` and the reason:

```python
def two(x):
    y = x.sin().cos() * 2 + x
    print(end="")
    y.add_(1)
    return (y.tanh() * y).exp().sum() - y.mean()


g = torch.vm(two, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs()))
for _ in range(3):
    g(torch.randn(6))
for found in torch.vm.regions(g):
    print(found["where"], found["state"], found.get("ops"), found.get("reason"))
# two (block 0)[1] built 4 None
# two (block 0)[15] built 6 None
# two (block 0)[10] vm None a call that runs no kernel
# two (block 0)[13] vm None aten::add_.Tensor writes to an argument
```

A graph is a part of one block of one function. It does not go on through a
call of a Python function, a branch or a loop: the function that is called
has graphs of its own, in its own blocks. A model made of modules therefore
has its operations spread over the `forward` of each module, one or two to a
block, and none of those runs is a graph unless `min_ops` is lowered:

```python
model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))


def infer(model, x):
    return model(x)


g = torch.vm(infer, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(min_ops=1)))
for _ in range(3):
    g(model, torch.randn(4, 8))
sorted((found["where"], found["operators"]) for found in torch.vm.regions(g))
# [('Linear.forward (block 0)[4]', ['aten::linear']), ('relu (block 4)[2]', ['aten::relu'])]
```

## API Reference

```{eval-rst}
.. autofunction:: torch.vm.vm
```

```{eval-rst}
.. autodata:: torch.vm.DEFAULT_PASSES
```

```{eval-rst}
.. autofunction:: torch.vm.is_exact
```

```{eval-rst}
.. autofunction:: torch.vm.swap
```

```{eval-rst}
.. autofunction:: torch.vm.fuse_checked
```

```{eval-rst}
.. autoexception:: torch.vm.Mismatch
```

```{eval-rst}
.. autofunction:: torch.vm.load_passes
```

```{eval-rst}
.. autofunction:: torch.vm.regions
```

```{eval-rst}
.. autofunction:: torch.vm.memory
```

```{eval-rst}
.. autofunction:: torch.vm.is_available
```

```{eval-rst}
.. autoexception:: torch.vm.Unsupported
```

## torch.vm.passes

```{eval-rst}
.. automodule:: torch.vm.passes
```

```{eval-rst}
.. autoclass:: torch.vm.passes.BuiltinPass
```

```{eval-rst}
.. autoclass:: torch.vm.passes.Inline
```

```{eval-rst}
.. autoclass:: torch.vm.passes.Unroll
```

```{eval-rst}
.. autoclass:: torch.vm.passes.ModuleShortcuts
```

```{eval-rst}
.. autoclass:: torch.vm.passes.Graphs
```

```{eval-rst}
.. autoclass:: torch.vm.passes.ResolveDispatch
```

```{eval-rst}
.. autoclass:: torch.vm.passes.TensorAccessors
```

```{eval-rst}
.. autoclass:: torch.vm.passes.MethodCalls
```

```{eval-rst}
.. autoclass:: torch.vm.passes.NarrowDims
```

```{eval-rst}
.. autoclass:: torch.vm.passes.Fold
```

```{eval-rst}
.. autoclass:: torch.vm.passes.DCE
```
