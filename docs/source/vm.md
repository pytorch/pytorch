# torch.vm

```{eval-rst}
.. automodule:: torch.vm
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
.. autoclass:: torch.vm.Type
```

```{eval-rst}
.. autoclass:: torch.vm.TensorType
    :members:
```

```{eval-rst}
.. autoclass:: torch.vm.IntType
```

```{eval-rst}
.. autoclass:: torch.vm.Interval
    :members:
```

```{eval-rst}
.. autofunction:: torch.vm.type_of
```

```{eval-rst}
.. autofunction:: torch.vm.materialize
```

```{eval-rst}
.. autofunction:: torch.vm.guard
```

```{eval-rst}
.. autoclass:: torch.vm.Check
```

```{eval-rst}
.. autoclass:: torch.vm.Dtype
```

```{eval-rst}
.. autoclass:: torch.vm.Device
```

```{eval-rst}
.. autoclass:: torch.vm.Dim
```

```{eval-rst}
.. autoclass:: torch.vm.Stride
```

```{eval-rst}
.. autoclass:: torch.vm.Contiguous
```

```{eval-rst}
.. autoclass:: torch.vm.Aligned
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
