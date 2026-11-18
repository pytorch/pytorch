# torch.vm

```{eval-rst}
.. automodule:: torch.vm
```

## API Reference

```{eval-rst}
.. autofunction:: torch.vm.vm
```

```{eval-rst}
.. autodata:: torch.vm.DefaultCFGPasses
```

```{eval-rst}
.. autodata:: torch.vm.DefaultBlockPasses
```

```{eval-rst}
.. autodata:: torch.vm.FAST
```

```{eval-rst}
.. autofunction:: torch.vm.inlining_pass
```

```{eval-rst}
.. autofunction:: torch.vm.loop_unrolling_pass
```

```{eval-rst}
.. autofunction:: torch.vm.module_shortcuts_pass
```

```{eval-rst}
.. autofunction:: torch.vm.graphs_pass
```

```{eval-rst}
.. autofunction:: torch.vm.resolve_dispatch_pass
```

```{eval-rst}
.. autofunction:: torch.vm.tensor_accessors_pass
```

```{eval-rst}
.. autofunction:: torch.vm.method_calls_pass
```

```{eval-rst}
.. autofunction:: torch.vm.narrow_dims_pass
```

```{eval-rst}
.. autofunction:: torch.vm.fold_pass
```

```{eval-rst}
.. autofunction:: torch.vm.dce_pass
```

What a pass of one's own is made with (`swap`, `fuse_checked`, `Mismatch`, `load_passes`), what reports on a function (`regions`, `memory`),
`is_available` and `Unsupported` are those of {mod}`torch.rewrite`, under the same names.
