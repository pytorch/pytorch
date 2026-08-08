# torch.utils.nan_guard

```{note}
`NaNGuard` registers a forward hook on every submodule of the watched module
and applies `torch.isnan` (and `torch.isinf` when `check_inf=True`) to every
floating-point or complex tensor in the output. On accelerator devices these
reductions force a host sync, so the guard is intended for debugging rather
than steady-state training. For backward-pass anomaly detection, see
{class}`torch.autograd.detect_anomaly`.

Forward hooks are not honored by modules that have been compiled with
`torch.jit.script`; the guard is a no-op for those modules.
```

```{eval-rst}
.. currentmodule:: torch.utils.nan_guard
.. autoclass:: NaNGuard
.. autoclass:: NaNGuardError
.. autofunction:: nan_guard
```
