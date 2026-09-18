# Design notes

## Tracing: what we forgo vs `ProxyTorchDispatchMode.proxy_call`

`tracing.py` emits the op's FX node by hand (`tracer.create_proxy` +
`track_tensor_tree` in `_run_and_build_overload` / `trace_custom_op`) instead of
going through the dispatcher and `proxy_call`. `proxy_call` does much more; this
records what we skip and why, so it isn't rediscovered as a "bug" later.

### Deliberate inversion
- **We run the impl under `disable_proxy_modes_tracing()`.** `proxy_call` runs a
  single aten op *under* the mode; our impl is a *composite Python function*, so
  if we didn't disable tracing, all of its internal aten ops would be traced as
  individual nodes and we'd get the decomposition PLUS our marker. Disabling is
  what makes the op opaque (exactly one node) -- the whole point of the design.

### Intentionally skipped (not applicable / undesirable)
- **Decomposition** (`maybe_handle_decomp`, CompositeImplicit `func.decompose`):
  we want the op to stay one opaque node.
- **aten special cases** (`lift_fresh`->`lift_fresh_copy`, `is_nonzero`).
- **`_maybe_record_pointwise_barrier`, `set_original_aten_op`**: inductor
  fusion / decomp metadata, irrelevant to a marker op.
- **Tensor-subclass deferral** (`can_handle_tensor` -> `NotImplemented`): our op
  is dispatch-less and never enters `__torch_dispatch__`, so subclass interop
  isn't supported by design.
- **Structured `proxy_args`/`proxy_kwargs`**: we emit FLAT leaves on purpose
  (the flatten-for-BC representation; structure lives in the pytree specs).

### Handled for us anyway
- **Constant tensor inputs.** An untracked (closure/global) tensor passed as an
  op arg is lifted to `self._tensor_constantN` by `create_proxy`/`create_arg`,
  so we don't need `proxy_call`'s constant-input machinery. (Verified.)

### Forgone -- minor, could add if needed
- **Output constant propagation.** `proxy_call` computes `constant=` for small
  all-constant results so a later `.item()` returns the known value; we always
  pass `constant=None`. To add: compute the constant from constant inputs and
  feed it to `track_tensor_tree`.
- **`_enable_thunkify` around execution.** Defers SymInt work; we don't wrap the
  impl in it. Basic symbolic tracing works, but this is the least-covered corner
  (no symbolic-with-`register_fake` correctness test).
- **`data_dependent_output` tag handling.** We don't tag our ops, so the
  error-on-real-data-dependent / constant-prop path doesn't apply; data
  dependence is instead handled by requiring `register_fake` in fake mode.
