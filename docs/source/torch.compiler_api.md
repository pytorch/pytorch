```{eval-rst}
.. currentmodule:: torch.compiler
.. automodule:: torch.compiler
```

(torch.compiler_api)=
# torch.compiler API reference

For a quick overview of `torch.compiler`, see {ref}`torch.compiler_overview`.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

     compile
     reset
     nonstrict_trace
     allow_in_graph
     substitute_in_graph
     assume_constant_result
     list_backends
     disable
     set_default_backend
     get_default_backend
     set_stance
     set_enable_guard_collectives
     cudagraph_mark_step_begin
     cudagraph_mark_warmup_incomplete
     is_compiling
     is_dynamo_compiling
     is_exporting
     keep_portable_guards_unsafe
     skip_guard_on_inbuilt_nn_modules_unsafe
     skip_guard_on_all_nn_modules_unsafe
     keep_tensor_guards_unsafe
     skip_guard_on_globals_unsafe
     skip_all_guards_unsafe
     nested_compile_region
     load_cache_artifacts
     load_compiled_function
     save_cache_artifacts
     wrap_numpy
```

## torch.compiler.precompile

```{eval-rst}
.. py:module:: torch.compiler.precompile
.. currentmodule:: torch.compiler
```

```{warning}
`torch.compiler.precompile` and everything reached through it is a prototype API.
Signatures, error types and the artifact format may change between releases without a
deprecation cycle.
```

`torch.compiler.precompile` captures a whole computation -- `fn(model, x)`, with the
model(s) passed as arguments -- ahead of time from the caller's own calls, and lowers it to
a self-contained Python source artifact plus an acceleration cache that a fresh process
reloads. No weights are baked in, so the model is passed again at runtime. The contract is
Note [precompile programming model] in `torch/_precompile.py`. Documented here are the
module's `load` and its public types: the tracer configuration a capture takes, the
capture handle it returns, the runnable a load returns, and the coverage and guard
report it produces. It is distinct from
`torch._dynamo.config.caching_precompile` (a `torch.compile` caching mode).

% Rendered from the docstrings, so this reference cannot drift from the source.

```{eval-rst}
.. autofunction:: torch.compiler.precompile.load

.. autoexception:: torch.compiler.PrecompileError

.. autoclass:: torch.compiler.precompile.MakeFxTracer

.. autoclass:: torch.compiler.precompile.Capture

.. autoclass:: torch.compiler.precompile.PrecompiledRunnable
   :members: unload

.. autoclass:: torch.compiler.precompile.PrecompileSummary
   :members: complete, dropped_guard_types, kept_guard_types
```
