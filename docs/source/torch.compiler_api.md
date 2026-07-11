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

`torch.compiler.precompile` is documented explicitly below rather than in the
autosummary above: it is a callable instance (not a plain function), which Sphinx
autosummary cannot render.

```{eval-rst}
.. py:function:: precompile(fn, *example_inputs, backend="eager", tracer="make_fx", decompositions=None)

   Ahead-of-time precompile ``fn`` against example inputs, returning a self-contained,
   runnable Python source string plus an acceleration cache as ``(python_code, cache)``.
   Capture uses ``make_fx`` (a non-strict trace; a Dynamo-based tracer is planned) and
   the ``eager`` backend keeps the captured ATen graph (an inductor backend is planned).
   ``fn`` is the whole computation, taking the model(s) as explicit arguments, e.g.
   ``lambda model, x: model(x)`` or a training step. The ``nn.Module`` arguments have
   their parameters/buffers lifted to graph inputs, so no weights are baked into the
   artifact -- you pass the model again at runtime to the reloaded callable. Reload with
   ``torch.compiler.precompile.load``. See Note [precompile programming model] in
   ``torch/_precompile.py`` for the full contract.

   Example::

       python_code, cache = torch.compiler.precompile(lambda m, x: m(x), model, x)
       f = torch.compiler.precompile.load(python_code, cache)
       out = f(model, x)   # pass the model again at runtime

.. py:method:: precompile.load(python_code, cache)

   Reconstruct a runnable from the ``(python_code, cache)`` pair returned by
   ``precompile``. The calling convention is read from ``python_code`` (the single
   source of truth); ``cache`` is a pure acceleration. You pass the model(s) again at
   runtime.

.. autoexception:: torch.compiler.PrecompileError
```
