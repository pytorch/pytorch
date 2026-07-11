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
.. py:function:: precompile(fn, *example_inputs, backend="inductor", tracer="make_fx", decompositions=None)

   Ahead-of-time precompile ``fn`` against example inputs, returning a self-contained,
   runnable Python source string plus an acceleration cache as ``(python_code, cache)``.
   Capture uses ``make_fx`` as the default tracer (a Dynamo-based tracer is planned).
   ``fn`` is the whole computation, taking the model(s) as
   explicit arguments, e.g. ``lambda model, x: model(x)`` or a training step. The
   ``nn.Module`` arguments have their parameters/buffers lifted to graph inputs, so no
   weights are baked into the artifact -- you pass the model again at runtime to the
   reloaded callable. Reload with ``torch.compiler.precompile.load`` (below).

   .. note::

      With the default ``make_fx`` tracer, capture is non-strict: control flow is
      specialized to the example inputs, and shapes are static (each size is baked). Each
      input's dtype and device are specialized too (a runtime mismatch is rejected), and
      the inductor backend additionally specializes on input memory format. See Note
      [precompile programming model] in ``torch/_precompile.py``.
      ``torch.compiler.precompile`` is distinct from
      ``torch._dynamo.config.caching_precompile`` (a ``torch.compile`` caching mode).

   :param fn: The whole computation to capture, taking the model(s) and runtime inputs
       as positional arguments.
   :param example_inputs: Example positional arguments to ``fn``; the ``nn.Module``
       arguments are lifted and the rest are the runtime inputs.
   :param backend: ``"inductor"`` (default) lowers through AOTAutograd + Inductor;
       ``"eager"`` keeps the captured ATen graph (layout-flexible, no kernels; shapes
       are still specialized to the example).
   :param tracer: capture front-end. ``"make_fx"`` (default) is a non-strict make_fx
       trace and the only tracer implemented today; ``"dynamo"`` is planned and raises
       ``NotImplementedError`` for now.
   :param decompositions: Optional decomposition table (``dict`` of ``OpOverload`` to a
       decomposition function) forwarded to ``make_fx``; defaults to ``None``.
   :returns: ``(python_code, cache)`` -- a self-contained Python source string (the
       single source of truth for the calling convention) and a binary acceleration
       cache (no weights, no calling-convention metadata; it carries a small
       format/version/backend/code_hash integrity tag that ``load`` verifies).
   :raises PrecompileError: if capture, lowering, or a runtime call violates the
       contract (see the exception below).

   Example::

       python_code, cache = torch.compiler.precompile(lambda m, x: m(x), model, x)
       f = torch.compiler.precompile.load(python_code, cache)
       out = f(model, x)   # pass the model again at runtime
```

```{eval-rst}
.. py:method:: precompile.load(python_code, cache)

   Reconstruct a runnable from the ``(python_code, cache)`` pair returned by
   ``precompile``. The calling convention is read from ``python_code`` (the single
   source of truth); ``cache`` is a pure acceleration carrying only the compiled
   inductor artifact (no weights). You pass the model(s) again at runtime.

   :param python_code: The self-contained Python source string returned by ``precompile``.
   :param cache: The binary acceleration cache returned by ``precompile``.
   :returns: A runnable callable with the same calling convention as the captured ``fn``.
   :raises PrecompileError: if ``cache`` is paired with a different ``python_code`` or a
       runtime call violates the precompile contract.

.. autoexception:: torch.compiler.PrecompileError
```
