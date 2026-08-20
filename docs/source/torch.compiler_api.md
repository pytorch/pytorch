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

% precompile is a callable instance (not a plain function), which Sphinx
% autosummary cannot render, so it is documented manually below and
% intentionally omitted from the autosummary block above.

```{eval-rst}
.. py:function:: precompile(fn, *, example_inputs, backend="inductor", tracer="make_fx", decompositions=None)

   Ahead-of-time precompile ``fn`` against example inputs, returning a self-contained,
   runnable Python source string plus an acceleration cache as ``(python_code, cache)``.
   ``fn`` is the whole computation, taking the model(s) as
   explicit arguments, e.g. ``lambda model, x: model(x)`` or a training step. The
   ``nn.Module`` arguments have their parameters/buffers lifted to graph inputs, so no
   weights are baked into the artifact -- you pass the model again at runtime to the
   reloaded callable. Reload with ``torch.compiler.precompile.load`` (below).

   .. note::

      With the default ``make_fx`` tracer, ``example_inputs`` must contain exactly one
      positional-argument tuple and capture is non-strict. Control flow is specialized
      to that example, and shapes are static -- each size is baked in.
      The exception is a tensor dim explicitly marked unbacked (inductor backend only)
      with ``torch._dynamo.decorators.mark_unbacked`` on the inputs before the call; such
      a dim is captured as an unbacked symint, so one artifact serves any runtime size of
      it, and a graph that needs to guard on it fails at capture. Each input's dtype and
      device are specialized too (a runtime mismatch is rejected), and the inductor backend
      additionally specializes on input memory format. See Note [precompile programming
      model] in ``torch/_precompile.py``. ``torch.compiler.precompile`` is distinct from
      ``torch._dynamo.config.caching_precompile`` (a ``torch.compile`` caching mode).

      With ``tracer="dynamo"``, every tuple in ``example_inputs`` is executed during
      capture. Recompilations become guarded variants in the artifact, including
      automatically dynamic graphs produced when dimensions vary across examples. The
      artifact drops a serialized guard record only when doing so preserves how every
      example matches the captured variants. This filtering is at guard-record
      granularity, so a retained composite record can still contain invariant leaf
      checks. Conditions removed from the dispatch guards become unchecked caller
      assumptions; changing one from all capture examples can silently miscompute. The
      loaded artifact raises only when a call fails every retained guard set, and never
      compiles a new variant. Graph breaks are captured as Dynamo resume frames.
      Closure-free Python functions wrapped with ``torch._dynamo.disable`` are embedded
      and execute eagerly between compiled graph segments. Global names left in
      transformed bytecode must resolve to recursive literal values or independently
      importable objects. Disabled functions cannot assign globals or use
      ``globals()``, ``eval()``, or ``exec()``; their importable module globals are
      rebound at load, while recursive literal globals and defaults are captured by
      value. Compiled graphs and kernels remain Python source; guard
      trees, transformed Dynamo entry/resume bytecode, and embedded disabled-function
      bytecode are stored as opaque inline data because they have no Python-source
      representation. The top-level function cannot have closure cells or nested
      functions that capture its locals, and ``nn.Module`` arguments are not supported
      yet because their identity guards are not serializable.

   With ``tracer="make_fx"``, if ``fn`` runs a backward, the artifact re-runs the whole
   forward and backward and scatters the resulting parameter gradients onto the runtime
   model's ``parameters()`` ``.grad`` fields, accumulating (``p.grad += g``) exactly like
   eager ``.backward()`` -- so keep your usual ``zero_grad()`` / ``optimizer.step()``
   loop. Which params receive a grad is fixed at capture time (frozen or
   non-contributing params stay ``.grad = None``). The artifact returns ``fn``'s own
   result (``None`` for a bare ``.backward()`` step), not the gradients.

   :param fn: The whole computation to capture, taking the model(s) and runtime inputs
       as positional arguments.
   :param example_inputs: A sequence of positional-argument tuples for ``fn``. The
       ``make_fx`` tracer requires exactly one tuple. The ``dynamo`` tracer accepts one
       or more tuples and records the guarded recompilations they exercise. With
       ``make_fx``, ``nn.Module`` arguments within the tuple are lifted and the rest are
       runtime inputs.
   :param backend: ``"inductor"`` (default) lowers through AOTAutograd + Inductor;
       ``"eager"`` keeps the captured ATen graph (layout-flexible, no kernels; shapes
       are still specialized to the example).
   :param tracer: capture front-end. ``"make_fx"`` (default) is a non-strict make_fx
       trace. ``"dynamo"`` captures guarded specializations and recompilations from a
       Python function, including graph-break resume frames; it does not yet support
       top-level closures, nested functions that capture locals, or ``nn.Module``
       arguments.
   :param decompositions: Optional decomposition table (``dict`` of ``OpOverload`` to a
       decomposition function) forwarded to ``make_fx``; defaults to ``None`` and is not
       yet supported with ``tracer="dynamo"``.
   :returns: ``(python_code, cache)`` -- a self-contained Python source string (the
       single source of truth for the calling convention) and a binary acceleration
       cache (no weights, no calling-convention metadata; it carries a small
       format/version/backend/code_hash integrity tag that ``load`` verifies).
   :raises PrecompileError: if capture, lowering, or a runtime call violates the
       contract (see the exception below).

   Example::

       python_code, cache = torch.compiler.precompile(
           lambda m, x: m(x), example_inputs=[(model, x)]
       )
       f = torch.compiler.precompile.load(python_code, cache)
       out = f(model, x)   # pass the model again at runtime

   Dynamo recompilation and automatic dynamic shapes::

       examples = [(torch.randn(2, 4),), (torch.randn(3, 4),)]
       python_code, cache = torch.compiler.precompile(
           fn, example_inputs=examples, tracer="dynamo"
       )
       f = torch.compiler.precompile.load(python_code, cache)
       out = f(torch.randn(7, 4))  # served by the captured dynamic variant
```

```{eval-rst}
.. py:method:: precompile.load(python_code, cache)

   Reconstruct a runnable from the ``(python_code, cache)`` pair returned by
   ``precompile``. The calling convention is read from ``python_code`` (the single
   source of truth); ``cache`` only accelerates loading -- it carries only the compiled
   backend artifact (the Inductor bundle for ``backend="inductor"``; empty for
   ``backend="eager"``) and no weights. You pass the model(s) again at runtime.

   .. warning::

      ``load`` runs the artifact as code: it executes ``python_code`` (via ``exec``) and,
      for the inductor backend, primes the kernel caches from the ``cache``. Treat
      ``(python_code, cache)`` as trusted, executable input -- only load a pair you
      produced yourself or otherwise trust, exactly as you would any code you are about to
      run (see Note [precompile programming model], invariant 7). ``load`` also emits a
      per-call warning before it runs.

   :param python_code: The self-contained Python source string returned by ``precompile``.
   :param cache: The binary acceleration cache returned by ``precompile``.
   :returns: A runnable callable with the same calling convention as the captured ``fn``.
       Arguments are matched positionally at both capture and load time; keyword-argument
       calling conventions are not supported.
   :raises PrecompileError: if ``python_code`` is not a valid precompile artifact (it
       fails to parse or is missing its calling-convention metadata), if ``cache`` is
       paired with a different ``python_code`` (mismatched ``backend`` tag or
       ``code_hash``), or if a runtime call violates the precompile contract.

.. autoexception:: torch.compiler.PrecompileError
```
