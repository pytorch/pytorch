```{eval-rst}
.. currentmodule:: torch.compiler
.. automodule:: torch.compiler
```

```{eval-rst}
.. py:module:: torch.compiler.precompile
.. currentmodule:: torch.compiler
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

```{warning}
`torch.compiler.precompile` and everything reached through it (`precompile.capture`,
`precompile.accumulate`, `precompile.load`, `torch.compiler.PrecompiledRunnable`,
`torch.compiler.PrecompiledCallable`, and the objects they return) is a prototype API.
Signatures, error types and the artifact format may change between releases without a
deprecation cycle.
```

% precompile is a module whose members are documented manually below.

```{eval-rst}
.. py:function:: precompile.capture(fn, *, artifact_path, cache_path, tracer=DynamoTracer(), backend="inductor", training=False)

   Return a caller-driven capture of ``fn`` as a :class:`precompile.Capture`. Capture is
   caller-driven: this runs nothing on its own. Enter the returned object as a context
   manager and call it exactly as you would ``fn`` inside the block -- each call runs for
   real, folds what it exercised into the capture, and returns what ``fn`` returned -- and
   the ``(python_code, cache)`` artifact is written to ``artifact_path`` / ``cache_path``
   when the block exits::

       with torch.compiler.precompile.capture(
           fn, artifact_path="m.py", cache_path="m.cache"
       ) as cap:
           y1 = cap(model, x1)
           y2 = cap(model, x2)
       f = torch.compiler.precompile.load("m.py", "m.cache")

   Because the caller makes the calls, inputs flow through naturally and return values stay
   available, so the capture drops into an ordinary training or pipeline loop where
   intermediate values are needed; to rewrite the artifact after every call instead of once
   at exit, use :func:`precompile.accumulate`, the per-call-rewrite counterpart with the
   same model. ``tracer`` picks the capture front-end and carries its tracer-specific
   configuration: :class:`precompile.DynamoTracer` (the default) takes as many calls as you
   give it and captures every graph-break continuation and guarded recompilation those
   calls exercise; :class:`precompile.MakeFxTracer` is one non-strict ATen trace and takes
   exactly one call (a second call raises). ``backend`` and ``training`` are shared across
   both tracers. This is execution-driven coverage, not an exhaustive analysis: paths and
   values that no call executes are absent. ``fn`` is the whole computation, taking the
   model(s) as explicit arguments, e.g. ``lambda model, x: model(x)`` or a training step.
   The ``nn.Module`` arguments have their parameters/buffers lifted to graph inputs, so no
   weights are baked into the artifact -- you pass the model again at runtime to the
   reloaded callable. Reload with ``torch.compiler.precompile.load`` (below).

   .. note::

      With :class:`precompile.MakeFxTracer`, capture is non-strict. Control flow is
      specialized to the captured call, and shapes are static -- each size is baked in. A
      data-dependent op (``.item()``, a branch over a tensor value) instead raises at
      capture, since the trace runs under fake mode where the value is unknown. The
      exception to static shapes is a tensor dim explicitly marked unbacked with
      ``torch._dynamo.decorators.mark_unbacked`` on the inputs before the call (with
      ``make_fx`` this requires the inductor backend; with :class:`precompile.DynamoTracer`
      either backend works); such a dim is captured as an unbacked symint, so one artifact
      serves any runtime size of it, and a graph that needs to guard on it fails at capture.
      Each input's dtype and device are specialized too (a runtime mismatch is rejected),
      and the inductor backend additionally specializes on input memory format. See Note
      [precompile programming model] in ``torch/_precompile.py``. ``torch.compiler.precompile``
      is distinct from ``torch._dynamo.config.caching_precompile`` (a ``torch.compile``
      caching mode).

   Gradients and return values keep their normal eager/``torch.compile`` semantics: your
   calls run in whatever grad mode you set, and precompile does not snapshot or clear the
   model's gradients -- there is no example call of its own to compensate for. If ``fn``
   runs a backward (pass ``training=True``), the artifact re-runs the whole forward and
   backward and scatters the resulting parameter gradients onto the runtime model's
   ``parameters()`` ``.grad`` fields, accumulating (``p.grad += g``) exactly like eager
   ``.backward()`` -- so keep your usual ``zero_grad()`` / ``optimizer.step()`` loop. Which
   params receive a grad is fixed at capture time (frozen or non-contributing params stay
   ``.grad = None``). The artifact returns ``fn``'s own result (``None`` for a bare
   ``.backward()`` step), not the gradients.

   :param fn: The whole computation to capture, taking the model(s) and runtime inputs
       as positional arguments. With :class:`precompile.DynamoTracer`, ``cap(...)`` also
       accepts keyword arguments and the loaded artifact takes them the same way;
       :class:`precompile.MakeFxTracer` is positional-only. Enter the returned capture and
       call it once (make_fx) or as many times as you need (dynamo). The ``nn.Module``
       arguments are lifted and the rest are the runtime inputs. Calls run in the caller's
       grad mode; serve the resulting artifact under the same one.
   :param artifact_path: File to write ``python_code`` to when the block exits. Required.
   :param cache_path: File to write the acceleration cache to. Required.
   :param tracer: The capture front-end and its configuration, a
       :class:`precompile.DynamoTracer` (default) or :class:`precompile.MakeFxTracer`.
   :param backend: ``"inductor"`` (default) lowers through AOTAutograd + Inductor;
       ``"eager"`` keeps the captured ATen graph (layout-flexible, no kernels; shapes
       are still specialized to the captured call).
   :param training: Run with grad enabled and lower a backward into the
       artifact; defaults to ``False``. Required for a ``fn`` that runs a backward. The
       caller still controls the grad mode of the calls it makes; this only asks the
       capture to keep a backward.
   :returns: A :class:`precompile.Capture` -- a context manager and callable. The artifact
       is written to the two files when the block exits.
   :raises PrecompileError: if capture, lowering, or a runtime call violates the
       contract (see the exception below); a second make_fx call also raises.
   :raises ValueError: for an unknown ``backend``, or a ``cache_path``/``artifact_path``
       given without the other.
   :raises TypeError: if ``tracer`` is not a :class:`precompile.MakeFxTracer` or
       :class:`precompile.DynamoTracer`.

   Example::

       with torch.compiler.precompile.capture(
           lambda m, x: m(x), artifact_path="m.py", cache_path="m.cache",
           tracer=torch.compiler.precompile.MakeFxTracer(),
       ) as cap:
           y = cap(model, x)   # runs for real, returns m(x)
       f = torch.compiler.precompile.load("m.py", "m.cache")
       out = f(model, x)   # pass the model again at runtime

       def staged(x):
           y = x + 1
           scale = y.sum().item()  # a graph break
           return y * scale

       # Graph breaks and several variants need the dynamo tracer (the default);
       # make_fx captures a single call as one graph.
       with torch.compiler.precompile.capture(
           staged, artifact_path="s.py", cache_path="s.cache"
       ) as cap:
           cap(example_a)
           cap(example_b)
       compiled = torch.compiler.precompile.load("s.py", "s.cache")
       # staged() breaks only within its own frame, so this artifact is
       # STANDALONE: a plain callable (an installing artifact -- one whose
       # capture holds frames the entry cannot reach -- supports `with`).
       with torch.no_grad():
           out = compiled(example_a)
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
