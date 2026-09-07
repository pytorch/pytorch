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
`precompile.load`, `torch.compiler.PrecompiledRunnable`,
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
   intermediate values are needed; to checkpoint the artifact partway instead of only at
   exit, call ``cap.save()`` inside the block, which re-renders and rewrites both files.
   ``tracer`` picks the capture front-end and carries its tracer-specific
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
.. py:function:: precompile.load(artifact_path, cache_path, *, fn=None)

   Reconstruct a runnable from the two files a precompile capture wrote -- the
   ``python_code`` artifact and its ``cache``. They load only as a matched pair (the cache
   carries a sha256 of exactly the ``python_code`` bytes it was emitted with). The calling
   convention is read from ``python_code`` (the single source of truth); ``cache`` only
   accelerates loading -- it carries only the compiled backend artifact (the Inductor bundle
   for ``backend="inductor"``; empty for ``backend="eager"``) and no weights. You pass the
   model(s) again at runtime.

   .. warning::

      ``load`` runs the artifact as code: it executes ``python_code`` (via ``exec``),
      reads the ``cache`` envelope (a ``weights_only`` load) and, for the inductor
      backend, writes its bundle into the compile caches. A crafted ``python_code`` runs
      whatever it contains, and a crafted bundle plants pickles that a later cache hit
      unpickles. Treat the two files as trusted, executable input -- only load an artifact
      you produced yourself or otherwise trust, exactly as you would any code you are about
      to run (see Note [precompile programming model], invariant 7). ``load`` also emits a
      per-call warning before it runs.

   :param artifact_path: File holding ``python_code``, as written by ``precompile``.
   :param cache_path: File holding ``cache``, as written by ``precompile``.
   :param fn: For a dynamo artifact that serves by installing onto live code objects,
       the function object to install onto, when it is not importable from where it was
       captured (e.g. defined in ``__main__`` or a notebook); pass it before the first
       call. A standalone artifact rejects ``fn=`` with ``PrecompileError``.
   :returns: A :class:`torch.compiler.PrecompiledRunnable` with the same calling
       convention as the captured ``fn``. A make_fx artifact takes positional arguments
       only; a dynamo artifact also accepts keyword arguments, the way the
       capture calls passed them. A dynamo artifact with captured frames the
       entry bytecode cannot reach on its own -- for example a graph break inside a child
       module's frame -- serves by INSTALLING onto the captured code objects: the returned
       callable mutates process state on first call (or on ``__enter__``) and supports
       ``with`` / ``unload()`` to take that back out. An artifact whose frames are all
       reachable from the entry -- including one that graph-broke or recompiled only
       within the entry frame -- is standalone: it installs nothing, and its ``with`` /
       ``unload()`` are no-ops. Both expose the same surface, and ``installed`` (``True``
       for the installing shape, ``False`` for standalone) tells them apart. Which one
       you get is a property of the capture, not a load-time choice.
   :raises PrecompileError: if ``python_code`` is not a valid precompile artifact (it
       fails to parse or is missing its calling-convention metadata), if ``cache`` is
       paired with a different ``python_code`` (mismatched ``backend`` tag, ``tracer``
       tag, or ``code_hash``), or if a runtime call violates the precompile contract.

.. autoexception:: torch.compiler.PrecompileError
   :members: result

.. autoclass:: torch.compiler.PrecompiledRunnable
   :members: unload

   Every object :func:`precompile.load` returns is one of these, whichever of the
   two shapes below the capture produced, so ``isinstance(loaded,
   torch.compiler.PrecompiledRunnable)`` holds for both.

.. autoclass:: torch.compiler.PrecompiledCallable
   :members: unload, serve_time_compiles

   Returned by :func:`precompile.load` for an artifact that serves by installing,
   and used as a callable or context manager; it is not constructed directly.

.. py:class:: precompile.MakeFxTracer(decompositions=None)

   The ``make_fx`` capture front-end, passed as ``tracer=`` to
   :func:`precompile.capture`. A NON-STRICT single make_fx trace: it records the ATen ops
   of ONE execution of ``fn``, so a capture with this tracer takes exactly one call and
   refuses a second, and control flow and shapes are specialized to that call. Frozen
   dataclass.

   :param decompositions: Optional decomposition table (``dict`` of ``OpOverload`` to a
       decomposition function) forwarded to ``make_fx`` as its ``decomposition_table``;
       specific to this tracer (Dynamo lowers through the backend instead). Defaults to
       ``None``.

.. py:class:: precompile.DynamoTracer(guard_filter_fn=None, recompile_limit=256, dynamic=None, invariants=None, require_complete=True, require_no_risky_drops=True, require_no_dropped_guards=False)

   The ``dynamo`` capture front-end (the default), passed as ``tracer=`` to
   :func:`precompile.capture`. An execution-driven
   multi-graph capture that analyzes the Python (bytecode) rather than tracing one path: it
   records graph-break continuations and every guarded recompilation the calls exercise, so
   a capture with this tracer takes as many calls as you make. The dynamo driver re-evaluates
   each variant's serialized guards, but unlike make_fx it does not otherwise re-validate the
   runtime model/inputs, so on the eager backend a drifted model or a broadcast-compatible
   input-shape mismatch can silently miscompute where make_fx would raise; pass a model and
   inputs matching the captured call. The dynamo artifact inlines marshalled bytecode plus a
   pickled state blob, so it is locked to the Python version that produced it and to a
   compatible torch build, unlike make_fx source. Frozen dataclass.

   :param guard_filter_fn: Multi-graph serialization filter; returns one boolean per guard
       entry. It composes with the default filter (which drops only the identity guards
       that cannot be serialized), so it can drop more guards, never fewer. Live capture
       retains all guards so later calls trigger their recompiles. Risky dropped guards are
       rejected by default when saving, and every drop a custom filter adds beyond the
       default's counts as risky.
   :param recompile_limit: Maximum multi-graph variants captured per frame; defaults to 256
       and overrides a lower ambient accumulated-recompile limit for this capture.
   :param dynamic: Multi-graph dynamic-shape policy forwarded to ``torch.compile``.
   :param invariants: Optional path receiving the multi-graph invariant report.
   :param require_complete: defaults to ``True``. Refuse to produce an artifact whose
       capture summary is not complete -- a frame that produced no guarded code, hit the
       recompile limit, or was bypassed, or a capture that compiled no graph at all.
   :param require_no_risky_drops: defaults to ``True``. Refuse to produce an artifact that
       dropped a guard whose loss could change the answer (every drop made by a custom
       ``guard_filter_fn`` counts as risky).
   :param require_no_dropped_guards: defaults to ``False``. Refuse to produce an artifact
       that dropped any guard at all. Off by default because every model drops identity
       guards that cannot be serialized.

.. py:class:: precompile.Capture

   The object :func:`precompile.capture` returns. Enter it as a context manager and call
   it like ``fn`` inside the block to fold each call into the capture (see
   :func:`precompile.capture` for the semantics); it is not constructed directly. The
   artifact is written to the two files when the block exits. Also exposes:

   .. py:method:: save()

      Checkpoint everything captured so far to the two files without ending the capture.
      Call it as often as you like inside the block; each call re-renders and rewrites both
      files, so a job that dies between saves leaves the last checkpoint loadable. A gate
      refusal (the ``DynamoTracer`` ``require_*`` fields) or a write failure raises but
      writes nothing partial: the previous files stay intact and the capture stays open.

   .. py:method:: summary()

      A :class:`precompile.PrecompileSummary` for everything captured so far. Dynamo capture
      only.

   .. py:method:: invariants()

      A tuple of :class:`precompile.FrameInvariants`, one per captured frame -- the guards that held
      across every captured variant of each frame. Dynamo capture only.

   .. py:method:: calls()

      How many calls have been folded into the capture. Dynamo capture only.

.. py:class:: precompile.PrecompileSummary

   Coverage and guard information from a capture, returned by
   :meth:`precompile.Capture.summary`. Frozen dataclass; ``str(summary)`` renders a
   one-line digest and :attr:`complete` says whether the capture covers everything it
   exercised.

   .. py:attribute:: frames
   .. py:attribute:: resume_functions
   .. py:attribute:: guarded_codes
   .. py:attribute:: backend_graphs

      Counts of captured frames, graph-break continuations, guarded code objects, and
      backend graphs.

   .. py:attribute:: bypassed
   .. py:attribute:: truncated
   .. py:attribute:: uncovered_frames
   .. py:attribute:: wont_generalize

      Frames that fell back to eager, hit the recompile limit, were never reached, or
      carry value-pinned guards that will not generalize.

   .. py:attribute:: dropped_guards
   .. py:attribute:: kept_guards
   .. py:attribute:: risky_dropped_guards
   .. py:attribute:: policy_dropped_guards

      ``(guard_type, source)`` pairs for guards the artifact omitted (could not serialize),
      kept, omitted riskily, or dropped by the invariance policy though serializable.

   .. py:attribute:: dropped_guard_code

      ``(guard_type, source, rendered_check)`` for each dropped slot that renders to a
      check. The slot's ``(guard_type, source)`` alone can be ambiguous -- a dropped
      ``HASATTR`` may be the benign companion of a kept ``TENSOR_MATCH`` or the only thing
      guarding an optional attribute -- so the rendered check is reported alongside to tell
      them apart.

   .. py:attribute:: capture_errors

      Messages from capture calls that raised.

   .. py:property:: complete

      Whether the capture covers everything it exercised: false if any frame produced no
      guarded code, hit the recompile limit, was bypassed, or a capture call raised.

.. py:class:: precompile.FrameInvariants

   Per-frame guard classification, returned by ``invariants()``. Frozen dataclass with the
   frame's name, ``filename``, ``lineno``, the number of ``variants`` seen, and three tuples
   of :class:`precompile.GuardFact`: ``invariant`` (held identically across every variant), ``varying``
   (differed between variants), and ``undetermined`` (a single variant could not decide).

.. py:class:: precompile.GuardFact

   One guard observed while compiling a frame variant. Frozen dataclass with ``guard_type``,
   ``source``, ``code`` (the rendered check parts), ``value``, and ``enforced`` (whether the
   artifact still checks it). ``render()`` returns one stable, human-readable line.


```
