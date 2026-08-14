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
.. py:function:: precompile(fn, *example_args, backend="inductor", tracer="make_fx", decompositions=None, example_inputs=None, guard_filter_fn=None, recompile_limit=256, dynamic=None, invariants=None)

   Ahead-of-time precompile ``fn`` using one of two input forms. Positional example
   arguments select the single-graph source-artifact path and return a self-contained,
   runnable Python source string plus an acceleration cache as ``(python_code, cache)``.
   The ``example_inputs`` keyword accepts a sequence of positional-argument tuples,
   captures every graph-break continuation and guarded recompilation exercised by those
   calls, and returns a completed session whose ``save(path)`` writes a package for
   ``precompile.load_package``. ``fn`` is the whole computation, taking the model(s) as
   explicit arguments, e.g. ``lambda model, x: model(x)`` or a training step. The
   ``nn.Module`` arguments have their parameters/buffers lifted to graph inputs, so no
   weights are baked into the artifact -- you pass the model again at runtime to the
   reloaded callable. Reload with ``torch.compiler.precompile.load`` (below).

   .. note::

      With the default ``make_fx`` tracer, capture is non-strict. Control flow is
      specialized to the example inputs, and shapes are static -- each size is baked in.
      The exception is a tensor dim explicitly marked unbacked with
      ``torch._dynamo.decorators.mark_unbacked`` on the inputs before the call (with
      ``make_fx`` this requires the inductor backend; with ``tracer="dynamo"`` either
      backend works); such
      a dim is captured as an unbacked symint, so one artifact serves any runtime size of
      it, and a graph that needs to guard on it fails at capture. Each input's dtype and
      device are specialized too (a runtime mismatch is rejected), and the inductor backend
      additionally specializes on input memory format. See Note [precompile programming
      model] in ``torch/_precompile.py``. ``torch.compiler.precompile`` is distinct from
      ``torch._dynamo.config.caching_precompile`` (a ``torch.compile`` caching mode).

   If ``fn`` runs a backward, the artifact re-runs the whole forward and backward and
   scatters the resulting parameter gradients onto the runtime model's ``parameters()``
   ``.grad`` fields, accumulating (``p.grad += g``) exactly like eager ``.backward()`` --
   so keep your usual ``zero_grad()`` / ``optimizer.step()`` loop. Which params receive a
   grad is fixed at capture time (frozen or non-contributing params stay ``.grad = None``).
   The artifact returns ``fn``'s own result (``None`` for a bare ``.backward()`` step), not
   the gradients.

   :param fn: The whole computation to capture, taking the model(s) and runtime inputs
       as positional arguments.
   :param example_args: Example positional arguments for the single-graph source artifact;
       the ``nn.Module`` arguments are lifted and the rest are the runtime inputs.
   :param example_inputs: Sequence of positional-argument tuples for multi-graph capture.
       Calls run automatically under ``torch.no_grad()``. Do not combine this with
       positional example arguments.
   :param backend: ``"inductor"`` (default) lowers through AOTAutograd + Inductor;
       ``"eager"`` keeps the captured ATen graph (layout-flexible, no kernels; shapes
       are still specialized to the example).
   :param tracer: capture front-end. ``"make_fx"`` (default) is a non-strict make_fx
       trace. ``"dynamo"`` analyzes the Python (bytecode) rather than tracing one path and
       inlines the transformed bytecode Dynamo produces into ``python_code``, lowering the
       compiled subgraph through the same ``backend`` choices; it honors ``mark_unbacked``
       dynamic shapes (on either backend, though ``mark_unbacked(strict=True)`` raises --
       Dynamo captures a strict mark as a guardable backed dim), ``decompositions``, and
       training steps (a ``.backward()`` / ``torch.autograd.grad`` is traced into the graph
       and the parameter gradients are accumulated onto the runtime model like eager). A
       source-artifact path requires one full graph; pass a list of calls through the
       ``example_inputs`` keyword when Python graph-breaks or when several guarded/recompiled
       variants must be retained, or use ``torch.compiler.precompile.capture`` below when
       the calls must be made manually. Unlike ``make_fx``, the dynamo driver
       does NOT re-validate the
       runtime model/inputs, so on the eager backend a drifted model (broken weight tying,
       a retyped/reshaped weight) or a broadcast-compatible input-shape mismatch can
       silently miscompute where ``make_fx`` would raise; pass a model and inputs matching
       the example. The dynamo artifact inlines marshalled bytecode plus a pickled state
       blob, so it is locked to the Python version that produced it AND, because its import
       aliases can reference private ``torch._dynamo`` modules, to a compatible torch build,
       unlike ``make_fx`` source (Python-version portable on either backend; use
       ``backend='eager'`` for portability across torch builds too, since the default
       ``make_fx`` inductor artifact inlines private ``torch._inductor`` modules).
   :param decompositions: Optional decomposition table (``dict`` of ``OpOverload`` to a
       decomposition function) controlling how ATen ops are broken down in the captured
       graph; defaults to ``None``. ``tracer="make_fx"`` forwards it to ``make_fx`` during
       capture; ``tracer="dynamo"`` applies the same table by re-tracing Dynamo's captured
       subgraph with it. ``tracer`` and ``decompositions`` apply only to positional input;
       keyword ``example_inputs`` always selects multi-graph Dynamo capture.
   :param guard_filter_fn: Multi-graph guard filter; returns one boolean per guard entry.
   :param recompile_limit: Maximum multi-graph variants captured per frame; defaults to 256.
   :param dynamic: Multi-graph dynamic-shape policy forwarded to ``torch.compile``.
   :param invariants: Optional path receiving the multi-graph invariant report.
   :returns: For positional input, ``(python_code, cache)`` -- a self-contained Python
       source string and binary acceleration cache. For keyword ``example_inputs``, a
       completed session exposing ``summary()``, ``save()``, and invariant reporting.
   :raises PrecompileError: if capture, lowering, or a runtime call violates the
       contract (see the exception below).

   Example::

       python_code, cache = torch.compiler.precompile(lambda m, x: m(x), model, x)
       f = torch.compiler.precompile.load(python_code, cache)
       out = f(model, x)   # pass the model again at runtime

       session = torch.compiler.precompile(
           model,
           example_inputs=[(example_a,), (example_b,)],
       )
       session.save("model.pt")
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
       paired with a different ``python_code`` (mismatched ``backend`` tag, ``tracer``
       tag, or ``code_hash``), or if a runtime call violates the precompile contract.

.. py:method:: precompile.capture(fn, *, backend="inductor", guard_filter_fn=None, recompile_limit=256, dynamic=None, example_inputs=None, invariants=None)

   Begin an execution-driven multi-graph capture. The yielded callable uses Dynamo and
   records every graph produced by the calls made during the capture block: the entry
   frame, resume continuations after graph breaks, and every guarded recompiled variant.
   Capture every path and specialization the artifact must serve, then call ``save(path)``
   on the returned session after the block exits.

   When every call is known up front, the shorter equivalent is
   ``precompile(fn, example_inputs=[(x1,), (x2,)])``; use ``capture`` when calls must be
   made manually or under a caller-selected grad mode.

   ``example_inputs`` may be a sequence of positional-argument tuples. Those calls run
   automatically under ``torch.no_grad()`` when the capture begins; calls made explicitly
   in the block use the ambient grad mode. ``recompile_limit`` defaults to 256 because an
   ahead-of-time capture intentionally collects variants rather than treating them as a
   runaway recompilation.

   ``guard_filter_fn`` receives a sequence of guard entries and returns one boolean per
   entry, with ``True`` keeping that guard. The default drops identity guards that cannot
   be serialized; a custom filter that keeps one makes capture fail. ``dynamic`` is
   forwarded to ``torch.compile``. ``invariants`` names a report file written after a
   successful capture.

   The session's ``summary()`` reports graph, frame, coverage, and guard information.
   ``save`` refuses incomplete captures by default; see its error and summary for uncovered,
   bypassed, or truncated frames. The callable and source it reaches must be importable on
   the loading host.

   Save with
   ``session.save(path, *, require_complete=True, require_no_risky_drops=False,``
   ``require_no_dropped_guards=False)``. ``require_complete`` rejects missing variants or
   frames. ``require_no_risky_drops`` rejects dropped identity guards on configuration-like
   slots, while ``require_no_dropped_guards`` rejects every unserializable guard. Both
   dropped-guard requirements default to ``False`` because ordinary captures contain
   identity guards; inspect ``summary().dropped_guards`` and
   ``summary().risky_dropped_guards`` before choosing the deployment policy.

   .. warning::

      Capture is by execution, so unexercised paths are absent. Non-tensor values crossing
      a graph break are equality-guarded and may need one captured variant per value.
      Identity guards cannot be serialized and are dropped by default; inspect
      ``summary().dropped_guards`` and ``summary().risky_dropped_guards``, and use the
      corresponding ``save`` requirements when the deployment must reject them. Explicit
      forward-only inference calls in the capture block should run under ``torch.no_grad()``
      or ``torch.inference_mode()``.

   Example::

       session = torch.compiler.precompile.capture(model, backend="inductor")
       with session as compiled:
           compiled(example_a)
           compiled(example_b)  # another guarded/recompiled variant
       session.save("model.pt")

.. py:method:: precompile.load_package(fn, path, *, backend="inductor", guard_filter_fn=None, recompile_limit=256, dynamic=None)

   Load a multi-graph artifact written by ``precompile.capture(...).save(path)`` and
   install its guarded bytecode and compiled backends on ``fn``'s code objects. The
   returned callable is also a context manager; exit it or call ``unload()`` to remove
   the installed entries and globals.

   ``guard_filter_fn``, ``recompile_limit``, and ``dynamic`` configure any uncovered call
   that is allowed to compile outside ``precompile.serving()``. The filter returns one
   boolean per guard entry, with ``True`` keeping that guard.

   Loading mutates process-global compiler state for the affected code objects. Load one
   artifact per callable/class at a time, and treat the artifact file as trusted input;
   ``load_package`` warns before unpickling it.

.. py:method:: precompile.serving()

   Return a context manager that forbids compilation. Use it around calls to a loaded
   multi-graph artifact so an input or path missing from the capture raises instead of
   silently compiling a new variant.

   Example::

       with (
           torch.compiler.precompile.load_package(model, "model.pt") as compiled,
           torch.compiler.precompile.serving(),
       ):
           out = compiled(runtime_input)

.. autoexception:: torch.compiler.PrecompileError
```
