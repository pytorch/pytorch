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

```{warning}
`torch.compiler.precompile` and everything reached through it (`precompile.load`,
`torch.compiler.PrecompiledCallable`, and the other objects it returns) is a
prototype API. Signatures, error types and the artifact format may change between
releases without a deprecation cycle.
```

% precompile is a callable instance (not a plain function), which Sphinx
% autosummary cannot render, so it is documented manually below and
% intentionally omitted from the autosummary block above.

```{eval-rst}
.. py:function:: precompile(fn, *, example_inputs, artifact_path, cache_path, backend="inductor", tracer="dynamo", guard_filter_fn=None, recompile_limit=256, dynamic=None, invariants=None, require_complete=True, require_no_risky_drops=True, require_no_dropped_guards=False, training=False, keep_example_grads=False)

   Ahead-of-time precompile ``fn`` against ``example_inputs`` -- a sequence of calls, each
   a tuple of positional arguments -- write the artifact to ``artifact_path`` and
   ``cache_path``, and return what the example calls returned, in order. precompile makes
   those calls itself, for real: a real capture's artifact runs to hundreds of megabytes of
   source, so it goes straight to disk, and the results are how a training capture over
   real batches hands its losses on without a second forward. Reload with
   ``torch.compiler.precompile.load(artifact_path=..., cache_path=...)``.

   This form is ``tracer="dynamo"`` only: a ``make_fx`` trace runs ``fn`` under proxy
   tensors and has no results to give, so it is refused before ``fn`` runs. The pair in
   memory, and the ``make_fx`` tracer, are :meth:`precompile.artifact`, which documents the
   contract both forms share and the remaining parameters -- read it, and
   Note [precompile programming model], before relying on an artifact.

   :param fn: The whole computation to capture, taking the model(s) and runtime inputs
       positionally, exactly as for :meth:`precompile.artifact`.
   :param example_inputs: Required. Sequence of calls to capture; see
       :meth:`precompile.artifact`.
   :param artifact_path: File to write ``python_code`` to. Required, together with
       ``cache_path`` -- the two halves load only as a matched pair, so naming one without
       the other raises. Parent directories are created.
   :param cache_path: File to write ``cache`` to; see ``artifact_path``.
   :param tracer: ``"dynamo"``, and nothing else.
   :returns: What each ``example_inputs`` call returned, in order.
   :raises PrecompileError: as :meth:`precompile.artifact` does.

   Example::

       losses = torch.compiler.precompile(
           train_step,
           training=True,
           example_inputs=[(model, batch) for batch in batches],
           artifact_path="model.py",
           cache_path="model.cache",
       )
       step = torch.compiler.precompile.load(
           artifact_path="model.py", cache_path="model.cache"
       )
```

```{eval-rst}
.. py:method:: precompile.artifact(fn, *, example_inputs, backend="inductor", tracer="make_fx", decompositions=None, guard_filter_fn=None, recompile_limit=256, dynamic=None, invariants=None, require_complete=True, require_no_risky_drops=True, require_no_dropped_guards=False, training=False, keep_example_grads=False)

   Ahead-of-time precompile ``fn`` against ``example_inputs``, a sequence of calls each
   given as a tuple of positional arguments. precompile makes those calls itself and
   produces a self-contained, runnable Python source string plus an acceleration cache as
   ``(python_code, cache)``, returned in memory; :func:`precompile` itself writes the same
   pair to disk instead. ``tracer`` picks the capture front-end: ``"make_fx"`` (the
   default) is one non-strict ATen trace and takes exactly one call, while ``"dynamo"``
   takes as many as you give it and captures every graph-break continuation and guarded
   recompilation those calls exercise, under full runtime guards.
   This is execution-driven coverage, not an
   exhaustive analysis: paths and values that no example executes are absent. ``fn`` is
   the whole computation, taking the model(s) as
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
   :param example_inputs: Required. Sequence of calls to capture, each a tuple of
       positional arguments (or a ``torch.compiler.precompile.ExampleInput`` when keyword
       arguments are needed; ``tracer="make_fx"`` is positional-only). ``tracer="make_fx"``
       requires exactly one; ``tracer="dynamo"`` accepts any number, and what differs
       between them is what the artifact can discriminate on. The ``nn.Module`` arguments
       are lifted and the rest are the runtime inputs. Calls run under ordinary
       ``torch.no_grad()`` unless ``training=True``, even if the caller is in
       ``torch.inference_mode()``; serve the resulting artifact under the same grad mode.
       Inference mode is a distinct guarded state and must be captured manually if needed.
       Tensors created inside inference mode remain inference tensors after that context is
       disabled, so they are rejected; create those inputs outside inference mode.
   :param backend: ``"inductor"`` (default) lowers through AOTAutograd + Inductor;
       ``"eager"`` keeps the captured ATen graph (layout-flexible, no kernels; shapes
       are still specialized to the example).
   :param tracer: capture front-end, defaulting to ``"make_fx"``. ``"make_fx"`` is a
       non-strict make_fx trace of a single call; passing more than one entry in
       ``example_inputs`` raises. ``"dynamo"`` analyzes the Python (bytecode) rather than tracing one path and
       inlines the transformed bytecode Dynamo produces into ``python_code``, lowering the
       compiled subgraphs through the same ``backend`` choices; it honors ``mark_unbacked``
       dynamic shapes (on either backend; ``mark_unbacked(strict=True)`` is read by Dynamo
       as a guardable backed dynamic dim rather than an unbacked one) and training steps
       (with ``training=True``: a ``.backward()`` in ``fn`` graph-breaks and re-runs at
       serve time through the live autograd engine, which accumulates the parameter
       gradients onto the runtime model like eager); ``decompositions`` is rejected.
       ``make_fx`` requires one full graph; use ``tracer="dynamo"`` when Python
       graph-breaks or when several guarded/recompiled variants must be retained.
       The dynamo driver re-evaluates each variant's serialized guards, but unlike
       ``make_fx`` it does not otherwise re-validate the
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
       capture. It applies only to ``tracer="make_fx"``; the dynamo tracer lowers through
       the backend instead and rejects it.
   :param guard_filter_fn: Multi-graph serialization filter; returns one boolean per guard
       entry. It composes with the default filter (which drops only the identity guards
       that cannot be serialized), so it can drop more guards, never fewer. Live capture
       retains all guards so later examples trigger their recompiles. Risky dropped
       guards are rejected by default when saving, and every drop a custom filter adds
       beyond the default's counts as risky.
   :param recompile_limit: Maximum multi-graph variants captured per frame; defaults to 256
       and overrides a lower ambient accumulated-recompile limit for this capture.
   :param dynamic: Multi-graph dynamic-shape policy forwarded to ``torch.compile``.
   :param invariants: Optional path receiving the multi-graph invariant report.
   :param require_complete: Refuse to produce an artifact whose capture was incomplete --
       a call raised, a frame hit ``recompile_limit``, a frame exercised during capture
       produced no guarded code, or no graph was compiled at all (every captured frame
       was empty, as when the whole body sits behind ``torch._dynamo.disable``). Applies
       only to ``tracer="dynamo"``.
   :param require_no_risky_drops: Refuse to produce an artifact that dropped a guard which
       can affect dispatch. Nothing checks such a guard at load, so a different value can
       silently select the wrong graph instead of recompiling. Applies only to
       ``tracer="dynamo"``.
   :param require_no_dropped_guards: Refuse to produce an artifact that dropped ANY guard.
       Off by default and deliberately so: every real model drops the identity guards
       precompile cannot serialize. Applies only to ``tracer="dynamo"``.
   :param training: Run the example calls with grad enabled, for a capture whose ``fn``
       performs a backward.
   :param keep_example_grads: Leave ``.grad`` exactly as the example calls left it.
       By default precompile snapshots and clears the example model's gradients before
       the calls and restores them afterwards, so capturing cannot double the gradients
       of the documented warmup-step-then-capture flow. Pass ``True`` when the example
       call IS your live training step and its gradients are the point -- otherwise the
       backward you just paid for is discarded, and the artifact is produced either way
       so nothing tells you a batch went missing. With it set, a gradient already present
       accumulates exactly as it would in eager. Applies only to ``tracer="dynamo"``.
   :returns: ``(python_code, cache)`` -- a self-contained Python source string and a
       binary acceleration cache.
   :raises PrecompileError: if capture, lowering, or a runtime call violates the
       contract (see the exception below).

   Example::

       python_code, cache = torch.compiler.precompile.artifact(
           lambda m, x: m(x), example_inputs=[(model, x)]
       )
       f = torch.compiler.precompile.load(python_code, cache)
       out = f(model, x)   # pass the model again at runtime

       def staged(x):
           y = x + 1
           scale = y.sum().item()  # a graph break
           return y * scale

       # Graph breaks and several variants need the dynamo tracer; make_fx
       # captures a single call as one graph.
       python_code, cache = torch.compiler.precompile.artifact(
           staged,
           tracer="dynamo",
           example_inputs=[(example_a,), (example_b,)],
       )
       compiled = torch.compiler.precompile.load(python_code, cache)
       # staged() breaks only within its own frame, so this artifact is
       # STANDALONE: a plain callable (an installing artifact -- one whose
       # capture holds frames the entry cannot reach -- supports `with`).
       with torch.no_grad():
           out = compiled(example_a)
```

```{eval-rst}
.. py:method:: precompile.accumulate(fn, *, artifact_path, cache_path, backend="inductor", tracer="dynamo", guard_filter_fn=None, recompile_limit=256, dynamic=None, invariants=None, require_complete=True, require_no_risky_drops=True, require_no_dropped_guards=False, training=False)

   Capture ``fn`` across calls that YOUR loop makes, rewriting the artifact each time.

   :func:`torch.compiler.precompile` makes its example calls itself, back to back, which is
   wrong whenever the calls are not independent -- a training step whose inputs come off a
   queue that the enclosing loop advances cannot be called twice in a row, because the second
   call finds the state the first one consumed. ``accumulate`` inverts that: the caller keeps
   their loop and precompile stops and resumes around each call.

   Each call runs ``fn`` for real, folds whatever graphs and variants it newly exercised into
   the capture, rewrites both files, and returns what ``fn`` returned. A call that exercises
   nothing new adds nothing. There is no finalize step: the two files are a complete, loadable
   artifact for everything captured so far from the first call onwards, so a job that dies
   partway through leaves a working artifact for the batches it did reach.

   Gradients pass straight through -- precompile makes no call of its own here, so there is
   nothing to snapshot and ``keep_example_grads`` does not apply.

   The returned object holds a LIVE compiled region, because that is the only way a later call
   can reuse an earlier one's variants: they are filed under an id that nothing can hand back
   to ``torch._dynamo.optimize``. Use it as a context manager, or call ``close()``, to release
   it; a capture left open holds it, and every variant compiled into it, for the life of the
   process.

   A call whose ``fn`` raises propagates that error and leaves the capture open for the next
   call. A refusal from the artifact gates (``require_*``) is different and final: the step has
   already run by then, so the ``PrecompileError`` carries what it returned as ``result``, the
   capture closes -- the files keep the last artifact that passed -- and every later call
   raises without running. Calls are serialized: a second thread calling the capture waits for
   the first call to finish, artifact rewrite included.

   :param fn: The whole computation to capture, taking the model(s) and runtime inputs
       positionally, exactly as :func:`torch.compiler.precompile` does.
   :param artifact_path: File to write ``python_code`` to, rewritten on every call. Required.
   :param cache_path: File to write the acceleration cache to. Required.
   :param tracer: ``"dynamo"``, and nothing else -- a make_fx trace is a single graph of a
       single call and has nothing to accumulate.
   :returns: A ``precompile.AccumulatingCapture``. Call it like ``fn``; it also exposes
       ``summary()`` (coverage, recompilation, failure and guard information for everything
       captured so far), ``invariants()`` (the guards that held across every captured variant
       of each frame), ``calls()`` (how many calls have been folded in) and ``close()`` (give
       back the compiled region; the files are unaffected, and closing twice is a no-op).
   :raises PrecompileError: as :func:`torch.compiler.precompile` does, on the call that
       violates the contract.

   Example::

       with torch.compiler.precompile.accumulate(
           train_step, artifact_path="m.py", cache_path="m.cache",
           training=True, require_no_risky_drops=False,
       ) as capture:
           for batch in loader:
               losses = capture(model, batch)   # runs for real, returns its result
               optimizer.step()

   .. note::

      Rewriting is proportional to everything captured so far, not to the call, so a long loop
      over a large model pays it every time. Capture the batches that add variants rather than
      all of them.
```

```{eval-rst}
.. py:method:: precompile.load(python_code=None, cache=None, *, artifact_path=None, cache_path=None, fn=None)

   Reconstruct a runnable from the ``(python_code, cache)`` pair produced by
   ``precompile.artifact`` -- passed in memory -- or named as the two files ``precompile``
   wrote, with ``load(artifact_path=..., cache_path=...)``. The two forms are exclusive and each needs
   both halves. The calling convention is read from ``python_code`` (the single
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

   :param python_code: The self-contained Python source string produced by ``precompile``.
   :param cache: The binary acceleration cache produced by ``precompile``.
   :param artifact_path: File holding ``python_code``, as written by ``precompile``'s
       ``artifact_path``. Pass it together with ``cache_path``.
   :param cache_path: File holding ``cache``; see ``artifact_path``.
   :param fn: The original callable, for an artifact that installs onto live code objects
       rather than dispatching its own entry. Supply it before the first call; a
       standalone artifact rejects ``fn=`` with ``PrecompileError``.
   :returns: A runnable callable with the same calling convention as the captured ``fn``.
       Arguments are matched positionally at both capture and load time; keyword-argument
       calling conventions are not supported. A dynamo artifact with captured frames the
       entry bytecode cannot reach on its own -- for example a graph break inside a child
       module's frame -- serves by INSTALLING onto the captured code objects: the returned
       callable mutates process state on first call (or on ``__enter__``) and supports
       ``with`` / ``unload()`` to take that back out. An artifact whose frames are all
       reachable from the entry -- including one that graph-broke or recompiled only
       within the entry frame -- is standalone: a plain callable, no installation and no
       ``unload``. Which one you get is a property of the capture, not a load-time choice.
   :raises PrecompileError: if ``python_code`` is not a valid precompile artifact (it
       fails to parse or is missing its calling-convention metadata), if ``cache`` is
       paired with a different ``python_code`` (mismatched ``backend`` tag, ``tracer``
       tag, or ``code_hash``), or if a runtime call violates the precompile contract.

.. py:class:: precompile.ExampleInput(args=(), kwargs={})

   One capture call for ``example_inputs`` when positional arguments alone are not
   enough (``tracer="dynamo"`` only). A plain tuple in ``example_inputs`` is the
   positional arguments of one
   call; wrap a call that needs keyword arguments in this instead::

       torch.compiler.precompile.artifact(
           fn,
           example_inputs=[
               (x,),
               torch.compiler.precompile.ExampleInput(args=(x,), kwargs={"scale": 2}),
           ],
       )

.. autoexception:: torch.compiler.PrecompileError
   :members: result

.. autoclass:: torch.compiler.PrecompiledCallable
   :members: unload, serve_time_compiles

   Returned by :func:`precompile.load` for an artifact that serves by installing,
   and used as a callable or context manager; it is not constructed directly.


```