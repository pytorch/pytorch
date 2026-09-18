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
.. py:function:: precompile.capture(fn, /, *, artifact_path, cache_path, tracer=MakeFxTracer(), backend="inductor", training=False)

   Return a caller-driven capture of ``fn`` as a :class:`precompile.Capture`: this runs
   nothing on its own. Enter the returned object as a context manager and call it with the
   positional arguments ``fn`` takes inside the block -- each call runs for real, folds
   what it exercised into the capture, and returns the served
   result (a computed output does not require grad; an output that IS an input or
   parameter comes back as that same tensor, hence with its own ``requires_grad``, while
   an output that ALIASES an input is rebuilt at serve time and its ``requires_grad`` is
   not part of the contract -- call ``.requires_grad_()`` on it if you depend on it) --
   and the ``(python_code, cache)`` artifact is written to ``artifact_path`` /
   ``cache_path`` when the block exits cleanly having captured at least one call::

       with torch.compiler.precompile.capture(
           fn, artifact_path="m.py", cache_path="m.cache"
       ) as cap:
           y = cap(model, x)
       f = torch.compiler.precompile.load("m.py", "m.cache")

   Because the caller makes the calls, inputs flow through naturally and return values stay
   available, so the capture drops into an ordinary training or pipeline loop where
   intermediate values are needed; :meth:`precompile.Capture.save` writes the artifact
   from inside the block without ending the capture.
   ``tracer`` picks the capture front-end and carries its tracer-specific
   configuration: :class:`precompile.MakeFxTracer` (the default) is one non-strict ATen
   trace and takes exactly one call (a second call raises); :class:`precompile.DynamoTracer`
   takes as many calls as you give it and captures every graph-break continuation and
   guarded recompilation those calls exercise, and is not available in this build yet
   (``capture`` raises ``PrecompileError`` for it). ``backend`` and ``training`` are shared
   across both tracers. This is execution-driven coverage, not an exhaustive analysis: paths and
   values that no call executes are absent. ``fn`` is the whole computation, taking the
   model(s) as explicit arguments, e.g. ``lambda model, x: model(x)`` or a training step.
   The ``nn.Module`` arguments have their parameters/buffers lifted to graph inputs, so no
   weights are baked into the artifact -- you pass the model again at runtime to the
   reloaded callable. Reload with ``torch.compiler.precompile.load`` (below).

   .. note::

      With :class:`precompile.MakeFxTracer`, capture is non-strict and traces ``fn`` on
      FAKE tensors. Python control flow is specialized to the captured call, and shapes
      are static -- each size is baked in; a control-flow HOP (``torch.cond`` /
      ``torch.while_loop``) is refused rather than specialized. Tracing on fakes also
      refuses, on BOTH capture paths, an op with no meta/fake kernel, a read of a traced
      tensor's data (``.data_ptr()``, ``.numpy()``), an example input a fake tensor cannot
      represent (quantized) or whose metadata it silently drops (pinned, mkldnn, sparse),
      and a nested one; and, on a STATIC capture, a data-dependent op (``.item()``,
      ``.nonzero()``, a Python branch over a tensor value). It also refuses to run inside
      another trace, whose fake mode would outrank its own.
      The exception to static shapes is a tensor dim explicitly marked unbacked with
      ``torch._dynamo.decorators.mark_unbacked`` on the inputs before the call (with
      ``make_fx`` this requires the inductor backend; with
      :class:`precompile.DynamoTracer` either backend works); such a dim is captured as an
      unbacked symint, so one artifact serves any runtime size of it, and a graph that
      needs to guard on it fails at capture. Dims that MUST be equal at runtime (two
      inputs a broadcast requires to match, e.g. ``model(a) + model(b)``) need a SHARED
      ``mark_unbacked`` ``shape_id``, which makes them one symbol: marked independently
      they bake a SILENT equal-size assumption, since the capture records that equality
      only as a deferred runtime assert and precompile does not harvest those yet, so a
      runtime mismatch does not raise the way eager does. Such an
      unbacked capture also holds a data-dependent value symbolically -- it can capture an
      ``.item()`` result or a ``.nonzero()``-sized intermediate that a static one refuses,
      and fails if the computation must guard on that value (or if the op is one no
      ``ShapeEnv`` can fake, e.g. ``aten.equal``). Each input's dtype and device are
      specialized too (a runtime mismatch is rejected), and the inductor backend
      additionally specializes on input memory format. A call served from the reloaded
      artifact also IGNORES the serving process's ambient ``torch.autocast``: whatever
      the capture ran under is already baked in, so autocast is neutralized for the
      duration of the call on every device this build can autocast, and the call returns
      the capture's dtypes, not the dtypes the same eager call returns inside that region
      -- so capture under the autocast you want baked in. The one case that still casts
      twice is a device that reports autocast enabled and whose disable then refuses to
      construct (a module registered under the privateuse1 backend name and missing
      ``get_amp_supported_dtype``); it is skipped with one logged warning per device per
      loaded artifact. See Note [precompile programming model] in
      ``torch/_precompile.py``. ``torch.compiler.precompile`` is distinct from
      ``torch._dynamo.config.caching_precompile`` (a ``torch.compile`` caching mode).

   Gradients and return values keep their normal eager/``torch.compile`` semantics:
   precompile snapshots and clears the example tensors' ``.grad`` around the trace, so that
   a live ``.grad`` is not baked into the trace as a constant, and restores the same objects
   afterwards -- your gradients survive a capture unchanged, but ``fn`` must not read a
   pre-existing ``.grad`` during the captured call. ``training`` selects the grad mode of
   the whole call, trace and serve alike, whatever mode is ambient at the call site: a
   ``training=False`` call runs under ``torch.no_grad()``, so nothing it computes carries a
   backward even inside a ``torch.enable_grad()`` block, and a ``training=True`` call
   runs under ``torch.enable_grad()``, so it carries its backward even inside a
   ``torch.no_grad()`` block. If ``fn`` runs a backward (pass ``training=True``), the
   artifact re-runs the whole forward and backward and scatters the resulting parameter
   gradients onto the runtime model's ``parameters()`` ``.grad`` fields, accumulating
   (``p.grad += g``) exactly like eager ``.backward()`` -- so keep your usual
   ``zero_grad()`` / ``optimizer.step()`` loop. Which params receive a grad is fixed at
   capture time (frozen or non-contributing params stay ``.grad = None``). The artifact
   returns ``fn``'s own result (``None`` for a bare ``.backward()`` step), not the
   gradients.

   Threading: the inductor lowering step drives process-global compiler state and is
   serialized by an internal lock, so concurrent ``backend="inductor"`` captures lower
   one at a time. The capture phase and the ``backend="eager"`` path are NOT
   serialized. The lock is taken inside the ``cap(...)`` calls, which is where the
   lowering runs, not held for the surrounding block.

   :param fn: The whole computation to capture, taking the model(s) and runtime inputs
       as positional arguments. With :class:`precompile.DynamoTracer`, ``cap(...)`` also
       accepts keyword arguments and the loaded artifact takes them the same way;
       :class:`precompile.MakeFxTracer` is positional-only. The ``nn.Module`` arguments
       are lifted and the rest are the runtime inputs.
   :param artifact_path: File to write ``python_code`` to when the block exits. Required.
   :param cache_path: File to write the acceleration cache to. Required.
   :param tracer: The capture front-end and its configuration, a
       :class:`precompile.MakeFxTracer` (default, and the only one that captures in this
       build) or :class:`precompile.DynamoTracer`.
   :param backend: ``"inductor"`` (default) lowers through AOTAutograd + Inductor;
       ``"eager"`` keeps the captured ATen graph (layout-flexible, no kernels; shapes
       are still specialized to the captured call).
   :param training: Run with grad enabled and lower a backward into the artifact;
       defaults to ``False``. Required for a ``fn`` that runs a backward.
   :returns: A :class:`precompile.Capture` -- a context manager and callable. The artifact
       is written to the two files on a clean exit from the block that captured at least
       one call. Three exits write nothing: a block that raised leaves the files untouched
       (the exception propagates), and a clean exit with nothing captured -- no call was
       made, or the only call raised and was caught -- raises ``PrecompileError`` instead
       of writing an empty artifact.
   :raises PrecompileError: if capture, lowering, or a runtime call violates the
       contract (see the exception below); if one of the two paths is handed artifact
       contents rather than a path; if ``tracer`` is a
       :class:`precompile.DynamoTracer` (not available in this build yet); if the block
       exits cleanly with nothing captured (no call was made, or the only call raised),
       so there is nothing to write; a second make_fx call also raises.
   :raises ValueError: for an unknown ``backend``, for one file named as both halves, or
       for a path that exists but is not a regular file.
   :raises TypeError: if ``tracer`` is not a :class:`precompile.MakeFxTracer` or
       :class:`precompile.DynamoTracer`, or if a ``MakeFxTracer`` capture is called with
       keyword arguments.
   :raises NotImplementedError: for ``backend="eager"`` with a ``mark_unbacked`` input
       (dynamic shapes need the inductor backend).

   Example::

       with torch.compiler.precompile.capture(
           lambda m, x: m(x), artifact_path="m.py", cache_path="m.cache",
           tracer=torch.compiler.precompile.MakeFxTracer(),
       ) as cap:
           y = cap(model, x)   # runs for real, returns the served result
       f = torch.compiler.precompile.load("m.py", "m.cache")
       out = f(model, x)   # pass the model again at runtime

       def staged(x):
           y = x + 1
           scale = y.sum().item()  # a graph break
           return y * scale

       # NOT RUNNABLE IN THIS BUILD: graph breaks and several variants need the dynamo
       # tracer, and passing one raises PrecompileError here; shown for the shape it
       # will take (make_fx captures a single call as one graph).
       with torch.compiler.precompile.capture(
           staged, artifact_path="s.py", cache_path="s.cache",
           tracer=torch.compiler.precompile.DynamoTracer(),
       ) as cap:
           cap(example_a)
           cap(example_b)
       compiled = torch.compiler.precompile.load("s.py", "s.cache")
       # staged() breaks only within its own frame, so this artifact is STANDALONE:
       # `installed` is False and its `with` / `unload()` are no-ops (something is
       # taken back out only for a capture holding frames the entry cannot reach).
       out = compiled(example_a)
```

```{eval-rst}
.. py:function:: precompile.load(artifact_path, cache_path, /, *, fn=None)

   Reconstruct a runnable from the two files a precompile capture wrote -- the
   ``python_code`` artifact and its ``cache``. They load only as a matched pair (the cache
   carries a sha256 of exactly the ``python_code`` bytes it was emitted with). The calling
   convention is read from ``python_code`` (the single source of truth); ``cache`` only
   accelerates loading -- it carries only the compiled backend artifact (the Inductor bundle
   for ``backend="inductor"``; empty for ``backend="eager"``) and no weights. You pass the
   model(s) again at runtime.
   Calling the result ignores this process's ambient ``torch.autocast`` on every device
   this build can autocast, so it returns the capture's dtypes rather than the dtypes the
   same eager call returns inside that region (see the autocast contract in the note
   above).

   .. warning::

      ``load`` runs the artifact as code: it executes ``python_code`` (via ``exec``),
      reads the ``cache`` envelope (a ``weights_only`` load) and, for the inductor
      backend, writes its bundle into the compile caches. A crafted ``python_code`` runs
      whatever it contains, and a crafted bundle plants pickles that a later cache hit
      unpickles. Treat the two files as trusted, executable input -- only load an artifact
      you produced yourself or otherwise trust, exactly as you would any code you are about
      to run (see Note [precompile programming model], invariant 7). ``load`` also emits a
      per-call warning before it runs.

   :param artifact_path: File holding ``python_code``, as written by
       :func:`precompile.capture` (or by :meth:`precompile.Capture.save`).
   :param cache_path: File holding ``cache``, as written by
       :func:`precompile.capture` (or by :meth:`precompile.Capture.save`).
   :param fn: For a dynamo artifact that serves by installing onto live code objects,
       the function object to install onto, when it is not importable from where it was
       captured (e.g. defined in ``__main__`` or a notebook); pass it before the first
       call. A standalone artifact rejects ``fn=`` with ``PrecompileError``, and since
       every artifact this build can produce is standalone, ``fn=`` is not available
       here: ``load`` refuses it unconditionally.
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
       you get is a property of the capture, not a load-time choice. The installing
       shape arrives with :class:`precompile.DynamoTracer` and is not available in this
       build yet (``load`` raises ``PrecompileError`` for an artifact whose
       ``SERVING_MODE`` is ``'installed'``).
   :raises PrecompileError: if either half cannot be read (a missing or unreadable file,
       one of the two paths handed artifact contents rather than a path, or the two paths
       swapped -- the cache's bytes then fail to decode as source); if ``python_code`` is
       not a valid precompile artifact (it fails to parse or is missing its
       calling-convention metadata); if ``cache`` is paired with a different
       ``python_code`` (mismatched ``backend`` tag, ``tracer`` tag, or ``code_hash``); if
       the artifact declares ``SERVING_MODE = 'installed'`` or ``fn=`` is passed (neither
       is available in this build); or if a runtime call violates the precompile
       contract.
   :raises ValueError: for one file named as both halves, or for a path that exists but
       is not a regular file. These are checked before either file is opened.

.. autoexception:: torch.compiler.PrecompileError

.. autoclass:: torch.compiler.PrecompiledRunnable
   :members: unload

   Every object :func:`precompile.load` returns is one of these, whichever shape the
   capture produced: this class is the standalone shape's contract, which is what every
   artifact this build can produce loads as, and
   :class:`torch.compiler.PrecompiledCallable` below is the installing shape, so
   ``isinstance(loaded, torch.compiler.PrecompiledRunnable)`` holds for both.

   .. py:attribute:: installed
      :type: bool

      Whether calling this handle installs onto the captured code objects. ``False`` for
      a standalone artifact, which serves by being called and has nothing to take out.

.. autoclass:: torch.compiler.PrecompiledCallable
   :members: unload, serve_time_compiles

   Returned by :func:`precompile.load` for an artifact that serves by installing,
   and used as a callable or context manager; it is not constructed directly.

   .. py:attribute:: installed
      :type: bool

      ``True``: this handle serves by installing onto the captured code objects, so
      :meth:`unload` -- or exiting it as a context manager -- is what takes that back
      out.

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

   The ``dynamo`` capture front-end, passed as ``tracer=`` to
   :func:`precompile.capture`. Not available in this build yet: ``capture`` raises
   ``PrecompileError`` for it until the front-end lands, and
   :class:`precompile.MakeFxTracer` stays the default until then. An execution-driven
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
       capture summary is not :attr:`precompile.PrecompileSummary.complete` (no guarded
       code at all, a frame that hit the recompile limit, was bypassed or was left
       uncovered, a capture call that raised, or no backend graph at all).
   :param require_no_risky_drops: defaults to ``True``. Refuse to produce an artifact that
       dropped a guard whose loss could change the answer (every drop made by a custom
       ``guard_filter_fn`` counts as risky).
   :param require_no_dropped_guards: defaults to ``False``. Refuse to produce an artifact
       that dropped any guard at all. Off by default because every model drops identity
       guards that cannot be serialized.

.. py:class:: precompile.Capture

   The object :func:`precompile.capture` returns. Enter it as a context manager and call
   it like ``fn`` inside the block to fold each call into the capture (see
   :func:`precompile.capture` for the semantics, including which exits write the two
   files); it is not constructed directly. Also exposes:

   .. py:method:: save()

      Write everything captured so far to the two files without ending the capture. Call
      it as often as you like inside the block once at least one call has been captured (an
      earlier ``save()`` raises ``PrecompileError``). With :class:`precompile.DynamoTracer`
      each call re-renders and rewrites both files, so a job that dies between saves leaves
      the last checkpoint loadable; a :class:`precompile.MakeFxTracer` capture records a
      single call, so ``save()`` and block exit write the same files. A gate refusal (the
      ``DynamoTracer`` ``require_*`` fields) or a write failure raises but writes nothing
      partial: the previous files stay intact and the capture stays open.

.. py:class:: precompile.PrecompileSummary

   Coverage and guard information from an observed capture, reported by the dynamo capture
   front-end (landing in a follow-up change). Frozen dataclass; ``str(summary)`` renders a
   one-line digest and :attr:`complete` says whether the capture covers everything it
   exercised.

   .. py:attribute:: frames

      How many frames the capture compiled.

   .. py:attribute:: resume_functions

      How many of those frames are graph-break continuations.

   .. py:attribute:: guarded_codes

      How many guarded code objects the artifact carries.

   .. py:attribute:: backend_graphs

      How many backend graphs were compiled.

   .. py:attribute:: bypassed

      Frames that fell back to eager.

   .. py:attribute:: truncated

      Frames that hit the recompile limit.

   .. py:attribute:: uncovered_frames

      Frames the capture calls never reached.

   .. py:attribute:: wont_generalize

      Frames whose guards pin a value and so will not generalize.

   .. py:attribute:: dropped_guards

      ``(guard_type, source)`` for guards the artifact omitted (could not serialize).

   .. py:attribute:: kept_guards

      ``(guard_type, source)`` for guards the artifact serialized and still checks.

   .. py:attribute:: risky_dropped_guards

      ``(guard_type, source)`` for the omitted guards that risk a wrong answer.

   .. py:attribute:: policy_dropped_guards

      ``(guard_type, source)`` for serializable guards the invariance policy dropped.

   .. py:attribute:: dropped_guard_code

      ``(guard_type, source, rendered_check)`` for each dropped slot that renders to a
      check. The slot's ``(guard_type, source)`` alone can be ambiguous -- a dropped
      ``HASATTR`` may be the benign companion of a kept ``TENSOR_MATCH`` or the only thing
      guarding an optional attribute -- so the rendered check is reported alongside to tell
      them apart.

   .. py:attribute:: capture_errors

      Messages from capture calls that raised.

   .. py:property:: complete

      Whether the capture covers everything it exercised: false if the capture produced no
      guarded code at all, if any frame hit the recompile limit, was bypassed or was left
      uncovered, or if a capture call raised, and false unless at least one backend graph
      was captured -- ``allow_empty_graphs`` lets a frame that compiled nothing still count
      as a guarded code, so ``guarded_codes`` alone cannot tell a real capture from an
      empty one.

   .. py:method:: dropped_guard_types()

      Count omitted guards by guard type.

   .. py:method:: kept_guard_types()

      Count serialized guards by guard type.

.. py:class:: precompile.FrameInvariants

   Per-frame guard classification, reported by the dynamo capture front-end (landing in a
   follow-up change). Frozen dataclass with the frame's code name in ``frame``, plus
   ``filename``, ``lineno``, the number of ``variants`` seen, and three tuples of
   :class:`precompile.GuardFact`: ``invariant`` (held identically across every variant),
   ``varying`` (differed between variants), and ``undetermined`` (a single variant could
   not decide).

.. py:class:: precompile.GuardFact

   One guard observed while compiling a frame variant. Frozen dataclass with ``guard_type``,
   ``source``, ``code`` (the rendered check parts), ``value``, and ``enforced`` (whether the
   artifact still checks it). ``render()`` returns one stable, human-readable line.

   .. py:method:: render()

      Render the guard as one stable, human-readable line.


```
