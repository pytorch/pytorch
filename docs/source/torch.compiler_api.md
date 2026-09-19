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
`precompile.load`, `torch.compiler.PrecompiledRunnable` and the objects they return) is a
prototype API: signatures, error types and the artifact format may change between releases
without a deprecation cycle.
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
   ``tracer`` picks the capture front-end and carries its tracer-specific configuration
   (:class:`precompile.MakeFxTracer`, the default and the only one in this build). This is
   execution-driven coverage, not an exhaustive analysis: paths and values that no call
   executes are absent.
   ``fn`` is the whole computation, taking the model(s) as explicit arguments, e.g.
   ``lambda model, x: model(x)`` or a training step; its ``nn.Module`` arguments have their
   parameters/buffers lifted to graph inputs, so no weights are baked into the artifact --
   you pass the model again at runtime. Reload with ``torch.compiler.precompile.load``
   (below).

   .. note::

      With :class:`precompile.MakeFxTracer`, capture is non-strict and traces ``fn`` on
      FAKE tensors. Python control flow is specialized to the captured call, and shapes
      are static -- each size is baked in; a control-flow HOP is refused rather than
      specialized when its branch choice is not already a Python constant
      (``torch.while_loop``, or a ``torch.cond`` with a tensor predicate -- a ``SymBool``
      one arises only on the unbacked path, where it is refused too),
      while a ``torch.cond`` whose predicate IS a Python constant (e.g. a comparison of
      static sizes) short-circuits to the taken branch and specializes like any other
      Python ``if``. Tracing on fakes also
      refuses, on BOTH capture paths, an op with no meta/fake kernel, a read of a traced
      tensor's data (``.data_ptr()``, ``.numpy()``), an example input a fake tensor cannot
      represent (quantized, a view out of a sparse tensor) or whose metadata it silently
      drops (pinned, mkldnn, sparse), and a nested one; and, on a STATIC capture, a
      data-dependent op (``.item()``,
      ``.nonzero()``, a Python branch over a tensor value). It also refuses to run inside
      another trace, whose fake mode would outrank its own.
      The exception to static shapes is a tensor dim explicitly marked unbacked with
      ``torch._dynamo.decorators.mark_unbacked`` on the inputs before the call (with
      ``make_fx`` this requires the inductor backend); such a dim is captured as an
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
      duration of the call and the call returns the capture's dtypes, not the dtypes the
      same eager call returns inside that region -- so capture under the autocast you
      want baked in. See Note [precompile programming model] in
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
   serialized by an internal lock, so concurrent ``backend="inductor"`` captures lower one
   at a time; the capture phase and the ``backend="eager"`` path are NOT serialized. The
   lock is taken inside the ``cap(...)`` calls, where the lowering runs, not across the
   block.

   :param fn: The whole computation to capture, taking the model(s) and runtime inputs
       as positional arguments; :class:`precompile.MakeFxTracer` is positional-only.
   :param artifact_path: File to write ``python_code`` to when the block exits. Required.
   :param cache_path: File to write the acceleration cache to. Required.
   :param tracer: The capture front-end and its configuration, a
       :class:`precompile.MakeFxTracer` (the default, and the only one in this build).
   :param backend: ``"inductor"`` (default) lowers through AOTAutograd + Inductor;
       ``"eager"`` keeps the captured ATen graph (layout-flexible, no kernels; shapes
       are still specialized to the captured call).
   :param training: Run with grad enabled and lower a backward into the artifact;
       defaults to ``False``. Required for a ``fn`` that runs a backward.
   :returns: A :class:`precompile.Capture` -- a context manager and callable. The artifact
       is written to the two files on a clean exit from a block that captured at least one
       call; a block that raised leaves them untouched (the exception propagates), and a
       clean exit with nothing captured -- no call was made, or the only call raised and was
       caught -- raises ``PrecompileError`` instead of writing an empty artifact.
   :raises PrecompileError: if ``fn`` IS a model, or HOLDS a tensor or an
       ``nn.Module`` where this can see it (a bound method's ``__self__``, a
       ``functools.partial``'s bound argument) instead of taking it as a call argument.
   :raises ValueError: for an unknown ``backend``, for one file named as both halves, or
       for a path that exists but is not a regular file.
   :raises TypeError: if ``tracer`` is not a :class:`precompile.MakeFxTracer`.

   Those are what this call raises; the capture object has its own, documented on
   :class:`precompile.Capture` and :meth:`precompile.Capture.save`: a ``TypeError`` for a
   ``MakeFxTracer`` call made with keyword arguments, a ``NotImplementedError`` for
   ``backend="eager"`` with a ``mark_unbacked`` input (dynamic shapes need the inductor
   backend), a ``PrecompileError`` for capture, lowering or a served call violating the
   contract (see the exception below) and for every refusal of the capture's state machine
   (a second make_fx call, a clean exit with nothing captured), and the re-raised
   ``OSError`` of a write that failed out of the block exit or ``cap.save()`` -- the one to
   catch to retry the write.

   Example::

       with torch.compiler.precompile.capture(
           lambda m, x: m(x), artifact_path="m.py", cache_path="m.cache",
           tracer=torch.compiler.precompile.MakeFxTracer(),
       ) as cap:
           y = cap(model, x)   # runs for real, returns the served result
       f = torch.compiler.precompile.load("m.py", "m.cache")
       out = f(model, x)   # pass the model again at runtime
```

```{eval-rst}
.. py:function:: precompile.load(artifact_path, cache_path, /)

   Reconstruct a runnable from the two files a precompile capture wrote -- the
   ``python_code`` artifact and its ``cache``. They load only as a matched pair (the cache
   carries a sha256 of exactly the ``python_code`` bytes it was emitted with). The calling
   convention is read from ``python_code`` (the single source of truth); ``cache`` only
   accelerates loading -- it carries only the compiled backend artifact (the Inductor bundle
   for ``backend="inductor"``; empty for ``backend="eager"``) and no weights. You pass the
   model(s) again at runtime.
   Calling the result ignores this process's ambient ``torch.autocast``, so it returns
   the capture's dtypes rather than the dtypes the same eager call returns inside that
   region (see the autocast contract in the note above).

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
   :returns: A :class:`torch.compiler.PrecompiledRunnable` with the same calling
       convention as the captured ``fn``, taking positional arguments only. Every
       artifact this build produces is standalone: it installs nothing, ``installed`` is
       ``False`` and its ``with`` / ``unload()`` are no-ops. The other shape -- one that
       serves by INSTALLING onto the captured code objects -- arrives with the dynamo
       front-end; which one a load returns is a property of the capture, not a load-time
       choice.
   :raises PrecompileError: if either half cannot be read (a missing or unreadable file,
       or the two paths swapped -- the cache's bytes then fail to decode as source); if
       ``python_code`` is not a valid precompile artifact (it fails to parse or is
       missing its calling-convention metadata); if ``cache`` is paired with a different
       ``python_code`` (mismatched ``backend`` tag or ``code_hash``); or if a runtime
       call violates the precompile contract.
   :raises ValueError: for one file named as both halves, or for a path that exists but
       is not a regular file. These are checked before either file is opened.

.. autoexception:: torch.compiler.PrecompileError

.. autoclass:: torch.compiler.PrecompiledRunnable
   :members: unload

   Every object :func:`precompile.load` returns is one of these, whichever shape the
   capture produced: this class is the standalone shape's contract, which is what every
   artifact this build can produce loads as, and the installing shape arriving with the
   dynamo front-end is a subclass, so ``isinstance(loaded,
   torch.compiler.PrecompiledRunnable)`` holds for both.

   .. py:attribute:: installed
      :type: bool

      Whether calling this handle installs onto the captured code objects. ``False`` for
      a standalone artifact, which serves by being called and has nothing to take out.

.. py:class:: precompile.MakeFxTracer(decompositions=None)

   The ``make_fx`` capture front-end, passed as ``tracer=`` to
   :func:`precompile.capture`. A NON-STRICT single make_fx trace: it records the ATen ops
   of ONE execution of ``fn``, so a capture with this tracer takes exactly one call and
   refuses a second, and control flow and shapes are specialized to that call. Frozen
   dataclass.

   :param decompositions: Optional decomposition table (``dict`` of ``OpOverload`` to a
       decomposition function) forwarded to ``make_fx`` as its ``decomposition_table``;
       specific to this tracer. Defaults to ``None``.

.. py:class:: precompile.Capture

   The object :func:`precompile.capture` returns. Enter it as a context manager and call
   it like ``fn`` inside the block to fold each call into the capture (see
   :func:`precompile.capture` for the semantics, including which exits write the two
   files); it is not constructed directly. Also exposes:

   .. py:method:: save()

      Write everything captured so far to the two files without ending the capture. Call
      it as often as you like inside the block once at least one call has been captured (an
      earlier ``save()`` raises ``PrecompileError``). A
      :class:`precompile.MakeFxTracer` capture records a single call, so ``save()`` and
      block exit write the same files. A write failure raises but writes nothing
      partial: the previous files stay intact and the capture stays open. A failed write
      re-raises its ``OSError``, and ``save()`` stays open to retry that write, from
      outside the block too.

```
