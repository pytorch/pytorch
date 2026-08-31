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
.. py:function:: precompile(fn, *, backend="inductor", tracer="make_fx", decompositions=None, example_inputs, guard_filter_fn=None, recompile_limit=256, dynamic=None, invariants=None, training=False)

   Ahead-of-time precompile ``fn`` against ``example_inputs``, a sequence of calls each
   given as a tuple of positional arguments. precompile makes those calls itself and
   returns a self-contained, runnable Python source string plus an acceleration cache as
   ``(python_code, cache)``. ``tracer`` picks the capture front-end: ``"make_fx"`` (the
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
       entry. Live capture retains all guards so later examples trigger their recompiles.
       Risky dropped guards are rejected by default when saving, and every
       custom-filter drop counts as risky.
   :param recompile_limit: Maximum multi-graph variants captured per frame; defaults to 256
       and overrides a lower ambient accumulated-recompile limit for this capture.
   :param dynamic: Multi-graph dynamic-shape policy forwarded to ``torch.compile``.
   :param invariants: Optional path receiving the multi-graph invariant report.
   :returns: ``(python_code, cache)`` -- a self-contained Python source string and a
       binary acceleration cache. Positional example arguments are rejected with a
       ``TypeError``; ``example_inputs`` is the only calling convention.
   :raises PrecompileError: if capture, lowering, or a runtime call violates the
       contract (see the exception below).

   Example::

       python_code, cache = torch.compiler.precompile(
           lambda m, x: m(x), example_inputs=[(model, x)]
       )
       f = torch.compiler.precompile.load(python_code, cache)
       out = f(model, x)   # pass the model again at runtime

       def staged(x):
           y = x + 1
           scale = y.sum().item()  # a graph break
           return y * scale

       python_code, cache = torch.compiler.precompile(
           staged,
           tracer="dynamo",  # graph breaks and several examples need the dynamo tracer
           example_inputs=[(example_a,), (example_b,)],
       )
       compiled = torch.compiler.precompile.load(python_code, cache)
       with compiled, torch.no_grad():
           out = compiled(example_a)
```

```{eval-rst}
.. py:method:: precompile.load(python_code, cache, *, fn=None)

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
   :param fn: For a dynamo artifact that serves by installing onto live code objects,
       the function object to install onto, when it is not importable from where it was
       captured (e.g. defined in ``__main__`` or a notebook); pass it before the first
       call. A standalone artifact rejects ``fn=`` with ``PrecompileError``.
   :returns: A runnable callable with the same calling convention as the captured ``fn``.
       Arguments are matched positionally at both capture and load time; keyword-argument
       calling conventions are not supported. A dynamo artifact whose capture graph-broke
       or recompiled serves by INSTALLING onto the captured code objects: the returned
       callable mutates process state on first call (or on ``__enter__``) and supports
       ``with`` / ``unload()`` to take that back out. An artifact that captured a single
       whole graph is standalone: a plain callable, no installation and no ``unload``.
       Which one you get is a property of the capture, not a load-time choice.
   :raises PrecompileError: if ``python_code`` is not a valid precompile artifact (it
       fails to parse or is missing its calling-convention metadata), if ``cache`` is
       paired with a different ``python_code`` (mismatched ``backend`` tag, ``tracer``
       tag, or ``code_hash``), or if a runtime call violates the precompile contract.

.. py:class:: precompile.ExampleInput(args=(), kwargs={})

   One capture call for ``example_inputs`` when positional arguments alone are not
   enough (``tracer="dynamo"`` only). A plain tuple in ``example_inputs`` is the
   positional arguments of one
   call; wrap a call that needs keyword arguments in this instead::

       torch.compiler.precompile(
           fn,
           tracer="dynamo",
           example_inputs=[
               (x,),
               torch.compiler.precompile.ExampleInput(args=(x,), kwargs={"scale": 2}),
           ],
       )

.. autoexception:: torch.compiler.PrecompileError

.. autoclass:: torch.compiler.PrecompiledCallable
   :members:


```