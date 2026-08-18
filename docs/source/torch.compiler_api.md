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
.. py:function:: precompile(fn, *example_args, backend="inductor", tracer=None, decompositions=None, example_inputs=None, guard_filter_fn=None, recompile_limit=256, dynamic=None, invariants=None)

   Ahead-of-time precompile ``fn`` using one of two input forms. Positional example
   arguments select the single-graph source-artifact path and return a self-contained,
   runnable Python source string plus an acceleration cache as ``(python_code, cache)``.
   The ``example_inputs`` keyword accepts a sequence of positional-argument tuples,
   captures every graph-break continuation and guarded recompilation exercised by those
   calls under full runtime guards, and returns the same ``(python_code, cache)`` pair.
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
   :param example_args: Example positional arguments for the single-graph source artifact;
       the ``nn.Module`` arguments are lifted and the rest are the runtime inputs.
   :param example_inputs: Sequence of positional-argument tuples for multi-graph capture.
       Calls run automatically under ordinary ``torch.no_grad()`` even if the caller is in
       ``torch.inference_mode()``; serve the resulting inference artifact under
       ``torch.no_grad()`` too. Inference mode is a distinct guarded state and must be
       captured manually if needed. Tensors created inside inference mode remain inference
       tensors after that context is disabled, so automatic examples reject them; create
       those inputs outside inference mode. Do not combine this with
       positional example arguments.
   :param backend: ``"inductor"`` (default) lowers through AOTAutograd + Inductor;
       ``"eager"`` keeps the captured ATen graph (layout-flexible, no kernels; shapes
       are still specialized to the example).
   :param tracer: capture front-end for the POSITIONAL path, where leaving it unset
       means ``"make_fx"``. It must not be passed at all alongside ``example_inputs``,
       which always uses Dynamo; that combination raises. ``"make_fx"`` is a non-strict
       make_fx trace. ``"dynamo"`` analyzes the Python (bytecode) rather than tracing one path and
       inlines the transformed bytecode Dynamo produces into ``python_code``, lowering the
       compiled subgraph through the same ``backend`` choices; it honors ``mark_unbacked``
       dynamic shapes (on either backend, though ``mark_unbacked(strict=True)`` raises --
       Dynamo captures a strict mark as a guardable backed dim), ``decompositions``, and
       training steps (a ``.backward()`` / ``torch.autograd.grad`` is traced into the graph
       and the parameter gradients are accumulated onto the runtime model like eager). A
       source-artifact path requires one full graph; pass a list of calls through the
       ``example_inputs`` keyword when Python graph-breaks or when several guarded/recompiled
       variants must be retained. Unlike ``make_fx``, the dynamo driver
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
   :param guard_filter_fn: Multi-graph serialization filter; returns one boolean per guard
       entry. Live capture retains all guards so later examples trigger their recompiles.
       Risky dropped guards are rejected by default when saving, and every
       custom-filter drop counts as risky.
   :param recompile_limit: Maximum multi-graph variants captured per frame; defaults to 256
       and overrides a lower ambient accumulated-recompile limit for this capture.
   :param dynamic: Multi-graph dynamic-shape policy forwarded to ``torch.compile``.
   :param invariants: Optional path receiving the multi-graph invariant report.
   :returns: For positional input, ``(python_code, cache)`` -- a self-contained Python
       source string and binary acceleration cache. For keyword ``example_inputs``, a
       session exposing ``summary()``, ``save()``, and invariant reporting. Its
       ``summary().complete`` covers the successful calls that ran, not every possible input.
   :raises PrecompileError: if capture, lowering, or a runtime call violates the
       contract (see the exception below).

   Example::

       python_code, cache = torch.compiler.precompile(lambda m, x: m(x), model, x)
       f = torch.compiler.precompile.load(python_code, cache)
       out = f(model, x)   # pass the model again at runtime

       def staged(x):
           y = x + 1
           scale = y.sum().item()  # a graph break
           return y * scale

       python_code, cache = torch.compiler.precompile(
           staged,
           example_inputs=[(example_a,), (example_b,)],
       )
       compiled = torch.compiler.precompile.load(python_code, cache)
       with compiled, torch.no_grad():
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
       paired with a different ``python_code`` (mismatched ``backend`` tag, ``tracer``
       tag, or ``code_hash``), or if a runtime call violates the precompile contract.

.. py:class:: precompile.ExampleInput(args=(), kwargs={})

   One capture call for ``example_inputs`` when positional arguments alone are not
   enough. A plain tuple in ``example_inputs`` is the positional arguments of one
   call; wrap a call that needs keyword arguments in this instead::

       torch.compiler.precompile(
           fn,
           example_inputs=[
               (x,),
               torch.compiler.precompile.ExampleInput(args=(x,), kwargs={"scale": 2}),
           ],
       )

.. autoexception:: torch.compiler.PrecompileError

.. autoclass:: torch.compiler.PrecompiledCallable
   :members:


```