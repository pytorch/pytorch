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
.. py:function:: precompile(fn, *example_args, example_inputs=None, backend="inductor", tracer="make_fx", decompositions=None, training=False, recompile_limit=None, dynamic=None)

   Ahead-of-time precompile ``fn`` against example inputs, returning a self-contained,
   runnable Python source string plus an acceleration cache as ``(python_code, cache)``.
   To capture incrementally from a loop the caller owns, use the sibling entry point
   ``precompile.stateful`` below.
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
      artifact retains guards derived from explicit inputs and may drop guards on the
      Python environment. The environment is an unchecked caller-provided invariant:
      changing globals or context-manager state after capture can silently run code
      specialized for the old environment. Input changes remain responsible for variant
      dispatch. The loaded artifact raises when a call fails every retained guard set,
      and never compiles a new variant. Graph breaks are not supported yet. Compiled graphs and
      kernels remain Python source; guard trees and transformed Dynamo bytecode are
      stored as opaque inline data because they have no Python-source representation.
      This initial path accepts a Python function with positional tensor/scalar arguments
      and containers of those values; closures and ``nn.Module`` arguments are not
      supported yet because their identity guards are not serializable. A global whose
      object graph contains a tensor is rejected (conservatively, even when the fn only
      reads a non-tensor field of it) because every tensor must be an explicit input,
      and functions that mutate globals are rejected because the artifact could not
      reproduce the side effect. Distinct tensor inputs must not share or overlap
      storage -- their aliasing relation has no serialized form -- though passing the
      same tensor object more than once is supported; capture rejects overlapping
      inputs and the loaded artifact raises on them. An explicit pytree-leaf input
      must not also be reachable through the Python environment (including a
      referenced global's class or module attributes); such aliasing is rejected --
      on every one-shot capture, and once at state creation for stateful capture,
      since the environment is declared invariant afterward (a later environment
      change is a contract violation that can silently serve capture-time results).
      Container inputs dissolve into their leaves, so a container's own identity is
      not checked. Exempt from the aliasing rejection: dtypes, layouts, and memory
      formats (process-wide, value-guarded singletons) and enum members (a used enum
      argument fails capture loudly on its unserializable identity guard regardless).
      Only strided and sparse input layouts are accepted -- sparse surfaces Dynamo's
      own rejection, and any other layout (e.g. jagged) is refused at capture and at
      serve because its aliasing cannot be verified.

      Pass ``training=True`` with ``tracer="dynamo"`` and ``backend="inductor"`` to
      capture differentiable graphs. Each compiled segment contains readable Inductor
      source for both its AOTAutograd forward and backward, bridged by an emitted
      ``torch.autograd.Function``. Outputs retain their ``grad_fn``, so a later
      ``backward()`` executes the captured backward kernels. Training works across
      captured recompilations. Backward variants are specialized to output-tangent
      patterns observed during capture, and an unseen pattern fails instead of compiling
      at runtime; the ordinary all-tangents-defined pattern is always covered, even
      when capture runs no backward. Only first-order backward is supported; tensor-subclass and
      ``BackwardState`` training graphs are rejected.

   With ``tracer="make_fx"``, if ``fn`` runs a backward, the artifact re-runs the whole
   forward and backward and scatters the resulting parameter gradients onto the runtime
   model's ``parameters()`` ``.grad`` fields, accumulating (``p.grad += g``) exactly like
   eager ``.backward()`` -- so keep your usual ``zero_grad()`` / ``optimizer.step()``
   loop. Which params receive a grad is fixed at capture time (frozen or
   non-contributing params stay ``.grad = None``). The artifact returns ``fn``'s own
   result (``None`` for a bare ``.backward()`` step), not the gradients.

   :param fn: The whole computation to capture, taking the model(s) and runtime inputs
       as positional arguments.
       Positional arguments after ``fn`` remain supported as one example call and cannot
       be combined with ``example_inputs``.
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
       Python function; it currently requires one full graph, rejects graph breaks, and
       does not yet support closures or ``nn.Module`` arguments.
   :param decompositions: Optional decomposition table (``dict`` of ``OpOverload`` to a
       decomposition function) forwarded to ``make_fx``; defaults to ``None`` and is not
       yet supported with ``tracer="dynamo"``.
   :param training: If ``True``, capture a differentiable Dynamo/Inductor artifact whose
       outputs can be passed to ``backward()``. Defaults to ``False`` and currently
       requires ``tracer="dynamo"`` and ``backend="inductor"``.
   :param recompile_limit: Cap on captured variants per Dynamo capture
       (``tracer="dynamo"`` only). The default is
       ``torch._dynamo.config.recompile_limit`` or the example count plus one,
       whichever is larger.
   :param dynamic: Forwarded to Dynamo (``tracer="dynamo"`` only): ``None`` keeps the
       automatic dynamic-shape policy, ``True``/``False`` forces or forbids symbolic
       shapes.
   :returns: ``(python_code, cache)`` -- a self-contained Python source string (the
       single source of truth for the calling convention) and a binary acceleration
       cache (no weights, no calling-convention metadata; it carries a small
       format/version/backend/code_hash integrity tag that ``load`` verifies).
   :raises PrecompileError: if capture, lowering, or a runtime call violates the
       contract (see the exception below).

   Dynamo artifacts are tied to the Python minor version and the torch version used
   to create them; loading one under a different version raises
   :class:`PrecompileError`.

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

   Dynamo training::

       examples = [(torch.randn(n, 4, requires_grad=True),) for n in (2, 3)]
       python_code, cache = torch.compiler.precompile(
           fn, example_inputs=examples, tracer="dynamo", training=True
       )
       f = torch.compiler.precompile.load(python_code, cache)
       x = torch.randn(7, 4, requires_grad=True)
       f(x).sum().backward()  # executes the captured backward kernels

```

```{eval-rst}
.. py:method:: precompile.stateful(fn, *, example_inputs, artifact_path, cache_path, state=None, backend="inductor", training=False, recompile_limit=None, dynamic=None)

   Capture ``fn`` incrementally from a loop the caller owns (Dynamo tracer,
   implied). Every call runs its example tuples for real, records whatever
   guarded variants they newly exercise into the returned opaque ``state``,
   atomically rewrites the artifact and cache files at the given paths, and
   returns ``(results, state)``. The files on disk are always a loadable
   artifact for everything captured so far, so a job that dies mid-loop keeps
   a working artifact for the batches it saw::

       state = None
       try:
           for batch in batches:
               [result], state = torch.compiler.precompile.stateful(
                   step, example_inputs=[(batch,)], state=state,
                   artifact_path="step.py", cache_path="step.cache",
                   training=True, recompile_limit=256,
               )
               # result is this call's real step output; run the training loop on it.
       finally:
           if state is not None:
               state.close()  # release the capture session

   Capture semantics (rejections, guard minimization, the programming-model
   contract) are exactly ``precompile(..., tracer="dynamo")``'s; only the
   delivery differs. A call whose guards all hit adds nothing; guard
   minimization is re-run over every example seen so far on each rewrite, so
   the state keeps a pre-execution snapshot of every example tuple alive
   (tensors by reference; a step may freely mutate its container inputs).
   Rewriting is proportional to everything captured so far, not to the call.

   :param fn: The computation to capture; same requirements as
       ``precompile(fn, ..., tracer="dynamo")``.
   :param example_inputs: A sequence of positional-argument tuples run (for
       real) by this call.
   :param artifact_path: Where the self-contained Python artifact is
       atomically rewritten on every call.
   :param cache_path: Where the binary acceleration cache is atomically
       rewritten on every call.
   :param state: ``None`` starts fresh; passing the state returned by a
       previous call resumes it. A resumed call must use the same ``fn``,
       ``backend``, ``training``, ``recompile_limit``, and ``dynamic`` as the
       state, else it raises rather than produce a mixed artifact. After each
       rewrite ``state.summary()`` reports what the artifact carries (calls,
       examples, variants, graphs, dynamic graphs, and the environment guards
       minimization dropped from at least one variant -- also embedded in the
       artifact as ``_DROPPED_GUARDS``). The state is process-local and not
       serializable; call ``state.close()`` when done capturing.
   :param backend: As on ``precompile``.
   :param training: As on ``precompile`` (requires ``backend="inductor"``).
   :param recompile_limit: Cap on captured variants; defaults to
       ``max(torch._dynamo.config.recompile_limit, 256)`` because accumulating
       captures outgrow the config default. Fixed when the state is created.
   :param dynamic: As on ``precompile``; must not change across resumed calls.
   :returns: ``(results, state)`` -- ``results`` is always a list with one
       entry per example tuple of THIS call (never unwrapped, so a fn that
       itself returns a list is unambiguous), and ``state`` is the opaque
       accumulated-capture state to pass back in. Reload the on-disk pair with
       ``precompile.load(artifact_path=..., cache_path=...)``.
   :raises PrecompileError: as ``precompile`` with ``tracer="dynamo"``.
```

```{eval-rst}
.. py:method:: precompile.load(python_code=None, cache=None, *, artifact_path=None, cache_path=None)

   Reconstruct a runnable from the ``(python_code, cache)`` pair returned by
   ``precompile``, or -- the natural companion of stateful capture's on-disk
   rewrites -- from the file pair at ``artifact_path``/``cache_path``. Pass one
   form or the other, not both (mixing raises ``TypeError``; giving only one
   path raises ``ValueError``). The calling convention is read from
   ``python_code`` (the single source of truth); ``cache`` only accelerates
   loading -- it carries only the compiled backend artifact (the Inductor
   bundle for ``backend="inductor"``; empty for ``backend="eager"``) and no
   weights. You pass the model(s) again at runtime.

   .. warning::

      ``load`` runs the artifact as code: it executes ``python_code`` (via ``exec``) and,
      for the inductor backend, primes the kernel caches from the ``cache``. Treat
      ``(python_code, cache)`` as trusted, executable input -- only load a pair you
      produced yourself or otherwise trust, exactly as you would any code you are about to
      run (see Note [precompile programming model], invariant 7). ``load`` also emits a
      per-call warning before it runs.

   :param python_code: The self-contained Python source string returned by ``precompile``.
   :param cache: The binary acceleration cache returned by ``precompile``.
   :param artifact_path: With ``cache_path``, read the pair from the files a
       stateful capture wrote (keyword-only).
   :param cache_path: The cache file path paired with ``artifact_path`` (keyword-only).
   :returns: A runnable callable with the same calling convention as the captured ``fn``.
       Arguments are matched positionally at both capture and load time; keyword-argument
       calling conventions are not supported.
   :raises PrecompileError: if ``python_code`` is not a valid precompile artifact (it
       fails to parse or is missing its calling-convention metadata), if a ``make_fx``
       artifact's ``cache`` is paired with a different ``python_code`` (mismatched
       ``backend`` tag or ``code_hash``), if a Dynamo artifact is loaded under a
       different Python minor version or torch version than produced it, or if a
       runtime call violates the precompile contract. A Dynamo artifact's mismatched
       (or missing, on the path form) cache instead degrades to a cold cache with a
       warning: the python_code is fully self-contained, and stateful capture's
       two-rename rewrite can legitimately leave a mismatched pair after a crash. A
       cache whose ``format``/``version`` tag does not match (a foreign or
       different-build envelope) also degrades to JIT'ing from ``python_code``.

.. autoexception:: torch.compiler.PrecompileError
```
