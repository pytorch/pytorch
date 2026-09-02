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
.. py:function:: precompile(fn, *example_args, example_inputs=None, backend="inductor", tracer="make_fx", decompositions=None, recompile_limit=None, dynamic=None)

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
      automatically dynamic graphs produced when dimensions vary across examples.
      Each variant's guards are filtered as Dynamo creates them: guards derived from
      explicit inputs are retained, and named guards on the Python environment
      (module globals, imports) are dropped and recorded -- with the guard type,
      source, and capture-time value -- in the artifact's ``_DROPPED_GUARDS`` (also
      in ``PrecompileStateSummary.dropped_guards`` for stateful capture). The
      environment is an unchecked caller-provided invariant: changing a global after
      capture can silently run code specialized for the old value. Ambient torch
      state is checked on every call: autocast, default dtype, deterministic
      algorithms and torch-function state must match the capture, and a mismatch
      raises ``PrecompileError`` naming what differs; grad mode is pinned to the
      capture-time mode by the artifact itself, and the thread count
      (``torch.get_num_threads()``) is not checked, so an artifact serves on a
      machine with a different core count. Input changes drive variant dispatch: the
      loaded artifact checks variants newest-first (the automatically dynamic
      recompile of a shape supersedes the static variant it grew out of; note that
      ``torch._dynamo.package.CompilePackage.install`` serves the same serialized
      guards oldest-first), raises -- naming the guard that failed -- when a call
      fails every retained guard set, and never compiles a new variant. Graph breaks
      are not supported yet. Compiled graphs and
      kernels remain Python source; guard trees and transformed Dynamo bytecode are
      stored as opaque inline data because they have no Python-source representation.
      This initial path accepts a Python function with positional tensor/scalar arguments
      and containers of those values; closures and ``nn.Module`` arguments are not
      supported yet because their identity guards are not serializable, and
      numpy array/scalar arguments are not supported yet (convert them with
      ``torch.from_numpy`` / ``float(...)``). Every tensor the function uses must
      arrive through its arguments: a tensor it reads from a global, class, or module
      attribute is rejected at capture, naming it. Functions whose transformed
      bytecode reads or mutates a module global -- returning a global object, mutating
      a global container or a class attribute -- are rejected too, because the
      artifact runs in its own namespace where the global would be missing or the
      mutation lost. Distinct tensor inputs must not share or overlap
      storage -- their aliasing relation has no serialized form -- though passing the
      same tensor object more than once is supported; capture rejects overlapping
      inputs, and the loaded artifact raises on them whenever a captured graph
      mutates an input.
      Only strided and sparse input layouts are accepted -- sparse surfaces Dynamo's
      own rejection, and any other layout (e.g. jagged) is refused at capture (and at
      serve, for a mutating graph) because its aliasing cannot be verified.

      With ``tracer="dynamo"``, capture runs under the caller's ambient grad mode, and
      each captured graph's differentiability is inferred from its inputs, exactly as
      ``torch.compile`` infers it: under grad mode, inputs with ``requires_grad``
      yield differentiable graphs and inputs without stay inference graphs;
      ``requires_grad`` is part of each input's guards, so an input whose flag flipped
      since capture fails dispatch loudly rather than silently changing behavior. A
      capture under ``torch.no_grad()`` yields an inference artifact: it serves under
      ``torch.no_grad()``, or with grad enabled when no input requires grad, and raises
      ``PrecompileError`` when called with grad enabled on a ``requires_grad`` input
      (eager would record autograd history the artifact cannot). With ``backend="inductor"``, a
      differentiable graph is a joint forward+backward whose compiled segments contain
      readable Inductor source for both the AOTAutograd forward and backward, bridged
      by an emitted ``torch.autograd.Function``; differentiable outputs retain their
      ``grad_fn``, so a later ``backward()`` executes the captured backward kernels,
      across captured recompilations. Backward variants are specialized to
      output-tangent patterns observed during capture; the ordinary all-tangents-defined
      pattern is always covered, even when capture runs no backward, and a pattern not
      observed during capture falls back to it (materializing the missing tangents, as
      ``torch.compile`` does) instead of compiling at runtime. On the inductor backend only
      first-order backward is supported, the captured backward does not run under
      compiled autograd, and tensor-subclass and ``BackwardState`` training graphs are
      rejected. With ``backend="eager"`` the backward is
      live eager autograd through the emitted forward ops -- neither captured nor
      specialized (any tangent pattern and higher-order grad work), and, like the
      eager forward's kernels, resolved against the loaded torch rather than frozen
      in the artifact. The loaded artifact serves under the capture-time grad mode:
      a grad-mode artifact called under an ambient ``torch.no_grad()`` returns eager
      no_grad results -- freshly created outputs carry no autograd history, a view of
      an input is returned as a no_grad view of that input (so a later in-place write
      under grad mode raises exactly as in eager), and inputs passed through are
      returned untouched -- and calling under ``torch.inference_mode()`` raises
      ``PrecompileError`` -- serve under ``torch.no_grad()`` instead.

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
       Python function under the caller's ambient grad mode; it currently requires one
       full graph, rejects graph breaks, and does not yet support closures or
       ``nn.Module`` arguments.
   :param decompositions: Optional decomposition table (``dict`` of ``OpOverload`` to a
       decomposition function) forwarded to ``make_fx``; defaults to ``None`` and is not
       yet supported with ``tracer="dynamo"``.
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
       contract (see the exception below): an unsupported construct or effectful op, a
       tensor baked as a constant or read from the Python environment, a Dynamo-tracer
       ``fn`` that is not a plain function, has closure cells, takes ``nn.Module`` or
       numpy arguments, or reads/mutates module globals, ``decompositions`` with
       ``tracer="dynamo"``, ``mark_unbacked`` with ``backend="eager"``, overlapping
       tensor inputs, and a runtime input the artifact rejects. Anything about the
       captured computation or its inputs is a ``PrecompileError``.
   :raises TypeError: for Python-level misuse of this API: ``example_inputs`` that is
       not a sequence of tuples (a generator, a list of lists), positional examples
       combined with ``example_inputs``, or a Dynamo-tracer example tuple that does not
       fit ``fn``'s signature.
   :raises ValueError: for an empty ``example_inputs``, more than one tuple with
       ``tracer="make_fx"``, an unknown ``backend`` or ``tracer``, or
       ``recompile_limit`` / ``dynamic`` without ``tracer="dynamo"``.

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

   Dynamo training (inferred from ``requires_grad``)::

       examples = [(torch.randn(n, 4, requires_grad=True),) for n in (2, 3)]
       python_code, cache = torch.compiler.precompile(
           fn, example_inputs=examples, tracer="dynamo"
       )
       f = torch.compiler.precompile.load(python_code, cache)
       x = torch.randn(7, 4, requires_grad=True)
       f(x).sum().backward()  # executes the captured backward kernels

```

```{eval-rst}
.. py:method:: precompile.stateful(fn, *, example_inputs, artifact_path, cache_path, state=None, backend="inductor", recompile_limit=None, dynamic=None)

   Capture ``fn`` incrementally from a loop the caller owns (Dynamo tracer,
   implied). Every call runs its example tuples for real, records whatever
   guarded variants they newly exercise into the returned
   :class:`PrecompileState`, atomically rewrites the artifact and cache files
   at the given paths, and returns ``(results, state)``. The files on disk are
   always a loadable artifact for everything captured so far, so a job that
   dies mid-loop keeps a working artifact for the batches it saw. Feed
   ``stateful`` the calls that add variants (a new shape, a new branch), not
   every batch of a training loop: rewriting is proportional to everything
   captured so far, not to the call, and a call whose guards all hit adds
   nothing. The state is a context manager that releases the capture session
   on exit::

       def warm_up(step, representative_batches):
           _, state = torch.compiler.precompile.stateful(
               step, example_inputs=[(representative_batches[0],)],
               artifact_path="step.py", cache_path="step.cache",
           )
           with state:  # close() on exit, even if a later call raises
               for batch in representative_batches[1:]:
                   [result], state = torch.compiler.precompile.stateful(
                       step, example_inputs=[(batch,)], state=state,
                       artifact_path="step.py", cache_path="step.cache",
                   )
                   # result is this call's real step output.

   Capture semantics (rejections, guard filtering, grad mode, the
   programming-model contract) are exactly ``precompile(..., tracer="dynamo")``'s;
   only the delivery differs. Each example runs on the caller's live objects and
   nothing is retained afterwards, so a step may freely mutate its inputs.

   :param fn: The computation to capture; same requirements as
       ``precompile(fn, ..., tracer="dynamo")``.
   :param example_inputs: A sequence of positional-argument tuples run (for
       real) by this call.
   :param artifact_path: Where the self-contained Python artifact is
       atomically rewritten on every call.
   :param cache_path: Where the binary acceleration cache is atomically
       rewritten on every call.
   :param state: ``None`` starts fresh; passing the :class:`PrecompileState`
       returned by a previous call resumes it. A resumed call must use the same
       ``fn``, ``backend``, ``recompile_limit``, ``dynamic``, and ambient grad
       mode as the state, else it raises ``ValueError`` rather than produce a
       mixed artifact. After each rewrite ``state.summary()`` returns a
       :class:`PrecompileStateSummary` of what the artifact carries (calls,
       examples, variants, graphs, dynamic graphs, and the environment guards
       dropped from dispatch with their capture-time values -- also embedded in
       the artifact as ``_DROPPED_GUARDS``). The state is process-local and not
       serializable; call ``state.close()`` (or use ``with state:``) when done
       capturing.
   :param backend: As on ``precompile``.
   :param recompile_limit: Cap on captured variants; defaults to
       ``max(torch._dynamo.config.recompile_limit, 256)`` because accumulating
       captures outgrow the config default. Fixed when the state is created.
   :param dynamic: As on ``precompile``; must not change across resumed calls.
   :returns: ``(results, state)`` -- ``results`` is always a list with one
       entry per example tuple of THIS call (never unwrapped, so a fn that
       itself returns a list is unambiguous), and ``state`` is the
       :class:`PrecompileState` to pass back in. Reload the on-disk pair with
       ``precompile.load_files(artifact_path, cache_path)``.
   :raises PrecompileError: as ``precompile`` with ``tracer="dynamo"``; also when
       an observed backward tangent pattern cannot be compiled, or when the
       accumulated capture can no longer be rendered (close the state and capture
       again without the offending example).
   :raises TypeError: as ``precompile``; also for a ``state`` that is not a
       ``PrecompileState``.
   :raises ValueError: as ``precompile``; also for equal ``artifact_path`` and
       ``cache_path``, a closed ``state``, or a resumed call whose ``fn``,
       ``backend``, ``recompile_limit``, ``dynamic`` or grad mode differs from the
       state's.
```

```{eval-rst}
.. py:class:: precompile.PrecompileState

   The accumulated capture state ``precompile.stateful`` returns and resumes
   from. It owns the live Dynamo capture session (installed code caches, the
   package new variants accumulate into, the isolated PGO record, the compiled
   graphs, and the environment guards dropped so far) and the ambient grad mode
   fixed at creation. Process-local and not serializable.

   .. py:method:: summary()

      The :class:`PrecompileStateSummary` of the most recently written artifact,
      or ``None`` before one exists.

   .. py:method:: close()

      Release the capture session. Idempotent; a closed state cannot be resumed,
      and artifact files written by earlier calls remain valid. Without it the
      session stays pinned by Dynamo's process-global registries until
      ``torch._dynamo.reset()``. The state is also a context manager
      (``with state:``) that calls ``close()`` on exit.
```

```{eval-rst}
.. py:class:: precompile.PrecompileStateSummary

   A named tuple describing the most recently written artifact of a stateful
   capture: ``calls`` (stateful calls so far), ``examples`` (example tuples
   run), ``variants`` (guarded Dynamo variants), ``graphs`` (compiled graphs),
   ``dynamic_graphs`` (graphs with symbolic shapes), and ``dropped_guards`` --
   the sorted, deduplicated ``(guard_type, source, value)`` triples of
   environment guards removed from dispatch, where ``value`` is the ``repr`` of
   the capture-time value the artifact is specialized to.
```

```{eval-rst}
.. py:method:: precompile.load(python_code, cache)

   Reconstruct a runnable from the ``(python_code, cache)`` pair returned by
   ``precompile``. The calling convention is read from ``python_code`` (the
   single source of truth); ``cache`` only accelerates loading -- it carries
   only the compiled backend artifact (the Inductor bundle for
   ``backend="inductor"``; empty for ``backend="eager"``) and no weights, and
   ``cache=None`` (or an empty cache) means no cache: JIT from ``python_code``
   with a warning. You pass the model(s) again at runtime. For the file pair a
   stateful capture writes, use ``precompile.load_files``.

   .. warning::

      ``load`` runs the artifact as code: it executes ``python_code`` (via ``exec``) and,
      for the inductor backend, primes the kernel caches from the ``cache``. Treat
      ``(python_code, cache)`` as trusted, executable input -- only load a pair you
      produced yourself or otherwise trust, exactly as you would any code you are about to
      run (see Note [precompile programming model], invariant 7). ``load`` also emits a
      per-call warning before it runs.

   :param python_code: The self-contained Python source string returned by ``precompile``.
   :param cache: The binary acceleration cache returned by ``precompile``, or ``None``.
   :returns: A runnable callable with the same calling convention as the captured ``fn``.
       A ``make_fx`` artifact takes positional arguments only; a ``dynamo`` artifact
       binds its arguments like the traced ``fn``, so keyword arguments and defaults
       work.
   :raises TypeError: if ``python_code`` is not a ``str`` or ``cache`` is not
       ``bytes`` / ``None``.
   :raises PrecompileError: if ``python_code`` is not a valid precompile artifact (it
       fails to parse or is missing its calling-convention metadata), if ``cache`` is
       paired with a different ``python_code`` (mismatched ``backend`` tag or
       ``code_hash``; on both tracers), if a Dynamo artifact is loaded under a
       different Python minor version or torch version than produced it, or if a
       runtime call violates the precompile contract (a keyword call of a ``make_fx``
       artifact included). A cache whose ``format``/``version`` tag does not match (a
       foreign or different-build envelope) is not an error: the cache is acceleration
       only, so it degrades to JIT'ing from ``python_code`` with a warning.
```

```{eval-rst}
.. py:method:: precompile.load_files(artifact_path, cache_path)

   Reconstruct a runnable from the file pair a ``precompile.stateful`` capture
   wrote, with the pairing rules an on-disk pair needs. A stateful rewrite
   renames the artifact and then the cache, so a crash between the two leaves
   a NEW artifact with the OLD cache, and the very first rewrite's crash window
   leaves an artifact with no cache file at all; both degrade to a cold cache
   with a warning (the artifact is fully self-contained), and the next
   successful rewrite repairs the pair.

   :param artifact_path: Path of the self-contained Python artifact.
   :param cache_path: Path of the binary acceleration cache.
   :returns: A runnable callable, as ``precompile.load``.
   :raises FileNotFoundError: if ``artifact_path`` does not exist.
   :raises PrecompileError: as ``precompile.load``, except that a cache whose
       ``backend``/``code_hash`` does not match the artifact degrades to a cold
       cache with a warning instead of raising.

.. autoexception:: torch.compiler.PrecompileError
```
