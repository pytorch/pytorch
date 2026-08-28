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
     ExampleInput
     GuardFact
     FrameInvariants
     PrecompiledCallable
     PrecompileSummary
```

## torch.compiler.precompile

% precompile is a callable instance (not a plain function), which Sphinx
% autosummary cannot render, so it is documented manually below and
% intentionally omitted from the autosummary block above.

```{eval-rst}
.. py:function:: precompile(fn, *example_args, example_inputs=None, backend="inductor", tracer="make_fx", decompositions=None, training=False, recompile_limit=256, dynamic=None, guard_filter_fn=None, invariants=None, require_complete=True, require_no_risky_drops=True, require_no_dropped_guards=False)

   Ahead-of-time precompile ``fn`` against example inputs, returning a runnable Python
   source string plus an acceleration cache as ``(python_code, cache)``.
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

      For compatibility, positional arguments after ``fn`` describe one example call.
      Do not combine positional examples with ``example_inputs``.

      With ``tracer="dynamo"``, every tuple or ``ExampleInput`` in ``example_inputs``
      is executed exactly once during capture. Recompilations become guarded variants
      in the artifact, including automatically dynamic graphs produced when dimensions
      vary across examples. Its contract requires that (1) globals, context-manager
      state, and the rest of the Python environment are semantically identical at
      capture and runtime, and (2) only explicit inputs vary in ways that would cause a
      recompile. Guards that only enforce the first promise are omitted. By default,
      every portable input-derived guard is retained, including invariant guards needed
      to reject an unseen input variation. Distinct tensor inputs must not share or
      overlap storage at capture or runtime. An explicit input must not
      also be reachable through globals or other environment state; statically visible
      identity relations are rejected during capture, and dynamic native indirection is
      outside the supported contract. Input pytree structures must be serializable so
      these checks can be reconstructed at runtime. User-defined code must access Python
      module objects (``types.ModuleType``) through statically visible attribute paths
      rather than pass or alias them as values. Python functions that mutate globals or
      mutable objects reachable through the Python environment are rejected. Calls and
      implicit protocol operations on such objects are also rejected when their behavior
      cannot be verified statically.

      Each retained guard is rebuilt independently from its frozen capture snapshot.
      An environment-only guard that cannot be rebuilt is omitted with its dependent
      attribute checks; an input-derived or unknown-provenance guard instead raises a
      ``PrecompileError``. Rebuilt guard facts and leaf predicates are compared with the
      live capture so a changed input predicate cannot silently ship. This filtering is
      at guard-record granularity, so a retained composite record can still contain
      invariant leaf checks. Breaking an unchecked environment assumption can silently
      miscompute. An artifact raises when a call fails every retained guard set,
      including when captured graph-break frames require installed mode. It never
      compiles an uncovered variant while serving. Graph breaks are captured as Dynamo
      resume frames.
      Closure-free Python functions wrapped with ``torch._dynamo.disable`` are embedded
      and execute eagerly between compiled graph segments. Global names left in
      standalone transformed bytecode must resolve to recursive literal values or
      independently importable objects. Installed frames may resolve globals from
      their defining modules. Python functions cannot mutate globals or mutable objects
      reachable through the Python environment, and unverified behavior on those objects
      is rejected conservatively. Disabled functions also cannot use
      ``globals()``, ``eval()``, or ``exec()``; their importable module globals are
      rebound at load, while recursive literal globals and defaults are captured by
      value. Top-level defaults must also be recursive literals; mutable or user-defined
      values must be passed explicitly rather than used as defaults. Tensor-valued
      defaults and tensor-valued globals referenced by user-defined code are rejected
      because user-owned tensors must be explicit inputs. Bound methods are unsupported;
      pass the unbound function and its receiver as an explicit input.
      Compiled graph bodies and kernels remain Python source. The eager backend
      supports higher-order graphs such as ``torch.cond``, ``torch.while_loop``,
      non-reentrant activation checkpointing, ``vmap``, autocast, and grad-mode regions.
      Their nested graph bodies are rendered as Python too, while the FX ``Graph``
      structure required by eager higher-order-op interpreters is stored as opaque
      inline data. Guard trees, transformed Dynamo entry/resume bytecode, and embedded
      disabled-function bytecode are also opaque. The top-level function cannot have
      closure cells or nested functions that capture its locals. ``nn.Module`` arguments
      are supported and are checked at runtime for type, training mode, parameter/buffer
      names, aliasing, shapes, strides, dtypes, devices, and ``requires_grad`` state.

      Entry and graph-break resume frames are dispatched directly from the generated
      source. If capture also compiles a nested frame reachable only through an ordinary
      Python call, the artifact uses an isolated installed mode so that frame is served
      too instead of silently running eager. Loading prepares the backends and guard
      trees without installing them. Installation happens on first call (or
      context-manager entry), and ``unload()`` removes only that artifact's entries.
      Installed artifacts require the defining Python modules to be importable. Pass the
      live callable as ``fn=`` to ``load`` when the entry itself must be rebound.
      ``capture_summary.variant_examples`` reports, for each captured frame, the index
      of the example that first produced each guarded variant.

      Pass ``training=True`` with ``tracer="dynamo"`` to capture differentiable graphs
      on either backend. Inductor segments contain readable source for both their
      AOTAutograd forward and backward, bridged by an emitted
      ``torch.autograd.Function``; eager segments retain differentiable ATen operations.
      Outputs retain their ``grad_fn``, so a later ``backward()`` runs the captured
      backward. Serving pins grad mode to ``training``: inference artifacts run with
      gradients disabled even inside ``torch.enable_grad()``, while training artifacts
      enable gradients even inside ``torch.no_grad()``. Training works across captured
      recompilations and graph breaks. Only first-order backward is supported;
      tensor-subclass and ``BackwardState`` training graphs are rejected.

   With ``tracer="make_fx"``, if ``fn`` runs a backward, the artifact re-runs the whole
   forward and backward and scatters the resulting parameter gradients onto the runtime
   model's ``parameters()`` ``.grad`` fields, accumulating (``p.grad += g``) exactly like
   eager ``.backward()`` -- so keep your usual ``zero_grad()`` / ``optimizer.step()``
   loop. Which params receive a grad is fixed at capture time (frozen or
   non-contributing params stay ``.grad = None``). The artifact returns ``fn``'s own
   result (``None`` for a bare ``.backward()`` step), not the gradients.

   :param fn: The whole computation to capture, taking the model(s) and runtime inputs
       as positional arguments.
   :param example_args: Positional arguments for one example call, retained for
       compatibility with the original API. Do not combine them with ``example_inputs``.
   :param example_inputs: A sequence of positional-argument tuples or
       ``torch.compiler.ExampleInput`` values for ``fn``. ``ExampleInput`` carries an
       ``args`` tuple and ``kwargs`` dict. The
       ``make_fx`` tracer requires exactly one tuple. The ``dynamo`` tracer accepts one
       or more calls and records the guarded recompilations they exercise. Keyword
       examples are supported only by the Dynamo tracer. With
       ``make_fx``, ``nn.Module`` arguments within the tuple are lifted and the rest are
       runtime inputs.
   :param backend: ``"inductor"`` (default) lowers through AOTAutograd + Inductor;
       ``"eager"`` keeps the captured ATen graph (layout-flexible, no kernels; shapes
       are still specialized to the example). With the Dynamo tracer, eager preserves
       nested higher-order-op graphs without symbolic retracing at load.
   :param tracer: capture front-end. ``"make_fx"`` (default) is a non-strict make_fx
       trace. ``"dynamo"`` captures guarded specializations and recompilations from a
       Python function, including graph-break resume frames; it does not yet support
       top-level closures or nested functions that capture locals.
   :param decompositions: Optional decomposition table (``dict`` of ``OpOverload`` to a
       decomposition function) forwarded to ``make_fx``; defaults to ``None`` and is not
       yet supported with ``tracer="dynamo"``.
   :param training: If ``True``, capture a differentiable Dynamo/Inductor artifact whose
       outputs can be passed to ``backward()``. Defaults to ``False`` and requires
       ``tracer="dynamo"``. Both eager and Inductor backends are supported.
   :param recompile_limit: Maximum captured variants per Dynamo code object. Defaults to
       256 and must be positive.
   :param dynamic: Dynamo dynamic-shape policy, with the same meaning as
       ``torch.compile``. ``None`` enables automatic promotion when example shapes vary.
   :param guard_filter_fn: Optional callable returning one boolean per candidate Dynamo
       guard. It may narrow the portable default set but cannot restore unserializable
       guards. Removing an input guard is considered risky and requires
       ``require_no_risky_drops=False``. Finalization omits guards covered by the
       invariant-environment contract and verifies that the frozen captured variants
       remain distinguishable.
   :param invariants: Optional path for a text report classifying captured guards as
       invariant, varying, or undetermined for each frame.
   :param require_complete: Reject bypassed, truncated, uncovered, or failed captures.
       Defaults to ``True``.
   :param require_no_risky_drops: Reject dropped guards that may affect dispatch.
       Defaults to ``True``.
   :param require_no_dropped_guards: Reject every dropped guard. Defaults to ``False``
       because ordinary programs commonly contain unserializable identity guards.
   :returns: ``(python_code, cache)`` -- an executable Python source string (the single
       source of truth for the calling convention) and a binary acceleration
       cache (no weights, no calling-convention metadata; it carries a small
       format/version/backend/code_hash integrity tag that ``load`` verifies).
   :raises PrecompileError: if capture, lowering, or a runtime call violates the
       contract (see the exception below).

   Example::

       python_code, cache = torch.compiler.precompile(
           lambda m, x: m(x), example_inputs=[(model, x)]
       )
       f = torch.compiler.precompile.load(python_code, cache)
       out = f(model, x)   # pass the model again at runtime

   With ``tracer="dynamo"``, if ``fn`` itself is a user-defined Python ``nn.Module``
   whose ``forward`` method Dynamo captures as a Python frame, pass the runtime module
   first when calling the loaded artifact::

       python_code, cache = torch.compiler.precompile(
           model, example_inputs=[(x,)], tracer="dynamo"
       )
       f = torch.compiler.precompile.load(python_code, cache)
       out = f(model, x)

   For built-in modules such as ``Linear`` and ``Sequential``, use the wrapper form
   above instead.

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
.. py:method:: precompile.load(python_code, cache, *, fn=None)

   Reconstruct a runnable from the ``(python_code, cache)`` pair returned by
   ``precompile``. The calling convention is read from ``python_code`` (the single
   source of truth); ``cache`` only accelerates loading -- it carries only the compiled
   backend artifact (the Inductor bundle for ``backend="inductor"``; empty for
   ``backend="eager"``) and no weights. You pass the model(s) again at runtime.
   ``fn`` is optional and is used by an installed Dynamo artifact to bind its captured
   package to a live callable. Loaded installed artifacts also support ``unload()`` and
   the context-manager protocol. Dynamo artifacts are tied to the Python minor version
   and torch version that produced them; an incompatible load raises
   :class:`PrecompileError`.

   .. warning::

      ``load`` runs the artifact as code: it executes ``python_code`` (via ``exec``) and,
      for the inductor backend, primes the kernel caches from the ``cache``. Treat
      ``(python_code, cache)`` as trusted, executable input -- only load a pair you
      produced yourself or otherwise trust, exactly as you would any code you are about to
      run (see Note [precompile programming model], invariant 7). ``load`` also emits a
      per-call warning before it runs.

   :param python_code: The executable Python source string returned by ``precompile``.
   :param cache: The binary acceleration cache returned by ``precompile``.
   :returns: A :class:`torch.compiler.PrecompiledCallable` with the same calling
       convention as the captured ``fn``, except that a directly captured user-defined
       module takes the runtime module as its first argument. The Dynamo tracer also
       preserves captured keyword-argument calling conventions.
   :raises PrecompileError: if ``python_code`` is not a valid precompile artifact (it
       fails to parse or is missing its calling-convention metadata), if ``cache`` is
       paired with a different ``python_code`` (mismatched ``backend`` tag or
       ``code_hash``), or if a runtime call violates the precompile contract.

.. autoexception:: torch.compiler.PrecompileError
```
