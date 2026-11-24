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

```{eval-rst}
.. py:module:: torch.compiler.precompile
.. currentmodule:: torch.compiler
```

```{warning}
`torch.compiler.precompile` and everything reached through it is a prototype API.
Signatures, error types and the artifact format may change between releases without a
deprecation cycle.
```

`torch.compiler.precompile` captures a whole computation -- `fn(model, x)`, with the
model(s) passed as arguments -- ahead of time from the caller's own calls, and lowers it to
a self-contained Python source artifact plus an acceleration cache that a fresh process
reloads. No weights are baked in, so the model is passed again at runtime:

```python
# step is a module-level function: def step(model, x): return model(x)
with torch.compiler.precompile.capture(
    step, artifact_path="m.py", cache_path="m.cache"
) as cap:
    y1 = cap(model, x1)
    y2 = cap(model, x2)  # a recompile or graph break becomes another variant

f = torch.compiler.precompile.load("m.py", "m.cache")  # in a fresh process
y = f(model, x1)
```

The default `DynamoTracer` records every frame Dynamo compiles while your calls run and
the artifact dispatches among them by their guards; `MakeFxTracer` is a single non-strict
trace under the contract of Note [precompile programming model] in `torch/_precompile.py`.
`torch.compiler.precompile` is distinct from `torch._dynamo.config.caching_precompile`
(a `torch.compile` caching mode).

### Coverage, the capture gates, and serve-time misses

A `DynamoTracer` capture records only what your calls executed: a branch, a shape or a
configuration no call reached is absent from the artifact. A dimension the capture saw at
one size compiles static, so feed at least two sizes of any dimension that varies in
serving, or set `DynamoTracer(dynamic=True)`.

At serve time the artifact dispatches among the captured variants by their serialized
guards, and a call no variant covers is a miss:

- By default a miss raises `PrecompileError`; nothing compiles at serve time.
- In a frame Dynamo gave up on during capture (`PrecompileSummary.eager_fallbacks`), a
  miss runs eagerly, as that frame did during capture.
- `load(..., allow_serve_time_compiles=True)` compiles a miss with the capture's backend
  and options, as `torch.compile` would. Only an *installed* artifact
  (`PrecompiledRunnable.installed`) has a compiler behind it; passing the flag for a
  standalone artifact raises.
- The `"force_eager"` and `"eager_on_recompile"` stances of
  {func}`torch.compiler.set_stance` also change serving; see
  {func}`torch.compiler.precompile.load`.

Because a miss surfaces in the serving process, the capture refuses, when it writes the
artifact, one it already knows has a gap or an ambiguity. Two `DynamoTracer` gates, both
on by default, raise `PrecompileError` instead of writing:

| Gate | Refuses to write when | Turning it off means |
|---|---|---|
| `require_complete` | a capture call raised, nothing compiled to a graph, a frame ran but produced no guarded code, or a frame was bypassed because its guards could not be serialized | The partial artifact is written, and `PrecompileSummary` lists the gaps. A frame that only dispatches to compiled children is harmless; a call that needs a missing frame is a miss. |
| `require_no_risky_drops` | a guard the serializer had to drop told captured variants apart, sits on a slot configuration chooses (an attribute, closure cell or aliased import holding a function), or was discarded by your `guard_filter_fn` | No load checks that slot, so a different value in the serving process silently selects the wrong variant instead of raising. Safe only when the capture and the serving process run the same code and configuration. |

Turning a gate off does not make serving more forgiving: it moves the failure from the
capture to a miss at serve time, or for a risky drop to a wrong result. Before turning
one off, read the capture's `summary()`: `capture_errors`, `uncovered_frames` and
`bypassed` explain a `require_complete` refusal, and `risky_dropped_guards` a
`require_no_risky_drops` one.

% Rendered from the docstrings, so this reference cannot drift from the source.

```{eval-rst}
.. autofunction:: torch.compiler.precompile.capture

.. autofunction:: torch.compiler.precompile.load

.. autofunction:: torch.compiler.precompile.no_compilation

.. autoexception:: torch.compiler.PrecompileError

.. autoclass:: torch.compiler.precompile.DynamoTracer

.. autoclass:: torch.compiler.precompile.MakeFxTracer

.. autoclass:: torch.compiler.precompile.Capture
   :members: save

.. autoclass:: torch.compiler.precompile.PrecompiledRunnable
   :members: unload

.. autoclass:: torch.compiler.precompile.PrecompileSummary
   :members: complete, dropped_guard_types, kept_guard_types
```
