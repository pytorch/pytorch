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

% precompile is a module whose members are documented manually below (autosummary cannot
% render them under the parent module's currentmodule), one entry per name in
% ``torch.compiler.precompile.__all__``, each deliberately a one-line description.

```{eval-rst}
.. py:function:: precompile.capture(fn, /, *, artifact_path, cache_path, tracer=MakeFxTracer(), backend="inductor", training=False)

   Return a caller-driven capture of ``fn`` as a :class:`precompile.Capture`: enter it as a
   context manager and call it with the positional arguments ``fn`` takes inside the block;
   the ``(python_code, cache)`` artifact is written to ``artifact_path`` / ``cache_path``
   when the block exits cleanly having captured at least one call.

.. py:function:: precompile.load(artifact_path, cache_path, /)

   Reconstruct a runnable from the two files a capture wrote, as a
   :class:`torch.compiler.PrecompiledRunnable` with the same calling convention as the
   captured ``fn``. It executes ``python_code``, so treat both files as trusted input.

.. py:class:: precompile.MakeFxTracer(decompositions=None)

   The ``make_fx`` capture front-end, passed as ``tracer=`` to
   :func:`precompile.capture`: the default and the only one in this build, a non-strict
   single make_fx trace of one call. ``decompositions`` is an optional table forwarded to
   ``make_fx``.

.. py:class:: precompile.Capture

   The object :func:`precompile.capture` returns; it is not constructed directly.

   .. py:method:: save()

      Write everything captured so far to the two files without ending the capture.

.. autoexception:: torch.compiler.PrecompileError

.. autoclass:: torch.compiler.PrecompiledRunnable
   :members: unload
```
