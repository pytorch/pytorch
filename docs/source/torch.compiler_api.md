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
     allow_in_graph
     substitute_in_graph
     assume_constant_result
     list_backends
     disable
     set_stance
     set_enable_guard_collectives
     cudagraph_mark_step_begin
     is_compiling
     is_dynamo_compiling
     is_exporting
     skip_guard_on_inbuilt_nn_modules_unsafe
     skip_guard_on_all_nn_modules_unsafe
     keep_tensor_guards_unsafe
     skip_guard_on_globals_unsafe
     nested_compile_region
```
