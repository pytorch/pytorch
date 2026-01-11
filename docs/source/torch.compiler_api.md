```{eval-rst}
.. currentmodule:: torch.compiler
.. automodule:: torch.compiler
```

(torch.compiler_api)=
# torch.compiler API reference

> ⚠️ **Warning**
>
> `torch._dynamo.mark_dynamic()` must **NOT** be called inside a model’s
> `forward()` method when using `torch.compile()`.
>
> This function is a *tracing-time API*. Calling it inside compiled
> functions will raise an error such as:
>
> ```
> AssertionError: Attempt to trace forbidden callable
> ```
>
> **Correct usage** is to call `mark_dynamic` on input tensors
> **outside** the compiled model, or to use:
>
> ```python
> torch.compile(model, dynamic=True)
> ```

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
     keep_portable_guards_unsafe
     skip_guard_on_inbuilt_nn_modules_unsafe
     skip_guard_on_all_nn_modules_unsafe
     keep_tensor_guards_unsafe
     skip_guard_on_globals_unsafe
     skip_all_guards_unsafe
     nested_compile_region
```
