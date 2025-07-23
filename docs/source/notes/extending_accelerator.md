# Extending PyTorch with New Accelerators

## Background

The PrivateUse1-based third-party device integration mechanism has become the official path for integrating new devices into PyTorch. Ensuring the usability of this mechanism is crucial for enriching the hardware ecosystem of PyTorch.

To assist third-party device developers in efficiently integrating new backends, this article introduces in detail the integration methods for typical PyTorch modules using a modular approach. It is accompanied by a streamlined code implementation from the official [torch_openreg][OpenReg URL] backend to help developers quickly get started while avoiding common pitfalls.

This document is suitable for the following readers:

* Developers who wish to integrate accelerator backends into PyTorch;
* Developers interested in the principles of typical PyTorch modules;

---

## Operator Registration

PyTorch provides multiple methods for operator registration and usage, both at the Python and C++ levels, along with a set of supporting tools to quickly locate issues and query information. The following sections detail the operator registration capabilities.

### Tools

#### Commands

PyTorch provides a set of commands prefixed with `torch._C._dispatch_` around its Dispatch feature. You can query all related interfaces using the following command:

```Shell
python -c 'import torch; print("\n".join([x for x in dir(torch._C) if x.startswith("_dispatch_")]))'

...
_dispatch_dump
_dispatch_dump_table
_dispatch_has_kernel
_dispatch_has_kernel_for_any_dispatch_key
_dispatch_has_kernel_for_dispatch_key
_dispatch_isTensorSubclassLike
_dispatch_is_alias_key
_dispatch_is_included_in_alias
_dispatch_is_main_interpreter
_dispatch_kernel_for_dispatch_key_is_fallthrough
_dispatch_key_for_device
_dispatch_key_name
_dispatch_key_parse
_dispatch_key_set
...
```

Here are explanations for several commonly used commands:

* `torch._C._dispatch_key_set`:

    Displays the DispatchKey of the current Tensor, with priority increasing from left to right.

    ```Python
    >>> import torch
    >>> a = torch.randn(3,3,device="cuda")
    >>> torch._C._dispatch_key_set(a)
    'DispatchKeySet(CUDA, ADInplaceOrView, AutogradCUDA, AutocastCUDA)'
    ```

* `torch._C._dispatch_dump_table`:

    Queries the support status of a given operator across different Dispatch Keys, making it easy to locate the corresponding implementation code.

    ```Python
    >>> import torch
    >>> print(torch._C._dispatch_dump_table("aten::add.Tensor"))
    >>> ...
        CPU: registered at ./build/aten/src/ATen/RegisterCPU_0.cpp:1309 [kernel]
        CUDA: registered at ./build/aten/src/ATen/RegisterCUDA_0.cpp:2420 [kernel]
        HIP: registered at ./build/aten/src/ATen/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:1373 [default backend kernel]
        MPS: registered at ./build/aten/src/ATen/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:1373 [default backend kernel]
        IPU: registered at ./build/aten/src/ATen/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:1373 [default backend kernel]
        XPU: registered at ./build/aten/src/ATen/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:1373 [default backend kernel]
        HPU: registered at ./build/aten/src/ATen/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:1373 [default backend kernel]
        VE: registered at ./build/aten/src/ATen/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:1373 [default backend kernel]
        MTIA: registered at ./build/aten/src/ATen/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:1373 [default backend kernel]
        MAIA: registered at ./build/aten/src/ATen/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:1373 [default backend kernel]
        PrivateUse1: registered at ./build/aten/src/ATen/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:1373 [default backend kernel]
        ...
    ```

#### Environment Variables

PyTorch also provides some Dispatcher-related environment variables that can help with learning and quickly locating issues.

* TORCH_SHOW_DISPATCH_TRACE

    Displays detailed internal dispatch key scheduling during PyTorch execution.

    ```Bash
    export TORCH_SHOW_DISPATCH_TRACE=1
    ```

    ```Python
    >>> import torch
    >>> a = torch.randn(3,3)
      [call] op=[aten::randn], key=[BackendSelect]
       [redispatch] op=[aten::randn], key=[CPU]
        [call] op=[aten::empty.memory_format], key=[BackendSelect]
         [redispatch] op=[aten::empty.memory_format], key=[CPU]
        [call] op=[aten::normal_], key=[CPU]
    ```

### Registration

::::{tabs}

:::{tab} C++

1. Scenario One

   This is the most common operator implementation scenario. PyTorch comes with many built-in operators, defining their namespace (mainly in `aten` and `c10d`), schema, and concrete implementations for backends like CPU and CUDA. Our task is to provide the corresponding implementations for new devices for these built-in operators.

    ```{eval-rst}
    .. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/OpenRegMinimal.cpp
        :language: c++
        :start-after: LITERALINCLUDE START: EMPTY.MEMORY_FORMAT
        :end-before: LITERALINCLUDE END: EMPTY.MEMORY_FORMAT
    ```

    ```{eval-rst}
    .. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/OpenRegMinimal.cpp
        :language: c++
        :start-after: LITERALINCLUDE START: TORCH_LIBRARY_IMPL
        :end-before:  LITERALINCLUDE END: TORCH_LIBRARY_IMPL
        :emphasize-lines: 2
        :linenos:
    ```

    This registers the `wrapper_empty_memory_format` implementation for the new device to the `aten::empty.memory_format` operator on the `PrivateUse1 DispatchKey`.

2. Scenario Two

    For built-in PyTorch operators, besides the registration method in Scenario One, a `STUB` registration method is also supported. Essentially, this approach is based on Scenario One but with added flexibility to enhance code reuse across devices or to enable further dispatching at other granularities (e.g., CPU feature capabilities).

    ```{eval-rst}
    .. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/OpenRegExtra.cpp
        :language: c++
        :start-after: LITERALINCLUDE START: STUB
        :end-before: LITERALINCLUDE END: STUB
        :linenos:
    ```

    ```{todo}
    List of operators that can be registered via `STUB`
    ```

3. Scenario Three

    Besides providing built-in operator definitions, PyTorch also supports user-defined operators, generally in two forms:

    * Adding custom operators to a new namespace:

    ```{todo}
    TODO(including forward and backward)
    ```

    * Extending existing namespaces with custom operators:

    ```{todo}
    TODO(including forward and backward)
    ```

4. Scenario Four

    In addition to separately registering forward and backward functions to `PrivateUse1` and `AutogradPrivateUse1` DispatchKeys, PyTorch also supports a more convenient option using `torch.autograd.Function`.

    ```{eval-rst}
    .. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native/Extra.cpp
        :language: c++
        :start-after: LITERALINCLUDE START: TORCH.AUTOGRAD.FUNCTION Part1
        :end-before: LITERALINCLUDE END: TORCH.AUTOGRAD.FUNCTION Part1
        :linenos:
    ```

    ```{eval-rst}
    .. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native/Extra.cpp
        :language: c++
        :start-after: LITERALINCLUDE START: TORCH.AUTOGRAD.FUNCTION Part2
        :end-before: LITERALINCLUDE END: TORCH.AUTOGRAD.FUNCTION Part2
        :linenos:
    ```

    ```{eval-rst}
    .. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/OpenRegExtra.cpp
        :language: c++
        :start-after: LITERALINCLUDE START: TORCH.AUTOGRAD.FUNCTION
        :end-before: LITERALINCLUDE END: TORCH.AUTOGRAD.FUNCTION
        :emphasize-lines: 2,7
        :linenos:
    ```

5. Scenario Five

    PyTorch provides a fallback mechanism that allows unsupported operators to fall back to CPU execution. This is crucial for in-development accelerator backends to ensure functional correctness at the cost of performance.

    * Per-operator fallback

    ```{todo}
    TODO
    ```

    * Global fallback

    ```{eval-rst}
    .. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/OpenRegMinimal.cpp
        :language: c++
        :start-after: LITERALINCLUDE START: FALLBACK GLOBAL
        :end-before: LITERALINCLUDE END: FALLBACK GLOBAL
        :linenos:
    ```

    This enables global fallback so all unimplemented operators on the new backend will default to CPU execution.

6. Scenario Six

    ```{todo}
    * Meta registration
    * Overriding default implementations
    * Fallthrough
    * ATen operator set
    * ...
    ```

:::

:::{tab} Python

TODO

:::

::::

### Minimum set of operators to support

To help developers better prioritize their work, we provide a minimal set of operators. Implementing these operators ensures basic operator functionality is available.

| Operator Name                      | Dispatch Key | Description                       |
| :---:                              | :---:        | :---:                             |
| empty.memory_format                | PrivateUse1  |                                   |
| empty_strided                      | PrivateUse1  |                                   |
| as_strided                         | PrivateUse1  |                                   |
| resize_                            | PrivateUse1  |                                   |
| _reshape_alias                     | PrivateUse1  |                                   |
| _copy_from                         | PrivateUse1  |                                   |
| _copy_from_and_resize              | PrivateUse1  |                                   |
| _local_scalar_dense                | PrivateUse1  |                                   |
| set_.source_Tensor                 | PrivateUse1  |                                   |
| set_.source_Storage                | PrivateUse1  |                                   |
| set_.source_Storage_storage_offset | PrivateUse1  |                                   |
| view                               | PrivateUse1  |                                   |
| fallback                           | PrivateUse1  |                                   |

```{todo}
Add/remove operators above to ensure the minimal set list is reliable and accurate
```

## Autocast

## Autoload

## Memory Management

## Custom Storage

## ...

[OpenReg URL]: https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg "OpenReg URL"
