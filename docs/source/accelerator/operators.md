# Operator Registration

For new accelerators, one of the most important and fundamental aspects of integration is supporting high-performance operators. To facilitate operator adaptation for users and accelerator developers, PyTorch provides multiple methods for developing and registering operators in both `Python` and `C++`. The following sections detail some of PyTorch's fundamental capabilities for operator registration.

```{note}
`Dispatch Key` is used to uniquely identify accelerator within PyTorch, such as `CPU`, `CUDA`, `MPS`, and `PrivateUse1`. In theory, all subsequent new accelerators will share `PrivateUse1`, leveraging its built-in comprehensive scaffolding capabilities to complete the integration of new accelerators. Please refer to [Let's talk about the PyTorch dispatcher](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/) if you are interested with dispatcher.
```

(operator-set)=

## Operator Set

PyTorch currently has over 3500 built-in operators (including related operator variants). This represents a significant workload from any perspective, and supporting this massive number of operators in a short period of time is no easy task. Therefore, as the first step in developing new backend operators, our goal should be to focus on the essential operators. For other operators, we can first use the community's fallback mechanism to support the feature as the first priority. After that, we can gradually complete other operators to improve the performance of the new backend.

The required operator set is listed below, primarily consisting of low-level operators required by factory functions and fallback operators:

| Operator Name                      | Dispatch Key | Description                                                                                                        |
| :---:                              | :---:        | :---:                                                                                                              |
| empty.memory_format                | PrivateUse1  | Create an uninitialized Tensor with the specified shape and memory layout (the stride is automatically calculated) |
| empty_strided                      | PrivateUse1  | Create an uninitialized Tensor of the specified shape and stride (more degrees of freedom)                         |
| as_strided                         | PrivateUse1  | Create a shared view of the input Tensor with new shape, stride, and offset (without allocating new memory)        |
| view                               | PrivateUse1  | Create a shared view of the input Tensor with new shape, but the original Tensor must be memory-contiguous         |
| _reshape_alias                     | PrivateUse1  | Creates a shared view without safety checks(Internal version of reshape)                                           |
| resize_                            | PrivateUse1  | Modify the shape of the Tensor in place and reallocate memory if capacity is insufficient                          |
| _copy_from                         | PrivateUse1  | The underlying core function of Tensor.copy_ is responsible for the actual cross-device data copying               |
| _copy_from_and_resize              | PrivateUse1  | Combine `resize_` and `_copy_from` to resize first and then copy                                                   |
| _local_scalar_dense                | PrivateUse1  | The underlying implementation of `.item()`, extracting values from Tensor to CPU scalars                           |
| set_.source_Tensor                 | PrivateUse1  | Set the current Tensor using the specified Tensor                                                                  |
| set_.source_Storage                | PrivateUse1  | Set the current Tensor using the specified Storage                                                                 |
| set_.source_Storage_storage_offset | PrivateUse1  | Set the current Tensor using the specified Storage with the storage offset                                         |
| fallback                           | PrivateUse1  | Fallback to CPU                                                                                                    |

## Basics

Now that we have defined the initial scope of operator support, we can begin developing operator adaptations. This section will explain these implementations in `Python` and `C++` based on actual scenarios.

(step-one)=

### Step 1

{ref}`The operators mentioned above <operator-set>` share a common characteristic: They are built-in PyTorch operators with defined `namespaces` and `Schemas`, and these operators' built-in accelerators (`CPU`, `CUDA`, etc.) have been implemented. What we have to do next is to implement these operators for the new accelerators.

::::{tab-set}

:::{tab-item} C++

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native/Minimal.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: EMPTY.MEMORY_FORMAT IMPL
    :end-before: LITERALINCLUDE END: EMPTY.MEMORY_FORMAT IMPL
    :linenos:

.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/OpenRegMinimal.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: EMPTY.MEMORY_FORMAT WRAPPER
    :end-before: LITERALINCLUDE END: EMPTY.MEMORY_FORMAT WRAPPER
    :linenos:
```

:::

::::

Taking the `empty.memory_format` operator as an example, we first need to query the operator's `schema` information in `native_functions.yaml`, which contains detailed signature information. Then, we can implement the operator based on the capabilities of the new accelerator.

```Yaml
- func: empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
dispatch:
    CPU: empty_cpu
    CUDA: empty_cuda
    ...
```

::::{tab-set-code}

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/OpenRegMinimal.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: TORCH_LIBRARY_IMPL DEFAULT
    :end-before:  LITERALINCLUDE END: TORCH_LIBRARY_IMPL DEFAULT
    :emphasize-lines: 1,2
    :linenos:
```

::::

After completing the `wrapper_empty_memory_format`, we can register `aten::empty.memory_format` for `PrivateUse1` through `TORCH_LIBRARY_IMPL`.

### Step 2

By following {ref}`Step 1<step-one>`, we can complete the development and registration of all operators except `fallback`. Next, to support operators related to operations (such as mathematical operations and convolution operations), we need to implement the registration of fallback semantics. This is a built-in capability provided by the PyTorch framework that can fallback some operations that are not supported by new accelerators to the CPU for execution. For new backends in development, this is an extremely effective way to ensure functionality at the expense of performance.

::::{tab-set}

:::{tab-item} C++

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native/Minimal.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: FALLBACK IMPL
    :end-before: LITERALINCLUDE END: FALLBACK IMPL
    :emphasize-lines: 15
    :linenos:

.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/OpenRegMinimal.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: FALLBACK WRAPPER
    :end-before: LITERALINCLUDE END: FALLBACK WRAPPER
    :linenos:

.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/OpenRegMinimal.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: FALLBACK GLOBAL
    :end-before: LITERALINCLUDE END: FALLBACK GLOBAL
    :linenos:
```

:::

::::

`wrapper_cpu_fallback` wraps the `at::native::cpu_fallback` method provided by PyTorch and is registered with `PrivateUse1` in PyTorch via `TORCH_LIBRARY_IMPL`. Subsequent operations not supported by the new backend will automatically fall back to the CPU for execution, and the results will be passed back to the new backend after execution.

## Advanced

### Selective Fallback

Enabling the fallback mechanism only for certain operators, while following PyTorch's default behavior for other operators (an error will be reported if the accelerator does not have a corresponding operator implementation), this is a very reasonable scenario as well.

::::{tab-set}

:::{tab-item} C++

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/OpenRegMinimal.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: FALLBACK WRAPPER
    :end-before: LITERALINCLUDE END: FALLBACK WRAPPER
    :linenos:

.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/OpenRegMinimal.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: FALLBACK SINGLE
    :end-before: LITERALINCLUDE END: FALLBACK SINGLE
    :linenos:
```

:::

::::

Per-operator fallbacks are very similar to global fallbacks, the only difference being the registration method: calling `m.impl` registers an implementation for a specific operator, while `m.fallback` registers a default implementation for all operators.

::::{tab-set-code}

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native/Minimal.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: FALLBACK IMPL
    :end-before: LITERALINCLUDE END: FALLBACK IMPL
    :emphasize-lines: 2-5
    :linenos:
```

::::

Of course, global fallbacks can also be combined with a blacklist of fallbacks, which is a common approach, especially when only a few operators do not support fallbacks.

### PyTorch STUB

PyTorch also provides another approach for built-in operators: `STUB`. This method is essentially based on the {ref}`Step 1<step-one>` approach, but adds secondary scheduling capabilities (for example, scheduling based on CPU characteristics).

```{note}
The `STUB` method currently supports only a limited set of operators. For new accelerator devices, the advantage of the `STUB` method is that it significantly reduces the cost of development at the cost of a small performance overhead. PyTorch currently does not clearly list the set of operators that can be registered through `STUB`. Due to the large number of related operators, only the query method for the supported operator list is provided here.
```

```shell
pushd ${TORCH_ROOT}

find aten -type f -a -name "*.h" | xargs -I {} grep -wl "^DECLARE_DISPATCH" {}

popd
```

`DECLARE_DISPATCH` is a macro used to explicitly declare `STUB`. It is currently distributed in the `aten` directory. Based on this macro, you can find all operators that can be integrated using the `STUB` method.

```text
...
aten/src/ATen/native/Activation.h
aten/src/ATen/native/FusedSGD.h
aten/src/ATen/native/nested/NestedTensorBinaryOps.h
aten/src/ATen/native/TensorCompare.h
aten/src/ATen/native/Sorting.h
...
```

```c++
using unary_fn = void(*)(TensorIteratorBase&);

DECLARE_DISPATCH(unary_fn, abs_stub)
```

The above listing contains the file that declares the `STUB` operator, where you can clearly see the STUB name and the associated function signature. Next, we will take `abs_stub` as an example to briefly introduce the path to support operators through `STUB`.

::::{tab-set}

:::{tab-item} C++

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native/Extra.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: STUB ABS
    :end-before: LITERALINCLUDE END: STUB ABS
    :linenos:
```

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/OpenRegExtra.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: STUB DEFAULT
    :end-before: LITERALINCLUDE END: STUB DEFAULT
    :emphasize-lines: 1
    :linenos:
```

:::

::::

From the signature, we can see that the input of `abs_stub` is `TensorIteratorBase`, a powerful helper class provided by PyTorch that contains all input and output operators, as well as some other auxiliary methods. Based on it, we can develop the `abs_kernel` operator and then call `REGISTER_PRIVATEUSE1_DISPATCH` to specify `abs_stub` to complete the registration.

### Custom Operators

In addition to PyTorch's built-in operators, custom accelerator operators are also very common to improve performance in specific scenarios. These can be categorized into three main approaches:

* Forward-only
* Forward and backward: Separate registration
* Forward and backward: Implemented using `torch.autograd.Function`

```{note}
There are more details in PyTorch tutorials, so refer to [PyTorch Custom Operators](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html) if you are interested.
```

#### Forward Only

Here, we'll briefly introduce the implementation process of custom operators, focusing on the forward-only approach. The implementation can be summarized into the following three points:

1. **Define Schema:**

    ::::{tab-set}

    :::{tab-item} C++

    ```{eval-rst}
    .. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/OpenRegExtra.cpp
        :language: c++
        :start-after: LITERALINCLUDE START: CUSTOM OPERATOR SCHEMA
        :end-before: LITERALINCLUDE END: CUSTOM OPERATOR SCHEMA
        :emphasize-lines: 2
        :linenos:
    ```

    :::

    ::::

    * Namespace Name: `openreg`
    * Function Name: `custom_abs`
    * Input Parameters:
        * Type: `Tensor`
        * Name: `input`
    * Output Type: `Tensor`

2. **Register Operator&Autograd Fallback:**

    ::::{tab-set}

    :::{tab-item} C++

    ```{eval-rst}
    .. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/OpenRegExtra.cpp
        :language: c++
        :start-after: LITERALINCLUDE START: CUSTOM OPERATOR DEFAULT
        :end-before: LITERALINCLUDE END: CUSTOM OPERATOR DEFAULT
        :linenos:

    .. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/OpenRegExtra.cpp
        :language: c++
        :start-after: LITERALINCLUDE START: CUSTOM OPERATOR FALLBACK
        :end-before: LITERALINCLUDE END: CUSTOM OPERATOR FALLBACK
        :emphasize-lines: 2
        :linenos:
    ```

    :::

    ::::

    Use `TORCH_LIBRARY_IMPL` to register the `wrapper_custom_abs` implementation for the `custom_abs` operator in `PrivateUse1`. However, because `Autograd` is always enabled in PyTorch, PyTorch defaults to finding and executing the corresponding backward implementation even if only forward computation is required(will fallthrough in backward implementation). Therefore, we also need to register the corresponding implementation for `AutogradPrivateUse1` of the `custom_abs` operator. Fortunately, PyTorch also provides a general `Autograd Fallback` mechanism named `torch::autograd::autogradNotImplementedFallback`, if only forward computation is involved, it is equivalent to a fallthrough operation, selecting the next DispatchKey for computation; if backward computation is involved, an error is thrown.

3. **Register Metadata(optional, but required by the graph mode, etc.):**

    ::::{tab-set-code}

    ```{eval-rst}
    .. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/openreg/meta.py
        :language: python
        :start-after: LITERALINCLUDE START: CUSTOM OPERATOR META
        :end-before: LITERALINCLUDE END: CUSTOM OPERATOR META
        :linenos:
    ```

    ::::

    PyTorch supports registering `Meta` in both C++ and Python. Since Python registration is simpler, Python is used as an example here. Similar to the `TORCH_LIBRARY_IMPL` function in C++, Python provides the more user-friendly `torch.library.impl` decorator.

## Tools

Operator registration in PyTorch is complex, with diverse registration methods and numerous scenarios. Therefore, the PyTorch community has provided a number of tools to help developers quickly understand the underlying principles and assist in troubleshooting. Here we briefly introduce several commonly used tools:

### Commands

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

    You can easily query the corresponding implementation of the `aten::add.Tensor` operator on other platforms, so that you can track the entire operator calling process from the source code level.

### Environment Variables

PyTorch also provides some dispatcher-related environment variables that can help with learning and quickly locating issues.

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

    You can clearly see all the underlying operators called by Python-level operators within PyTorch: including the operator name, calling hierarchy, and corresponding `Dispatch Key`.
