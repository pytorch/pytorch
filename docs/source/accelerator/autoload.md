# Autoload Mechanism

The **Autoload** mechanism in PyTorch simplifies the integration of a custom backend by enabling automatic discovery and initialization at runtime. This eliminates the need for explicit imports or manual initialization, allowing developers to seamlessly integrate a new accelerator or backend into PyTorch.

## Background

The **Autoload Device Extension** proposal in PyTorch is centered on improving support for various hardware backend devices, especially those implemented as out-of-the-tree extensions (not part of PyTorch’s main codebase). Currently, users must manually import or load these device-specific extensions to use them, which complicates the experience and increases cognitive overhead.

In contrast, in-tree devices (devices officially supported within PyTorch) are seamlessly integrated—users don’t need extra imports or steps. The goal of autoloading is to make out-of-the-tree devices just as easy to use, so users can follow the standard PyTorch device programming model without explicit loading or code changes. This would allow existing PyTorch applications to run on new devices without any modification, making hardware support more user-friendly and reducing barriers to adoption.

For more information about the background of **Autoload**, please refer to its [RFC](https://github.com/pytorch/pytorch/issues/122468).

## Design

The core idea of **Autoload** is to Use Python’s plugin discovery (entry points) so PyTorch automatically loads out-of-tree device extensions when torch is imported—no explicit user import needed.

For more instructions of the design of **Autoload**, please refer to [**How it works**](https://docs.pytorch.org/tutorials/unstable/python_extension_autoload.html#how-it-works).

## Implementation

This tutorial will take **OpenReg** as a new out-of-the-tree device and guide you through the steps to enable and use the **Autoload** mechanism.

### Entry Point Setup

To enable **Autoload**, register the `_autoload` function as an entry point in [setup.py](https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/setup.py) file.

::::{tab-set}

:::{tab-item} Python

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/setup.py
    :language: python
    :start-after: LITERALINCLUDE START: SETUP
    :end-before: LITERALINCLUDE END: SETUP
    :linenos:
    :emphasize-lines: 9-13
```

:::

::::

### Backend Setup

Define the initialization hook `_autoload` for backend initialization in [torch_openreg](https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/__init__.py). This hook will be automatically invoked by PyTorch during startup.

::::{tab-set-code}

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/__init__.py
    :language: python
    :start-after: LITERALINCLUDE START: AUTOLOAD
    :end-before: LITERALINCLUDE END: AUTOLOAD
    :linenos:
```

::::

### Lazy Inductor Backend Initialization

Integrating with torch.compile typically means registering device-specific codegen
classes, decompositions, and passes with Inductor. Doing all of that inside `_autoload`
would make every user pay the cost at `import torch`, even those who never compile.
To defer it, define an `_inductor_backend_init` method on the device module class
passed to `torch._register_device_module`: a no-arg callable that Inductor invokes at
the first Inductor compilation, on every entry point (`torch.compile`, direct
`compile_fx()`, and AOTInductor), and again on later compilations until it has
registered the device. Until registration succeeds, backend-feature queries may
invoke the hook multiple times within one compilation, so the hook must be safe
and cheap to retry. Inductor looks it up with
a plain attribute lookup, so it works the same whether the device module is a class
(methods, as below) or a module object (attribute assignment).

```python
from torch._inductor.codegen.common import register_backend_for_device


class MyDeviceModule:
    # ... other device module APIs (is_available, device_count, ...)

    @staticmethod
    def _inductor_backend_init():
        # Runs on the first inductor compile, not at import time.
        from my_backend._inductor import MyCppWrapperCodegen, MyScheduling, MyWrapperCodegen

        register_backend_for_device(
            "my_device", MyScheduling, MyWrapperCodegen, MyCppWrapperCodegen
        )
        # May also register decompositions, device op overrides, and custom passes.


torch.utils.rename_privateuse1_backend("my_device")
torch._register_device_module("my_device", MyDeviceModule)
```

The hook must call `register_backend_for_device` itself; that registration is
what stops Inductor from invoking the hook again on later compilations. Concurrent
callers wait for an in-flight hook invocation to finish. Because the hook runs
under Dynamo's compile lock, it must not wait on another thread that may import the
same integration or invoke `torch.compile`, and it should not fork. If the hook
raises, the exception propagates
out of the compile, and the hook is invoked again on the next compile unless it
registered the device before raising. Note that the hook is not gated on the
compiled device: while the device is unregistered it fires on every Inductor
compile, whichever device that graph targets. On the `compile_fx` paths, the hook
runs before that compilation selects its decomposition table.

Device modules that instead expose `Scheduling`, `PythonWrapperCodegen`, and
`CppWrapperCodegen` attributes directly are still supported, but that form imports
the codegen classes eagerly at `import torch`. A device that is already registered
(for example eagerly from `_autoload`) skips the hook entirely.

## Result

After setting up the entry point and backend, build and install your backend. Now, we can use the new accelerator without explicitly importing it.

```{eval-rst}
.. grid:: 2

    .. grid-item-card:: :octicon:`terminal;1em;` Without Autoload

           >>> import torch
           >>> import torch_openreg
           >>> torch.tensor(1, device="openreg")
           tensor(1, device='openreg:0')

    .. grid-item-card:: :octicon:`terminal;1em;` With Autoload

           >>> import torch # Automatically import torch_openreg
           >>>
           >>> torch.tensor(1, device="openreg")
           tensor(1, device='openreg:0')
```
