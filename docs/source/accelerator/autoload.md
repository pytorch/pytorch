# Autoload Mechanism

The **Autoload** mechanism in PyTorch simplifies the integration of custom backends by enabling automatic discovery and initialization at runtime. This eliminates the need for explicit imports or manual initialization, allowing developers to seamlessly integrate new accelerators or backends into PyTorch.

## Background

The **Autoload Device Extension** proposal in PyTorch is centered on improving support for various hardware backend devices, especially those implemented as out-of-the-tree extensions (not part of PyTorch’s main codebase). Currently, users must manually import or load these device-specific extensions to use them, which complicates the experience and increases cognitive overhead.

In contrast, in-tree devices (devices officially supported within PyTorch) are seamlessly integrated—users don’t need extra imports or steps. The goal of autoloading is to make out-of-the-tree devices just as easy to use, so users can follow the standard PyTorch device programming model without explicit loading or code changes. This would allow existing PyTorch applications to run on new devices without any modification, making hardware support more user-friendly and reducing barriers to adoption.

For more information about the background of **Autoload**, please refer to its [RFC](https://github.com/pytorch/pytorch/issues/122468).

## Design

The core idea of **Autoload** is to Use Python’s plugin discovery (entry points) so PyTorch automatically loads out-of-tree device extensions when torch is imported—no explicit user import needed.

Here is how it works:
- PyTorch defines a plugin group: `torch.plugins`.
- Once import, `torch/__init__.py` discovers all entry points in that group via `importlib.metadata.entry_points(group='torch.plugins')` and loads each plugin (`plugin.load()`).
- A canonical plugin name is proposed (device_extension) that points to a small module/package (torch_plugins_device_extension).
- The vendor installs torch_plugins_device_extension into site-packages; its `__init__.py` imports and initializes the vendor’s actual device extension modules.

A hardware vendor should have the following responsibilities:
- Declares an entry point under the torch.plugins group (e.g., `device_extension = 'torch_plugins_device_extension'`).
- Provides torch_plugins_device_extension (or an equivalent module) that safely imports and initializes the vendor runtime/bridge at import time.
- For PrivateUse1-based devices, set up the “friendly” device name (e.g., **OpenReg**) and register needed dispatch keys/backends.

Therefore, for user experience, we notice that:
- No additional import lines (e.g., no `import torch_openreg` required).
- Existing code using standard device semantics (`to("openreg")` etc.) works unchanged.

Overall, such design leverages Python’s built-in plugin/entry-point mechanism to make device extensions autoload at torch import time, giving a simple, scalable, and vendor-driven path to “zero code-change” enablement for out-of-tree devices.

## Implementation

This tutorial will take **OpenReg** as a new out-of-the-tree device and guide you through the steps to enable and use the **Autoload** mechanism.

### Entry Point Setup

To enable **Autoload**, register the `_autoload` function as an entry point in `setup.py` file.

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

Define the initialization hook `_autoload` for backend initialization. This hook will be automatically invoked by PyTorch during startup.

::::{tab-set-code}
```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/__init__.py
    :language: python
    :start-after: LITERALINCLUDE START: AUTOLOAD
    :end-before: LITERALINCLUDE END: AUTOLOAD
    :linenos:
    :emphasize-lines: 10-12
```


::::

## Result

After setting up the entry point and backend, build and install your backend. Now, we can use the new accelerators without explicitly importing it.

```Python
>>> import torch
>>> torch.tensor(1, device="openreg")
tensor(1, device='openreg:0')
```
