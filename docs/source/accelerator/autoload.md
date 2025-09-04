## Autoload

The **Autoload** mechanism in PyTorch simplifies the integration of custom backends by enabling automatic discovery and initialization at runtime. This eliminates the need for explicit imports or manual initialization, allowing developers to seamlessly integrate new accelerators or backends into PyTorch.

This tutorial will guide you through the steps to enable and use the Autoload mechanism for a custom backend.

### Hook Initialization

The first step is to define an initialization hook for your backend. This hook will be automatically invoked by PyTorch during startup.

::::{tab-set-code}
```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/__init__.py
    :language: python
    :start-after: LITERALINCLUDE START: AUTOLOAD
    :end-before: LITERALINCLUDE END: AUTOLOAD
    :linenos:
```


::::

### Backend Registrition

To enable Autoload, you need to register the `_autoload` function as an entry point in your `setup.py` file.

The first step is to define an initialization hook for your backend. This hook will be automatically invoked by PyTorch during startup.

::::{tab-set}

:::{tab-item} Python

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/setup.py
    :language: python
    :start-after: LITERALINCLUDE START: SETUP
    :end-before: LITERALINCLUDE END: SETUP
    :linenos:
```

:::

::::

### Using Backend with Autoload

After defining the initialization hook and registering the entry point, build and install your backend. Now, we can use the new accelerators without explicitly importing it.

    ```Python
    >>> import torch
    >>> torch.tensor(1, device="openreg")
    tensor(1, device='openreg:0')
    ```
