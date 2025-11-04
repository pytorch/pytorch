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

### Clarification: what autoload does — and what it doesn't

The autoload entry point (for example, ``torch_foo = torch_foo:_autoload``) causes Python to import your package and invoke the ``_autoload`` callable. The loader expects ``_autoload`` to return a module-like object. Importantly, the autoload mechanism itself does not implicitly modify the already-imported ``torch`` package object or attach the returned module as an attribute on ``torch``. If you want your backend to be available as ``torch.foo`` (or another attribute on ``torch``), your package must explicitly attach the module to the ``torch`` namespace.

Minimal example ``_autoload``

Below is a minimal ``_autoload`` that returns the implementation module and, when ``torch`` is already importable, attaches it as ``torch.foo``. Adapt module names to your package layout.

.. code-block:: python

    # torch_foo/__init__.py

    import importlib
    import types

    def _autoload(*args, **kwargs) -> types.ModuleType:
        """
        Entry point invoked by the autoload mechanism. Return a module-like object.

        Also attach the implementation module to `torch.foo` when `torch` is already
        importable. This makes the backend accessible via `import torch` followed by
        `torch.foo`.
        """
        # Import the actual implementation submodule (adjust name as needed).
        module = importlib.import_module("torch_foo.impl")  # e.g. the real backend code

        # Try to attach to the torch namespace for convenience.
        # Guard against environments where torch isn't importable yet.
        try:
            import torch
        except Exception:
            # If torch is not importable at the time of _autoload, simply return the module.
            # In typical autoload usage the caller is inside torch so torch will be importable.
            return module

        # Attach the loaded module under the chosen attribute on the torch package.
        # Choose a name that avoids colliding with existing attributes.
        setattr(torch, "foo", module)

        return module

Notes and edge cases

- The autoload entry point only calls your ``_autoload`` and expects a module-like object; it does not itself set attributes on ``torch``. Attaching the module to the ``torch`` namespace is an explicit step performed by your package (as shown above).
- If ``torch`` is not importable at the time ``_autoload`` runs, the attachment step will be skipped; the returned module is still usable by the caller that received it from the entry point. To make ``torch.foo`` available later, you can:
  - Attach the attribute at import-time of a module that is imported by ``torch`` (if you control that code path), or
  - Re-run a small attach helper when ``torch`` is imported (for example, a lightweight function that checks ``if hasattr(torch, "foo"): return`` then sets ``torch.foo``), but be careful to avoid import cycles.
- Avoid clobbering existing attributes on ``torch``. Pick a unique name (for example, ``torch.foo_backend`` or ``torch._foo_backend``) if appropriate.

See also

- Minimal reproducible example demonstrating the difference: https://github.com/pganssle-google/torch-backend-mwe

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
