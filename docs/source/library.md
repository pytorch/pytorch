(torch-library-docs)=

# torch.library

```{eval-rst}
.. py:module:: torch.library
.. currentmodule:: torch.library
```

torch.library is a collection of APIs for extending PyTorch's core library
of operators. It contains utilities for testing custom operators, creating new
custom operators, and extending operators defined with PyTorch's C++ operator
registration APIs (e.g. aten operators).

For a detailed guide on effectively using these APIs, please see
[PyTorch Custom Operators Landing Page](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
for more details on how to effectively use these APIs.

## Testing custom ops

Use {func}`torch.library.opcheck` to test custom ops for incorrect usage of the
Python torch.library and/or C++ TORCH_LIBRARY APIs. Also, if your operator supports
training, use {func}`torch.autograd.gradcheck` to test that the gradients are
mathematically correct.

```{eval-rst}
.. autofunction:: opcheck
```

## Creating new custom ops in Python

Use {func}`torch.library.custom_op` to create new custom ops.

```{eval-rst}
.. autofunction:: custom_op
.. autofunction:: triton_op
.. autofunction:: wrap_triton
```

## Extending custom ops (created from Python or C++)

Use the `register.*` methods, such as {func}`torch.library.register_kernel` and
{func}`torch.library.register_fake`, to add implementations
for any operators (they may have been created using {func}`torch.library.custom_op` or
via PyTorch's C++ operator registration APIs).

```{eval-rst}
.. autofunction:: register_kernel
.. autofunction:: register_autocast
.. autofunction:: register_autograd
.. autofunction:: register_fake
.. autofunction:: register_vmap
.. autofunction:: impl_abstract
.. autofunction:: get_ctx
.. autofunction:: register_torch_dispatch
.. autofunction:: infer_schema
.. autoclass:: torch._library.custom_ops.CustomOpDef
   :members: set_kernel_enabled
```

## Low-level APIs

The following APIs are direct bindings to PyTorch's C++ low-level
operator registration APIs.

```{eval-rst}
.. warning:: The low-level operator registration APIs and the PyTorch Dispatcher are a complicated PyTorch concept. We recommend you use the higher level APIs above (that do not require a torch.library.Library object) when possible. `This blog post <http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/>`_ is a good starting point to learn about the PyTorch Dispatcher.
```

A tutorial that walks you through some examples on how to use this API is available on [Google Colab](https://colab.research.google.com/drive/1RRhSfk7So3Cn02itzLWE9K4Fam-8U011?usp=sharing).

```{eval-rst}
.. autoclass:: torch.library.Library
  :members:

.. autofunction:: fallthrough_kernel

.. autofunction:: define

.. autofunction:: impl
```
