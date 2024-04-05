torch.library
===================================
.. py:module:: torch.library
.. currentmodule:: torch.library

torch.library is a collection of APIs for extending PyTorch's core library
of operators. It contains utilities for creating new custom operators as
well as extending operators defined with PyTorch's C++ operator
registration APIs (e.g. aten operators).

For a detailed guide on effectively using these APIs, please see
`this gdoc <https://docs.google.com/document/d/1W--T6wz8IY8fOI0Vm8BF44PdBgs283QvpelJZWieQWQ/edit>`_

Creating new custom ops in Python
---------------------------------

Use :func:`torch.library.custom_op` to create new custom ops.

.. autofunction:: custom_op

Extending custom ops created from C++
-------------------------------------

Use the impl methods, such as :func:`torch.library.impl` and
func:`torch.library.impl_abstract`, to add implementations
for any operators (they may have been created using :func:`torch.library.custom_op` or
via PyTorch's C++ operator registration APIs).

.. autofunction:: impl
.. autofunction:: impl_abstract
.. autofunction:: get_ctx

Low-level APIs
--------------

The following APIs are direct bindings to PyTorch's C++ low-level
operator registration APIs.

.. warning::
   The low-level operator registration APIs and the PyTorch Dispatcher are a
   complicated PyTorch concept. We recommend you use the higher level APIs above
   (that do not require a torch.library.Library object) when possible.
   This blog post <http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/>`_
   is a good starting point to learn about the PyTorch Dispatcher.

A tutorial that walks you through some examples on how to use this API is available on `Google Colab <https://colab.research.google.com/drive/1RRhSfk7So3Cn02itzLWE9K4Fam-8U011?usp=sharing>`_.

.. autoclass:: torch.library.Library
  :members:

.. autofunction:: fallthrough_kernel

.. autofunction:: define
