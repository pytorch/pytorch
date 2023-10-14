torch.library
===================================
.. py:module:: torch.library
.. currentmodule:: torch.library

torch.library is a collection of APIs for extending PyTorch's core library
of operators. It contains utilities for creating new custom operators as
well as extending existing C++ operators (e.g. aten ops).

Higher level APIs
-----------------

Use :func:`define` to define new custom operators. Use the impl methods, such
as :func:`impl_backend` and func:`impl_abstract`, to add implementations
for any operators (they may have been created using define or via PyTorch's C++
operator registration APIs).

.. autofunction:: define
.. autofunction:: impl_device
.. autofunction:: impl_abstract
.. autofunction:: get_ctx

Low-level APIs
--------------

The following APIs are direct bindings to PyTorch's C++ low-level
operator registration APIs.

.. warning::
   The low-level operator registration APIs and the PyTorch Dispatcher are a
   complicated PyTorch concept. We recommend you use the higher level APIs above
   when possible.
   This blog post <http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/>`_
   is a good starting point to learn about the PyTorch Dispatcher.

A tutorial that walks you through some examples on how to use this API is available on `Google Colab <https://colab.research.google.com/drive/1RRhSfk7So3Cn02itzLWE9K4Fam-8U011?usp=sharing>`_.

.. autoclass:: torch.library.Library
  :members:

.. autofunction:: impl
.. autofunction:: fallthrough_kernel
