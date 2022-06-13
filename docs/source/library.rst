torch.library
===================================

Python operator registration API provides capabilities for extending PyTorch's core library
of operators with user defined operators. Currently, this can be done in two ways:

#. Creating new libraries
   * Lets you to register new operators and kernels for various backends and functionalities by specifying the appropriate dispatch keys.

   * These operators can then be accessed using the `torch.ops` API. So, a newly registered operator
     `add` in your namespace `foo` can be accessed by calling `torch.ops.foo.add`. You can also access
     specific registered overloads by calling `torch.ops.foo.add.{overload_name}`.

   * This can be done by creating Library class objects of `"DEF"`` kind.

#. Extending existing C++ libraries (e.g., aten)
   * Lets you register kernels for *existing* operators corresponding to various backends and functionalities by specifying the appropriate dispatch keys.

   * Fill up spotty operator support for a feature implemented through a dispatch key. For example.,
      * Add operator support for Meta Tensors (by registering function to the Meta dispatch key).

   * This can be done by creating Library class objects of `"IMPL"` kind.

A tutorial that walks you through some examples on how to use this API is available on `Google Colab <https://colab.research.google.com/drive/1RRhSfk7So3Cn02itzLWE9K4Fam-8U011?usp=sharing>`.

.. warning::
  Dispatcher is a complicated PyTorch concept and having a sound understanding of Dispatcher is crucial
  to be able to do anything advanced with this API. `This blog post <http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/>`
  is a good starting point to learn about Dispatcher.

.. currentmodule:: torch.library

.. autoclass:: torch.library.Library
  :members:

  .. automethod:: __init__
  .. automethod:: impl
    :noindex:
  .. automethod:: define
    :noindex:

We have also added some function decorators to make it convenient to register functions for operators:

:func:`torch.library.impl`
:func:`torch.library.define`
