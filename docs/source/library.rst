torch.library
===================================

Python operator registration API provides capabilities for extending PyTorch's core library
of operators with user defined operators. Currently, this can be done in two ways:

#. Creating new libraries

   * Lets you to register **new operators** and kernels for various backends and functionalities by specifying the appropriate dispatch keys. For example,

      * Consider registering a new operator ``add`` in your newly created namespace ``foo``. You can access this operator using the ``torch.ops`` API and calling into by calling ``torch.ops.foo.add``. You can also access specific registered overloads by calling ``torch.ops.foo.add.{overload_name}``.

      * If you registered a new kernel for the ``CUDA`` dispatch key for this operator, then your custom defined function will be called for CUDA tensor inputs.

   * This can be done by creating Library class objects of ``"DEF"`` kind.

#. Extending existing C++ libraries (e.g., aten)

   * Lets you register kernels for **existing operators** corresponding to various backends and functionalities by specifying the appropriate dispatch keys.

   * This may come in handy to fill up spotty operator support for a feature implemented through a dispatch key. For example.,

      * You can add operator support for Meta Tensors (by registering function to the ``Meta`` dispatch key).

   * This can be done by creating Library class objects of ``"IMPL"`` kind.

A tutorial that walks you through some examples on how to use this API is available on `Google Colab <https://colab.research.google.com/drive/1RRhSfk7So3Cn02itzLWE9K4Fam-8U011?usp=sharing>`_.

.. warning::
  Dispatcher is a complicated PyTorch concept and having a sound understanding of Dispatcher is crucial
  to be able to do anything advanced with this API. `This blog post <http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/>`_
  is a good starting point to learn about Dispatcher.

.. currentmodule:: torch.library

.. autoclass:: torch.library.Library
  :members:

.. autofunction:: fallthrough_kernel

We have also added some function decorators to make it convenient to register functions for operators:

* :func:`torch.library.impl`
* :func:`torch.library.define`
