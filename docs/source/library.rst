torch.library
===================================

Python operator registration API provides capabilities for extending PyTorch's core library
of operators with user defined operators. Currently, this can be done in two ways:

A tutorial that walks you through some examples on how to use this API is available on `Google Colab <https://colab.research.google.com/drive/1RRhSfk7So3Cn02itzLWE9K4Fam-8U011?usp=sharing>`.
1. Creating new libraries (kind = `DEF`)
   - Lets you to register new operators and kernels for various backends and functionalities
     by specifying the appropriate dispatch keys.
   - These operators can then be accessed using the `torch.ops` API. So, a newly registered operator
     `add` in your namespace `foo` can be accessed by calling `torch.ops.foo.add`. You can also access
     specific registered overloads by calling `torch.ops.foo.add.{overload_name}`.
   - This can be done by creating Library class objects of "DEF" kind.
2. Extending existing libraries (kind = `IMPL`)
   - Lets you register kernels for *existing* operators corresponding to various
     backends and functionalities by specifying the appropriate dispatch keys.
   - This allows you to override the functionality of existing operators as well as
     extend functionalities for existing operators. For example., ...
   - This can be done by creating Library class objects of "IMPL" kind.

Extensions implemented using the torch.library API are made available for use in both the PyTorch eager
API as well as in TorchScript.

.. currentmodule:: torch.library

.. autoclass:: torch.library.Library
   :members:

   .. automethod:: __init__
   .. automethod:: impl
   .. automethod:: define

We have also added some decorators to make it convenient to register functions for operators:

:func:`torch.library.impl`
:func:`torch.library.define`
