Torch Library API
=================

The PyTorch C++ API allows users to add custom operators and data types to PyTorch's core library of operators. Extensions implemented
using the Torch Library API are made available for use in both the PyTorch eager
API as well as in TorchScript.

For a tutorial style introduction to the library API, check out the
`Extending TorchScript with Custom C++ Operators
<https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html>`_
tutorial.

Macros
------

.. doxygendefine:: TORCH_LIBRARY

.. doxygendefine:: TORCH_LIBRARY_IMPL

Classes
-------

.. doxygenclass:: torch::Library
  :members:

.. doxygenclass:: torch::CppFunction
  :members:

Functions
---------

.. doxygengroup:: torch-dispatch-overloads
  :content-only:

.. doxygengroup:: torch-schema-overloads
  :content-only:
