Torch Library API
=================

The Torch Library API provides capabilities for extending PyTorch's core library
of operators with user-defined operators and data types. This is the primary
mechanism for registering custom C++ operators that can be called from both
Python and C++.

**When to use the Library API:**

- When creating custom operators for PyTorch
- When implementing backend-specific kernels (CPU, CUDA, etc.)
- When registering custom classes for use in TorchScript
- When extending PyTorch with new functionality

**Basic usage:**

.. code-block:: cpp

   #include <torch/library.h>

   // Define a custom operator
   torch::Tensor my_add(const torch::Tensor& a, const torch::Tensor& b) {
       return a + b;
   }

   // Register the operator
   TORCH_LIBRARY(myops, m) {
       m.def("add(Tensor a, Tensor b) -> Tensor", &my_add);
   }

   // Use from C++
   auto result = torch::dispatcher::call("myops::add", tensor_a, tensor_b);

For a tutorial-style introduction to the library API, check out the
`Extending TorchScript with Custom C++ Operators
<https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html>`_
tutorial.

Header Files
------------

- ``torch/library.h`` - Main library API header
- ``torch/custom_class.h`` - Custom class registration

Library API Categories
----------------------

.. toctree::
   :maxdepth: 1

   registration
   custom_classes
   versioning

See Also
--------

- :doc:`../stable/index` - For stable ABI operator registration
