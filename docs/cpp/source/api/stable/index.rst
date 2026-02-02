Torch Stable API
================

The PyTorch Stable C++ API provides a binary-compatible interface for calling
tensor operations and utilities that is guaranteed to remain stable across
PyTorch versions. This enables ahead-of-time compiled extensions that don't
need recompilation when PyTorch is updated.

**When to use the Stable API:**

- When building extensions that must work across multiple PyTorch versions
- When distributing pre-compiled binaries
- When binary compatibility is more important than access to the latest features
- When writing custom operators for production deployment

**Basic usage:**

.. code-block:: cpp

   #include <torch/csrc/stable/library.h>
   #include <torch/csrc/stable/ops.h>

   // Create a tensor using stable API
   auto tensor = torch::stable::empty(
       {3, 4},
       torch::headeronly::ScalarType::Float,
       torch::headeronly::Layout::Strided,
       torch::stable::Device(torch::headeronly::DeviceType::CPU),
       false,
       torch::headeronly::MemoryFormat::Contiguous);

   // Register operators with stable ABI
   STABLE_TORCH_LIBRARY(myops, m) {
       m.def("my_op(Tensor input) -> Tensor");
   }

   STABLE_TORCH_LIBRARY_IMPL(myops, CPU, m) {
       m.impl("my_op", TORCH_BOX(&my_cpu_kernel));
   }

For more information on the stable ABI, see the
`Stable ABI notes <https://docs.pytorch.org/docs/stable/notes/libtorch_stable_abi.html>`_.

Header Files
------------

- ``torch/csrc/stable/library.h`` - Stable library registration
- ``torch/csrc/stable/ops.h`` - Stable operator definitions
- ``torch/csrc/stable/tensor_struct.h`` - Stable tensor structures
- ``torch/csrc/stable/device_struct.h`` - Stable device structures
- ``torch/csrc/stable/accelerator.h`` - Accelerator support
- ``torch/csrc/stable/macros.h`` - Stable API macros

Stable API Categories
---------------------

.. toctree::
   :maxdepth: 1

   registration
   operators
   utilities

See Also
--------

- :doc:`../library/index` - For standard (non-stable) operator registration
- `Stable ABI documentation <https://pytorch.org/docs/stable/cpp_extension.html>`_
