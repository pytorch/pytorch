Library Versioning
==================

PyTorch provides version number macros for identifying the version of LibTorch in use.

**Example:**

.. code-block:: cpp

   #include <torch/torch.h>
   #include <iostream>

   int main() {
     std::cout << "PyTorch version from parts: "
       << TORCH_VERSION_MAJOR << "."
       << TORCH_VERSION_MINOR << "."
       << TORCH_VERSION_PATCH << std::endl;
     std::cout << "PyTorch version: " << TORCH_VERSION << std::endl;
   }

This will output something like:

.. code-block:: text

   PyTorch version from parts: 1.8.0
   PyTorch version: 1.8.0

.. note::

   These macros are only available in PyTorch >= 1.8.0.

Version Macros
--------------

- ``TORCH_VERSION_MAJOR`` - Major version number
- ``TORCH_VERSION_MINOR`` - Minor version number
- ``TORCH_VERSION_PATCH`` - Patch version number
- ``TORCH_VERSION`` - Full version string (e.g., "1.8.0")
