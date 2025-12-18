Torch Stable API
================

The PyTorch Stable C++ API provides ABI-stable interfaces for tensor operations.
These functions are designed to maintain binary compatibility across PyTorch versions,
making them suitable for use in ahead-of-time compiled code.

For more information on the stable ABI, see the
:doc:`Stable ABI notes </notes/libtorch_stable_abi>`.

new_empty
---------

.. doxygenfunction:: torch::stable::new_empty
