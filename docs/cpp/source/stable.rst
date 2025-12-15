Torch Stable API
================

The PyTorch Stable C++ API provides ABI-stable interfaces for tensor operations.
These functions are designed to maintain binary compatibility across PyTorch versions,
making them suitable for use in ahead-of-time compiled code.

For more information on the stable ABI, see the
:doc:`Stable ABI notes </notes/libtorch_stable_abi>`.

2.9 Operations
--------------

.. doxygenfunction:: torch::stable::new_empty(const torch::stable::Tensor&, torch::headeronly::IntHeaderOnlyArrayRef, std::optional<torch::headeronly::ScalarType>)

2.10 Operations
---------------

.. doxygenfunction:: torch::stable::new_empty(const torch::stable::Tensor&, torch::headeronly::IntHeaderOnlyArrayRef, std::optional<torch::headeronly::ScalarType>, std::optional<torch::headeronly::Layout>, std::optional<torch::stable::Device>, std::optional<bool>)
