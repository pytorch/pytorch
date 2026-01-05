Torch Stable API
================

The PyTorch Stable C++ API provides a convenient high level interface to call
ABI-stable tensor operations and other utilities commonly used in custom operators.
These functions are designed to maintain binary compatibility across PyTorch versions,
making them suitable for use in ahead-of-time compiled code.

For more information on the stable ABI, see the
`Stable ABI notes <https://docs.pytorch.org/docs/stable/notes/libtorch_stable_abi.html>`_.

Library Registration Macros
---------------------------

These macros provide stable ABI equivalents of the standard PyTorch operator
registration macros (``TORCH_LIBRARY``, ``TORCH_LIBRARY_IMPL``, etc.).
Use these when building custom operators that need to maintain binary
compatibility across PyTorch versions.

``STABLE_TORCH_LIBRARY(ns, m)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Defines a library of operators in a namespace using the stable ABI.

This is the stable ABI equivalent of :c:macro:`TORCH_LIBRARY`.
Use this macro to define operator schemas that will maintain
binary compatibility across PyTorch versions. Only one ``STABLE_TORCH_LIBRARY``
block can exist per namespace; use ``STABLE_TORCH_LIBRARY_FRAGMENT`` for
additional definitions in the same namespace from different translation units.

**Parameters:**

- ``ns`` - The namespace in which to define operators (e.g., ``mylib``).
- ``m`` - The name of the StableLibrary variable available in the block.

**Example:**

.. code-block:: cpp

   STABLE_TORCH_LIBRARY(mylib, m) {
       m.def("my_op(Tensor input, int size) -> Tensor");
       m.def("another_op(Tensor a, Tensor b) -> Tensor");
   }

Minimum compatible version: PyTorch 2.9.

``STABLE_TORCH_LIBRARY_IMPL(ns, k, m)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Registers operator implementations for a specific dispatch key using the stable ABI.

This is the stable ABI equivalent of ``TORCH_LIBRARY_IMPL``. Use this macro
to provide implementations of operators for a specific dispatch key (e.g.,
CPU, CUDA) while maintaining binary compatibility across PyTorch versions.

.. note::

   All kernel functions registered with this macro must be boxed using
   the ``TORCH_BOX`` macro.

**Parameters:**

- ``ns`` - The namespace in which the operators are defined.
- ``k`` - The dispatch key (e.g., ``CPU``, ``CUDA``).
- ``m`` - The name of the StableLibrary variable available in the block.

**Example:**

.. code-block:: cpp

   STABLE_TORCH_LIBRARY_IMPL(mylib, CPU, m) {
       m.impl("my_op", TORCH_BOX(&my_cpu_kernel));
   }

   STABLE_TORCH_LIBRARY_IMPL(mylib, CUDA, m) {
       m.impl("my_op", TORCH_BOX(&my_cuda_kernel));
   }

Minimum compatible version: PyTorch 2.9.

``STABLE_TORCH_LIBRARY_FRAGMENT(ns, m)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extends operator definitions in an existing namespace using the stable ABI.

This is the stable ABI equivalent of ``TORCH_LIBRARY_FRAGMENT``. Use this macro
to add additional operator definitions to a namespace that was already
created with ``STABLE_TORCH_LIBRARY``.

**Parameters:**

- ``ns`` - The namespace to extend.
- ``m`` - The name of the StableLibrary variable available in the block.

Minimum compatible version: PyTorch 2.9.

``TORCH_BOX(&func)``
^^^^^^^^^^^^^^^^^^^

Wraps a function to conform to the stable boxed kernel calling convention.

This macro takes an unboxed kernel function pointer and generates a boxed wrapper
that can be registered with the stable library API.

**Parameters:**

- ``func`` - The unboxed kernel function to wrap.

**Example:**

.. code-block:: cpp

   Tensor my_kernel(const Tensor& input, int64_t size) {
       return input.reshape({size});
   }

   STABLE_TORCH_LIBRARY_IMPL(my_namespace, CPU, m) {
       m.impl("my_op", TORCH_BOX(&my_kernel));
   }

Minimum compatible version: PyTorch 2.9.

Tensor Class
------------

The ``torch::stable::Tensor`` class offers a user-friendly C++ interface similar
to ``torch::Tensor`` while maintaining binary compatibility across PyTorch versions.

.. doxygenclass:: torch::stable::Tensor
   :members:


Device Class
------------

The ``torch::stable::Device`` class provides a user-friendly C++ interface similar
to ``c10::Device`` while maintaining binary compatibility across PyTorch versions.
It represents a compute device (CPU, CUDA, etc.) with an optional device index.

.. doxygenclass:: torch::stable::Device
   :members:

DeviceGuard Class
-----------------

The ``torch::stable::accelerator::DeviceGuard`` provides a user-friendly C++
interface similar to ``c10::DeviceGuard`` while maintaining binary compatibility
across PyTorch versions.

.. doxygenclass:: torch::stable::accelerator::DeviceGuard
   :members:

.. doxygenfunction:: torch::stable::accelerator::getCurrentDeviceIndex


Stream Utilities
----------------

For CUDA stream access, we currently recommend the ABI stable C shim API. This
will be improved in a future release with a more ergonomic wrapper.

Getting the Current CUDA Stream
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To obtain the current ``cudaStream_t`` for use in CUDA kernels:

.. code-block:: cpp

   #include <torch/csrc/inductor/aoti_torch/c/shim.h>
   #include <torch/headeronly/util/shim_utils.h>

   // For now, we rely on the ABI stable C shim API to get the current CUDA stream.
   // This will be improved in a future release.
   // When using a C shim API, we need to use TORCH_ERROR_CODE_CHECK to
   // check the error code and throw an appropriate runtime_error otherwise.
   void* stream_ptr = nullptr;
   TORCH_ERROR_CODE_CHECK(
       aoti_torch_get_current_cuda_stream(tensor.get_device_index(), &stream_ptr));
   cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

   // Now you can use 'stream' in your CUDA kernel launches
   my_kernel<<<blocks, threads, 0, stream>>>(args...);

.. note::

   The ``TORCH_ERROR_CODE_CHECK`` macro is required when using C shim APIs
   to properly check error codes and throw appropriate exceptions.

CUDA Error Checking Macros
--------------------------

These macros provide stable ABI equivalents for CUDA error checking.
They wrap CUDA API calls and kernel launches, providing detailed error
messages using PyTorch's error formatting.

``STD_CUDA_CHECK(EXPR)``
^^^^^^^^^^^^^^^^^^^^^^^^

Checks the result of a CUDA API call and throws an exception on error.
Users of this macro are expected to include ``cuda_runtime.h``.

**Example:**

.. code-block:: cpp

   STD_CUDA_CHECK(cudaMalloc(&ptr, size));
   STD_CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));

Minimum compatible version: PyTorch 2.10.

``STD_CUDA_KERNEL_LAUNCH_CHECK()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Checks for errors from the most recent CUDA kernel launch. Equivalent to
``STD_CUDA_CHECK(cudaGetLastError())``.

**Example:**

.. code-block:: cpp

   my_kernel<<<blocks, threads, 0, stream>>>(args...);
   STD_CUDA_KERNEL_LAUNCH_CHECK();

Minimum compatible version: PyTorch 2.10.

Header-Only Utilities
---------------------

The ``torch::headeronly`` namespace provides header-only versions of common
PyTorch types and utilities. These can be used without linking against libtorch,
making them ideal for maintaining binary compatibility across PyTorch versions.

Error Checking
^^^^^^^^^^^^^^

``STD_TORCH_CHECK`` is a header-only macro for runtime assertions:

.. code-block:: cpp

   #include <torch/headeronly/util/Exception.h>

   STD_TORCH_CHECK(condition, "Error message with ", variable, " interpolation");

Core Types
^^^^^^^^^^

The following ``c10::`` types are available as header-only versions under
``torch::headeronly::``:

- ``torch::headeronly::ScalarType`` - Tensor data types (Float, Double, Int, etc.)
- ``torch::headeronly::DeviceType`` - Device types (CPU, CUDA, etc.)
- ``torch::headeronly::MemoryFormat`` - Memory layout formats (Contiguous, ChannelsLast, etc.)
- ``torch::headeronly::Layout`` - Tensor layouts (Strided, Sparse, etc.)

.. code-block:: cpp

   #include <torch/headeronly/core/ScalarType.h>
   #include <torch/headeronly/core/DeviceType.h>
   #include <torch/headeronly/core/MemoryFormat.h>
   #include <torch/headeronly/core/Layout.h>

   auto dtype = torch::headeronly::ScalarType::Float;
   auto device_type = torch::headeronly::DeviceType::CUDA;
   auto memory_format = torch::headeronly::MemoryFormat::Contiguous;
   auto layout = torch::headeronly::Layout::Strided;

TensorAccessor
^^^^^^^^^^^^^^

``TensorAccessor`` provides efficient, bounds-checked access to tensor data.
You can construct one from a stable tensor's data pointer, sizes, and strides:

.. code-block:: cpp

   #include <torch/headeronly/core/TensorAccessor.h>

   // Create a TensorAccessor for a 2D float tensor
   auto sizes = tensor.sizes();
   auto strides = tensor.strides();
   torch::headeronly::TensorAccessor<float, 2> accessor(
       static_cast<float*>(tensor.mutable_data_ptr()),
       sizes.data(),
       strides.data());

   // Access elements
   float value = accessor[i][j];

Dispatch Macros
^^^^^^^^^^^^^^^

Header-only dispatch macros (THO = Torch Header Only) are available for
dtype and device dispatching:

.. code-block:: cpp

   #include <torch/headeronly/core/Dispatch.h>

   THO_DISPATCH_FLOATING_TYPES(tensor.scalar_type(), "my_kernel", [&] {
       // scalar_t is the resolved type
       auto* data = tensor.data_ptr<scalar_t>();
   });

Full API List
^^^^^^^^^^^^^

For the complete list of header-only APIs, see ``torch/header_only_apis.txt``
in the PyTorch source tree.

Stable Operators
----------------

Tensor Creation
^^^^^^^^^^^^^^^

.. doxygenfunction:: torch::stable::empty

.. doxygenfunction:: torch::stable::empty_like

.. doxygenfunction:: torch::stable::new_empty(const torch::stable::Tensor &self, torch::headeronly::IntHeaderOnlyArrayRef size, std::optional<torch::headeronly::ScalarType> dtype, std::optional<torch::headeronly::Layout> layout, std::optional<torch::stable::Device> device, std::optional<bool> pin_memory)

.. doxygenfunction:: torch::stable::new_zeros(const torch::stable::Tensor &self, torch::headeronly::IntHeaderOnlyArrayRef size, std::optional<torch::headeronly::ScalarType> dtype, std::optional<torch::headeronly::Layout> layout, std::optional<torch::stable::Device> device, std::optional<bool> pin_memory)

.. doxygenfunction:: torch::stable::full

.. doxygenfunction:: torch::stable::from_blob

Tensor Manipulation
^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: torch::stable::clone

.. doxygenfunction:: torch::stable::contiguous

.. doxygenfunction:: torch::stable::reshape

.. doxygenfunction:: torch::stable::view

.. doxygenfunction:: torch::stable::flatten

.. doxygenfunction:: torch::stable::squeeze

.. doxygenfunction:: torch::stable::unsqueeze

.. doxygenfunction:: torch::stable::transpose

.. doxygenfunction:: torch::stable::select

.. doxygenfunction:: torch::stable::narrow

.. doxygenfunction:: torch::stable::pad


Device and Type Conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: torch::stable::to(const torch::stable::Tensor &self, std::optional<torch::headeronly::ScalarType> dtype, std::optional<torch::headeronly::Layout> layout, std::optional<torch::stable::Device> device, std::optional<bool> pin_memory, bool non_blocking, bool copy, std::optional<torch::headeronly::MemoryFormat> memory_format)

.. doxygenfunction:: torch::stable::to(const torch::stable::Tensor &self, torch::stable::Device device, bool non_blocking, bool copy)

.. doxygenfunction:: torch::stable::fill_

.. doxygenfunction:: torch::stable::zero_

.. doxygenfunction:: torch::stable::copy_

.. doxygenfunction:: torch::stable::matmul

.. doxygenfunction:: torch::stable::amax(const torch::stable::Tensor &self, int64_t dim, bool keepdim)

.. doxygenfunction:: torch::stable::amax(const torch::stable::Tensor &self, torch::headeronly::IntHeaderOnlyArrayRef dims, bool keepdim)

.. doxygenfunction:: torch::stable::sum

.. doxygenfunction:: torch::stable::sum_out

.. doxygenfunction:: torch::stable::subtract

.. doxygenfunction:: torch::stable::parallel_for

.. doxygenfunction:: torch::stable::get_num_threads


Parallelization Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: torch::stable::parallel_for

.. doxygenfunction:: torch::stable::get_num_threads
