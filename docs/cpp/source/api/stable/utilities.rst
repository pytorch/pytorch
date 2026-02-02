Utilities
=========

The stable API provides various utility functions and types for working with
tensors and CUDA operations.

DeviceGuard Class
-----------------

.. doxygenclass:: torch::stable::accelerator::DeviceGuard
   :members:
   :undoc-members:

.. cpp:function:: int16_t torch::stable::accelerator::getCurrentDeviceIndex()

   Returns the current accelerator device index.

**Example:**

.. code-block:: cpp

   {
       torch::stable::accelerator::DeviceGuard guard(1);
       // Operations here run on device 1
   }
   // Previous device is restored

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

STD_CUDA_CHECK
^^^^^^^^^^^^^^

.. c:macro:: STD_CUDA_CHECK(EXPR)

   Checks the result of a CUDA API call and throws an exception on error.
   Users of this macro are expected to include ``cuda_runtime.h``.

   **Example:**

   .. code-block:: cpp

      STD_CUDA_CHECK(cudaMalloc(&ptr, size));
      STD_CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));

   Minimum compatible version: PyTorch 2.10.

STD_CUDA_KERNEL_LAUNCH_CHECK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. c:macro:: STD_CUDA_KERNEL_LAUNCH_CHECK()

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

For the complete list of header-only APIs, see ``torch/header_only_apis.txt``
in the PyTorch source tree.

Parallelization Utilities
-------------------------

.. cpp:function:: void torch::stable::parallel_for(int64_t begin, int64_t end, int64_t grain_size, const std::function<void(int64_t, int64_t)>& f)

   Parallel for loop over a range.

   :param begin: Start of the range.
   :param end: End of the range (exclusive).
   :param grain_size: Minimum iterations per thread.
   :param f: Function to execute, receives (start, end) of sub-range.

   **Example:**

   .. code-block:: cpp

      torch::stable::parallel_for(0, tensor.numel(), 1000, [&](int64_t start, int64_t end) {
          for (int64_t i = start; i < end; i++) {
              data[i] = compute(i);
          }
      });

.. cpp:function:: int64_t torch::stable::get_num_threads()

   Get the number of threads used for parallelization.
