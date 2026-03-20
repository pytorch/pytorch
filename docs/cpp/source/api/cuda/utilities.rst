CUDA Utility Functions
======================

PyTorch provides utility functions for querying and managing CUDA devices
and streams.

Device Management
-----------------

.. cpp:function:: int c10::cuda::device_count()

   Returns the number of available CUDA devices.

.. cpp:function:: int c10::cuda::current_device()

   Returns the index of the current CUDA device.

**Example:**

.. code-block:: cpp

   #include <c10/cuda/CUDAFunctions.h>

   // Check available devices
   int num_devices = c10::cuda::device_count();

   // Get current device
   int current = c10::cuda::current_device();

Stream Management
-----------------

.. doxygenfunction:: c10::cuda::getDefaultCUDAStream

.. doxygenfunction:: c10::cuda::getCurrentCUDAStream

.. doxygenfunction:: c10::cuda::setCurrentCUDAStream

.. doxygenfunction:: c10::cuda::getStreamFromPool(const bool isHighPriority, DeviceIndex device)

**Example:**

.. code-block:: cpp

   #include <c10/cuda/CUDAFunctions.h>

   // Create and set custom stream
   auto stream = c10::cuda::getStreamFromPool();
   c10::cuda::setCurrentCUDAStream(stream);

   // Get default stream
   auto default_stream = c10::cuda::getDefaultCUDAStream();
