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

.. cpp:function:: CUDAStream c10::cuda::getDefaultCUDAStream(int device_index = -1)

   Returns the default CUDA stream for the given device.

.. cpp:function:: CUDAStream c10::cuda::getCurrentCUDAStream(int device_index = -1)

   Returns the current CUDA stream for the given device.

.. cpp:function:: void c10::cuda::setCurrentCUDAStream(CUDAStream stream)

   Sets the current CUDA stream.

.. cpp:function:: CUDAStream c10::cuda::getStreamFromPool(bool high_priority = false, int device_index = -1)

   Gets a CUDA stream from the stream pool.

**Example:**

.. code-block:: cpp

   #include <c10/cuda/CUDAFunctions.h>

   // Create and set custom stream
   auto stream = c10::cuda::getStreamFromPool();
   c10::cuda::setCurrentCUDAStream(stream);

   // Get default stream
   auto default_stream = c10::cuda::getDefaultCUDAStream();
