CUDA Streams
============

CUDA streams provide a mechanism for asynchronous execution of operations
on the GPU. Operations queued to the same stream execute in order, while
operations on different streams can execute concurrently.

CUDAStream
----------

.. doxygenclass:: c10::cuda::CUDAStream
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   #include <c10/cuda/CUDAStream.h>

   // Get the default stream for current device
   auto stream = c10::cuda::getDefaultCUDAStream();

   // Create a new stream
   auto new_stream = c10::cuda::getStreamFromPool();

   // Get current stream
   auto current = c10::cuda::getCurrentCUDAStream();

   // Synchronize
   stream.synchronize();

Acquiring CUDA Streams
----------------------

PyTorch provides several ways to acquire CUDA streams:

1. **From the stream pool** (round-robin allocation):

   .. code-block:: cpp

      // Normal priority stream
      at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

      // High priority stream
      at::cuda::CUDAStream high_prio = at::cuda::getStreamFromPool(/*isHighPriority=*/true);

      // Stream for specific device
      at::cuda::CUDAStream dev1_stream = at::cuda::getStreamFromPool(false, /*device=*/1);

2. **Default stream** (where most computation occurs):

   .. code-block:: cpp

      at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();

3. **Current stream** (may differ if changed with guards):

   .. code-block:: cpp

      at::cuda::CUDAStream currentStream = at::cuda::getCurrentCUDAStream();

Setting CUDA Streams
--------------------

**Using setCurrentCUDAStream:**

.. code-block:: cpp

   torch::Tensor tensor0 = torch::ones({2, 2}, torch::device(torch::kCUDA));

   // Get a new stream and set it as current
   at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool();
   at::cuda::setCurrentCUDAStream(myStream);

   // Operations now use myStream
   tensor0.sum();

   // Restore default stream
   at::cuda::setCurrentCUDAStream(at::cuda::getDefaultCUDAStream());

**Using CUDAStreamGuard (recommended):**

.. code-block:: cpp

   torch::Tensor tensor0 = torch::ones({2, 2}, torch::device(torch::kCUDA));
   at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool();

   {
       at::cuda::CUDAStreamGuard guard(myStream);
       // Operations use myStream within this scope
       tensor0.sum();
   }
   // Stream automatically restored to default

Multi-Device Stream Management
------------------------------

**Streams on multiple devices:**

.. code-block:: cpp

   // Acquire streams for different devices
   at::cuda::CUDAStream stream0 = at::cuda::getStreamFromPool(false, 0);
   at::cuda::CUDAStream stream1 = at::cuda::getStreamFromPool(false, 1);

   // Set current streams on each device
   at::cuda::setCurrentCUDAStream(stream0);
   at::cuda::setCurrentCUDAStream(stream1);

   // Create tensors on device 0
   torch::Tensor tensor0 = torch::ones({2, 2}, torch::device(at::kCUDA));
   tensor0.sum();  // Uses stream0

   // Switch to device 1
   {
       at::cuda::CUDAGuard device_guard(1);
       torch::Tensor tensor1 = torch::ones({2, 2}, torch::device(at::kCUDA));
       tensor1.sum();  // Uses stream1
   }

**Using CUDAMultiStreamGuard:**

.. code-block:: cpp

   torch::Tensor tensor0 = torch::ones({2, 2}, torch::device({torch::kCUDA, 0}));
   torch::Tensor tensor1 = torch::ones({2, 2}, torch::device({torch::kCUDA, 1}));

   at::cuda::CUDAStream stream0 = at::cuda::getStreamFromPool(false, 0);
   at::cuda::CUDAStream stream1 = at::cuda::getStreamFromPool(false, 1);

   {
       // Set streams on both devices simultaneously
       at::cuda::CUDAMultiStreamGuard multi_guard({stream0, stream1});

       tensor0.sum();  // Uses stream0 on device 0
       tensor1.sum();  // Uses stream1 on device 1
   }
   // Both streams restored to defaults

.. attention::

   ``CUDAMultiStreamGuard`` does not change the current device index. It only
   changes the stream on each passed-in stream's device.
