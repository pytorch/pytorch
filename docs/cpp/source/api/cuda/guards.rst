CUDA Guards
===========

CUDA guards are RAII wrappers that set a CUDA device or stream as the current
context and automatically restore the previous context when the guard goes
out of scope.

CUDAGuard
---------

.. doxygenstruct:: c10::cuda::CUDAGuard
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   #include <c10/cuda/CUDAGuard.h>

   {
       c10::cuda::CUDAGuard guard(1);  // Switch to device 1
       // All CUDA operations here run on device 1
       auto tensor = torch::zeros({2, 2}, torch::device(torch::kCUDA));
   }
   // Previous device is restored

CUDAStreamGuard
---------------

.. doxygenstruct:: c10::cuda::CUDAStreamGuard
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   #include <c10/cuda/CUDAGuard.h>

   auto stream = c10::cuda::getStreamFromPool();
   {
       c10::cuda::CUDAStreamGuard guard(stream);
       // Operations here use the specified stream
   }
   // Previous stream is restored

OptionalCUDAGuard
-----------------

.. cpp:class:: c10::cuda::OptionalCUDAGuard

   A CUDA guard that can optionally be initialized. Unlike CUDAGuard,
   this can be created without setting a device, then set_device() can
   be called later.

   .. cpp:function:: OptionalCUDAGuard()

      Default constructor. Does not set any device.

   .. cpp:function:: void set_device(Device device)

      Sets the device. Can be called at most once.

**Example:**

.. code-block:: cpp

   c10::cuda::OptionalCUDAGuard guard;
   if (use_cuda) {
       guard.set_device(0);
   }
   // Guard only switches device if set_device was called
