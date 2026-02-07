CUDA Support
============

PyTorch provides comprehensive CUDA support for GPU-accelerated tensor
operations and neural network training. The CUDA API allows you to manage
GPU devices, streams for asynchronous execution, and memory efficiently.

**When to use CUDA APIs:**

- When you need explicit control over which GPU device to use
- When implementing custom CUDA kernels or operations
- When optimizing performance with asynchronous stream execution
- When managing multi-GPU workloads

**Basic usage:**

.. code-block:: cpp

   #include <torch/torch.h>
   #include <c10/cuda/CUDAGuard.h>

   // Check if CUDA is available
   if (torch::cuda::is_available()) {
       // Create tensor on GPU
       auto tensor = torch::randn({2, 3}, torch::device(torch::kCUDA));

       // Switch to a specific GPU
       c10::cuda::CUDAGuard guard(0);  // Use GPU 0

       // Get the current CUDA stream
       auto stream = c10::cuda::getCurrentCUDAStream();

       // Move model to GPU
       model->to(torch::kCUDA);
   }

Header Files
------------

- ``c10/cuda/CUDAStream.h`` - CUDA stream management
- ``c10/cuda/CUDAGuard.h`` - CUDA device guards
- ``ATen/cuda/CUDAContext.h`` - CUDA context management
- ``ATen/cudnn/Descriptors.h`` - cuDNN tensor descriptors

CUDA Categories
---------------

.. toctree::
   :maxdepth: 1

   streams
   guards
   utilities
