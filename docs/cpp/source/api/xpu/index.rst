XPU Support
===========

PyTorch provides XPU support for Intel GPU-accelerated tensor operations.
The XPU API allows you to manage Intel GPU devices, streams for asynchronous
execution, and synchronization.

**When to use XPU APIs:**

- When running on Intel GPUs (Data Center GPU Max, Arc, etc.)
- When implementing custom XPU kernels or operations
- When managing asynchronous execution with XPU streams
- When writing device-portable code alongside CUDA

**Basic usage:**

.. code-block:: cpp

   #include <torch/torch.h>

   // Check if XPU is available
   if (torch::xpu::is_available()) {
       // Create tensor on XPU
       auto tensor = torch::randn({2, 3}, torch::device(torch::kXPU));

       // Move model to XPU
       model->to(torch::kXPU);
   }

Header Files
------------

- ``torch/xpu.h`` - High-level XPU utilities (device count, availability, seeding)
- ``c10/xpu/XPUStream.h`` - XPU stream management

XPU Categories
--------------

.. toctree::
   :maxdepth: 1

   streams
   utilities
