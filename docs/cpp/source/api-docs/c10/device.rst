Device and DeviceType
=====================

PyTorch provides device abstractions for writing code that works across
CPU, CUDA, and other backends.

Device
------

.. doxygenstruct:: c10::Device
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   c10::Device cpu_device(c10::kCPU);
   c10::Device cuda_device(c10::kCUDA, 0);  // CUDA device 0

   if (cuda_device.is_cuda()) {
       std::cout << "Using CUDA device " << cuda_device.index() << std::endl;
   }

DeviceType
----------

.. cpp:enum-class:: c10::DeviceType

   Enumeration of supported device types.

   .. cpp:enumerator:: CPU
   .. cpp:enumerator:: CUDA
   .. cpp:enumerator:: XPU
   .. cpp:enumerator:: MPS
   .. cpp:enumerator:: Meta

Convenience constants:

- ``c10::kCPU``
- ``c10::kCUDA``
- ``c10::kXPU``
- ``c10::kMPS``
- ``c10::kMeta``
