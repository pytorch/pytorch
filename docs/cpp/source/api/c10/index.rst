C10: Core Utilities
===================

C10 (Caffe2 + ATen = C10) is the core library that provides fundamental
utilities and data types used throughout PyTorch. It contains device
abstractions, memory management utilities, and common data structures.

**When to use C10:**

- When working with device-agnostic code (CPU, CUDA, XPU, etc.)
- When you need efficient array views without copying data
- When handling optional values or type-erased containers
- When writing code that needs to work across different PyTorch backends

**Basic usage:**

.. code-block:: cpp

   #include <c10/core/Device.h>
   #include <c10/util/ArrayRef.h>

   // Device abstraction
   c10::Device device(c10::kCUDA, 0);
   if (device.is_cuda()) {
       std::cout << "Using CUDA device " << device.index() << std::endl;
   }

   // Efficient array views (no copy)
   std::vector<int64_t> sizes = {3, 4, 5};
   c10::ArrayRef<int64_t> sizes_ref(sizes);

   // Optional values
   c10::optional<int64_t> maybe_dim = 2;
   int64_t dim = maybe_dim.value_or(-1);

Header Files
------------

- ``c10/core/Device.h`` - Device abstraction
- ``c10/core/DeviceType.h`` - Device type enumeration
- ``c10/util/ArrayRef.h`` - Non-owning array reference
- ``c10/util/OptionalArrayRef.h`` - Optional array reference
- ``c10/util/Optional.h`` - Optional value wrapper
- ``c10/util/Half.h`` - Half-precision float
- ``c10/util/Exception.h`` - Exception utilities
- ``c10/cuda/CUDAGuard.h`` - CUDA device guards (see :doc:`../cuda/index`)
- ``c10/cuda/CUDAStream.h`` - CUDA stream management (see :doc:`../cuda/index`)
- ``c10/xpu/XPUStream.h`` - Intel XPU stream management
- ``ATen/core/ivalue.h`` - IValue for TorchScript interop

C10 Categories
--------------

.. toctree::
   :maxdepth: 1

   device
   types
   utilities
