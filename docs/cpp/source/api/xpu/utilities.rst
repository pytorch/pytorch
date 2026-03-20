XPU Utility Functions
=====================

High-level utility functions for querying and managing XPU devices.

Device Management
-----------------

.. doxygenfunction:: torch::xpu::device_count

.. doxygenfunction:: torch::xpu::is_available

.. doxygenfunction:: torch::xpu::synchronize

**Example:**

.. code-block:: cpp

   #include <torch/torch.h>

   if (torch::xpu::is_available()) {
       size_t num_devices = torch::xpu::device_count();
       std::cout << "Found " << num_devices << " XPU device(s)" << std::endl;

       // Synchronize all streams on device 0
       torch::xpu::synchronize(0);
   }

Random Number Generation
------------------------

.. doxygenfunction:: torch::xpu::manual_seed

.. doxygenfunction:: torch::xpu::manual_seed_all

**Example:**

.. code-block:: cpp

   // Set seed for reproducibility on current XPU device
   torch::xpu::manual_seed(42);

   // Set seed for all XPU devices
   torch::xpu::manual_seed_all(42);
