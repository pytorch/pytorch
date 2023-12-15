.. _cuda_environment_variables:

CUDA Environment Variables
==========================
.. list-table::
   :header-rows: 1

   * - Variable
     - Description
   * - ``CUDA_VISIBLE_DEVICES``
     - Comma-separated list of GPU device IDs that should be made available to CUDA runtime. If set to ``-1``, no GPUs are made available.
   * - ``CUDA_LAUNCH_BLOCKING``
     - If set to ``1``, makes CUDA calls synchronous. This can be useful for debugging.