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
   * - ``TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT``
     - The cache limit for the cuDNN v8 API. This is used to limit the memory used by the cuDNN v8 API. The default value is 10000, which roughly corresponds to 2GiB assuming 200KiB per ExecutionPlan. Set to ``0`` for no limit or a negative value for no caching.
   * - ``PYTORCH_NO_CUDA_MEMORY_CACHING=1``
     - If set to ``1``, disables caching of memory allocations in CUDA. This can be useful for debugging.
   * - ``PYTORCH_NVML_BASED_CUDA_CHECK=1``
     - If set to ``1``, before importing PyTorch modules that check if CUDA is available, PyTorch will use NVML to check if the CUDA driver is functional instead of using the CUDA runtime. This can be helpful if forked processes fail with a CUDA initialization error.