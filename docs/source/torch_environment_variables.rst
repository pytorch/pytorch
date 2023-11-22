.. _torch_environment_variables:

Torch Environment Variables
===============================
PyTorch uses environment variables to control various settings, such as which
gloo device to use, the number of threads used for parallelism, and the number
of OpenMP threads, and many others. As well, some libraries that PyTorch uses
(e.g., MKL) also use environment variables to control their behavior. This
page lists the environment variables that can be used to configure PyTorch.
Note: There are many environment variables and this list is not exhaustive.
This page lists the environment variables that can be used
to configure PyTorch.

Threading Environment Variables
=====================
.. list-table::
   :header-rows: 1

   * - Variable
     - Description
   * - ``OMP_NUM_THREADS``
     - Sets the maximum number of threads to use for OpenMP parallel regions.
   * - ``MKL_NUM_THREADS``
     - Sets the maximum number of threads to use for the Intel MKL library. Note that MKL_NUM_THREADS takes precedence over ``OMP_NUM_THREADS``.

CUDA Environment Variables
=====================
.. list-table::
   :header-rows: 1

   * - Variable
     - Description
   * - ``CUDA_VISIBLE_DEVICES``
       - Comma-separated list of GPU device IDs that should be made available to CUDA runtime. If set to ``-1``, no GPUs are made available.
   * - ``CUDA_LAUNCH_BLOCKING``
         - If set to ``1``, makes CUDA calls synchronous. This can be useful for debugging.

Debugging Environment Variables
=====================
.. list-table::
   :header-rows: 1

   * - Variable
     - Description
   * - ``TORCH_SHOW_CPP_STACKTRACES
       - If set to ``1``, makes PyTorch print out a stack trace when it detects a C++ error.