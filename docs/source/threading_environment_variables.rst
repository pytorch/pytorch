.. _threading_environment_variables:

Threading Environment Variables
===============================
.. list-table::
  :header-rows: 1

  * - Variable
    - Description
  * - ``OMP_NUM_THREADS``
    - Sets the maximum number of threads to use for OpenMP parallel regions.
  * - ``MKL_NUM_THREADS``
    - Sets the maximum number of threads to use for the Intel MKL library. Note that MKL_NUM_THREADS takes precedence over ``OMP_NUM_THREADS``.