CUDA_OPTIX_COMPILATION
----------------------

.. versionadded:: 3.27

Compile CUDA sources to ``.optixir`` files instead of ``.obj`` files
within :ref:`Object Libraries`.

For example:

.. code-block:: cmake

  add_library(myoptix OBJECT a.cu b.cu)
  set_property(TARGET myoptix PROPERTY CUDA_OPTIX_COMPILATION ON)
