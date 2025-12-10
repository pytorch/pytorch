CUDA_CUBIN_COMPILATION
----------------------

.. versionadded:: 3.27

Compile CUDA sources to ``.cubin`` files instead of ``.obj`` files
within :ref:`Object Libraries`.

For example:

.. code-block:: cmake

  add_library(mycubin OBJECT a.cu b.cu)
  set_property(TARGET mycubin PROPERTY CUDA_CUBIN_COMPILATION ON)
