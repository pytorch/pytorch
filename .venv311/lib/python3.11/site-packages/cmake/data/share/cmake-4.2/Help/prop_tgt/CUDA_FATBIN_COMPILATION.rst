CUDA_FATBIN_COMPILATION
-----------------------

.. versionadded:: 3.27

Compile CUDA sources to ``.fatbin`` files instead of ``.obj`` files
within :ref:`Object Libraries`.

For example:

.. code-block:: cmake

  add_library(myfbins OBJECT a.cu b.cu)
  set_property(TARGET myfbins PROPERTY CUDA_FATBIN_COMPILATION ON)
