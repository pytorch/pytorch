CUDA_SEPARABLE_COMPILATION
--------------------------

.. versionadded:: 3.8

CUDA only: Enables separate compilation of device code

If set this will enable separable compilation for all CUDA files for
the given target.

For instance:

.. code-block:: cmake

  set_property(TARGET myexe PROPERTY CUDA_SEPARABLE_COMPILATION ON)

This property is initialized by the value of the
:variable:`CMAKE_CUDA_SEPARABLE_COMPILATION` variable if it is set when a
target is created.
