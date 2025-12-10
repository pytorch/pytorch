CUDA_RUNTIME_LIBRARY
--------------------

.. versionadded:: 3.17

Select the CUDA runtime library for use by compilers targeting the CUDA language.

The allowed case insensitive values are:

.. include:: include/CUDA_RUNTIME_LIBRARY-VALUES.rst

Contents of ``CUDA_RUNTIME_LIBRARY`` may use
:manual:`generator expressions <cmake-generator-expressions(7)>`.

If that property is not set then CMake uses an appropriate default
value based on the compiler to select the CUDA runtime library.

.. note::

  This property has effect only when the ``CUDA`` language is enabled. To
  control the CUDA runtime linking when only using the CUDA SDK with the
  ``C`` or ``C++`` language we recommend using the :module:`FindCUDAToolkit`
  module.
