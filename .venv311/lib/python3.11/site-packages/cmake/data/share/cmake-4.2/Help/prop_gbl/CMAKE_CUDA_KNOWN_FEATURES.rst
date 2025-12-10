CMAKE_CUDA_KNOWN_FEATURES
-------------------------

.. versionadded:: 3.17

List of CUDA features known to this version of CMake.

The features listed in this global property may be known to be available to the
CUDA compiler.  If the feature is available with the C++ compiler, it will
be listed in the :variable:`CMAKE_CUDA_COMPILE_FEATURES` variable.

The features listed here may be used with the :command:`target_compile_features`
command.  See the :manual:`cmake-compile-features(7)` manual for information on
compile features and a list of supported compilers.


The features known to this version of CMake are:

``cuda_std_03``
  Compiler mode is at least CUDA/C++ 03.

``cuda_std_11``
  Compiler mode is at least CUDA/C++ 11.

``cuda_std_14``
  Compiler mode is at least CUDA/C++ 14.

``cuda_std_17``
  Compiler mode is at least CUDA/C++ 17.

``cuda_std_20``
  Compiler mode is at least CUDA/C++ 20.

``cuda_std_23``
  .. versionadded:: 3.20

  Compiler mode is at least CUDA/C++ 23.

``cuda_std_26``
  .. versionadded:: 3.30

  Compiler mode is at least CUDA/C++ 26.

.. include:: include/CMAKE_LANG_STD_FLAGS.rst
