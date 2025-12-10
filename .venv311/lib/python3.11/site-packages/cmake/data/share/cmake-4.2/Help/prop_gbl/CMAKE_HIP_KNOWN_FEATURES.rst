CMAKE_HIP_KNOWN_FEATURES
------------------------

.. versionadded:: 3.30

List of HIP features known to this version of CMake.

The features listed in this global property may be known to be available to the
HIP compiler.  If the feature is available with the HIP compiler, it will
be listed in the :variable:`CMAKE_HIP_COMPILE_FEATURES` variable.

The features listed here may be used with the :command:`target_compile_features`
command.  See the :manual:`cmake-compile-features(7)` manual for information on
compile features and a list of supported compilers.


The features known to this version of CMake are:

``hip_std_98``
  Compiler mode is at least HIP/C++ 98.

``hip_std_11``
  Compiler mode is at least HIP/C++ 11.

``hip_std_14``
  Compiler mode is at least HIP/C++ 14.

``hip_std_17``
  Compiler mode is at least HIP/C++ 17.

``hip_std_20``
  Compiler mode is at least HIP/C++ 20.

``hip_std_23``
  Compiler mode is at least HIP/C++ 23.

``hip_std_26``
  .. versionadded:: 3.30

  Compiler mode is at least HIP/C++ 26.

.. include:: include/CMAKE_LANG_STD_FLAGS.rst
