CUDA_STANDARD
-------------

.. versionadded:: 3.8

The CUDA/C++ standard whose features are requested to build this target.

This property specifies the CUDA/C++ standard whose features are requested
to build this target.  For some compilers, this results in adding a
flag such as ``-std=gnu++11`` to the compile line.

Supported values are:

``98``
  CUDA C++98. Note that this maps to the same as ``03`` internally.

``03``
  CUDA C++03

``11``
  CUDA C++11

``14``
  CUDA C++14. While CMake 3.8 and later *recognize* ``14`` as a valid value,
  CMake 3.9 was the first version to include support for any compiler.

``17``
  CUDA C++17. While CMake 3.8 and later *recognize* ``17`` as a valid value,
  CMake 3.18 was the first version to include support for any compiler.

``20``
  .. versionadded:: 3.12

  CUDA C++20. While CMake 3.12 and later *recognize* ``20`` as a valid value,
  CMake 3.18 was the first version to include support for any compiler.

``23``
  .. versionadded:: 3.20

  CUDA C++23

``26``
  .. versionadded:: 3.25

  CUDA C++26. CMake 3.25 and later *recognize* ``26`` as a valid value,
  no version has support for any compiler.

If the value requested does not result in a compile flag being added for
the compiler in use, a previous standard flag will be added instead.  This
means that using:

.. code-block:: cmake

  set_property(TARGET tgt PROPERTY CUDA_STANDARD 11)

with a compiler which does not support ``-std=gnu++11`` or an equivalent
flag will not result in an error or warning, but will instead add the
``-std=gnu++03`` flag if supported.  This "decay" behavior may be controlled
with the :prop_tgt:`CUDA_STANDARD_REQUIRED` target property.
Additionally, the :prop_tgt:`CUDA_EXTENSIONS` target property may be used to
control whether compiler-specific extensions are enabled on a per-target basis.

See the :manual:`cmake-compile-features(7)` manual for information on
compile features and a list of supported compilers.

This property is initialized by the value of
the :variable:`CMAKE_CUDA_STANDARD` variable if it is set when a target
is created.
