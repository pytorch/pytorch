HIP_STANDARD
------------

.. versionadded:: 3.21

The HIP/C++ standard requested to build this target.

Supported values are:

``98``
  HIP C++98

``11``
  HIP C++11

``14``
  HIP C++14

``17``
  HIP C++17

``20``
  HIP C++20

``23``
  HIP C++23

``26``
  .. versionadded:: 3.25

  HIP C++26. CMake 3.25 and later *recognize* ``26`` as a valid value,
  no version has support for any compiler.

If the value requested does not result in a compile flag being added for
the compiler in use, a previous standard flag will be added instead.  This
means that using:

.. code-block:: cmake

  set_property(TARGET tgt PROPERTY HIP_STANDARD 11)

with a compiler which does not support ``-std=gnu++11`` or an equivalent
flag will not result in an error or warning, but will instead add the
``-std=gnu++98`` flag if supported.  This "decay" behavior may be controlled
with the :prop_tgt:`HIP_STANDARD_REQUIRED` target property.
Additionally, the :prop_tgt:`HIP_EXTENSIONS` target property may be used to
control whether compiler-specific extensions are enabled on a per-target basis.

See the :manual:`cmake-compile-features(7)` manual for information on
compile features and a list of supported compilers.

This property is initialized by the value of
the :variable:`CMAKE_HIP_STANDARD` variable if it is set when a target
is created.
