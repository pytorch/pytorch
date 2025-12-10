OBJCXX_STANDARD
---------------

.. versionadded:: 3.16

The ObjC++ standard whose features are requested to build this target.

This property specifies the ObjC++ standard whose features are requested
to build this target.  For some compilers, this results in adding a
flag such as ``-std=gnu++11`` to the compile line.

Supported values are:

``98``
  Objective C++98

``11``
  Objective C++11

``14``
  Objective C++14

``17``
  Objective C++17

``20``
  Objective C++20

``23``
  .. versionadded:: 3.20

  Objective C++23

``26``
  .. versionadded:: 3.25

  Objective C++26. CMake 3.25 and later *recognize* ``26`` as a valid value,
  no version has support for any compiler.

If the value requested does not result in a compile flag being added for
the compiler in use, a previous standard flag will be added instead.  This
means that using:

.. code-block:: cmake

  set_property(TARGET tgt PROPERTY OBJCXX_STANDARD 11)

with a compiler which does not support ``-std=gnu++11`` or an equivalent
flag will not result in an error or warning, but will instead add the
``-std=gnu++98`` flag if supported.  This "decay" behavior may be controlled
with the :prop_tgt:`OBJCXX_STANDARD_REQUIRED` target property.
Additionally, the :prop_tgt:`OBJCXX_EXTENSIONS` target property may be used to
control whether compiler-specific extensions are enabled on a per-target basis.

If the property is not set, and the project has set the :prop_tgt:`CXX_STANDARD`,
the value of :prop_tgt:`CXX_STANDARD` is set for ``OBJCXX_STANDARD``.

See the :manual:`cmake-compile-features(7)` manual for information on
compile features and a list of supported compilers.

This property is initialized by the value of
the :variable:`CMAKE_OBJCXX_STANDARD` variable if it is set when a target
is created.
