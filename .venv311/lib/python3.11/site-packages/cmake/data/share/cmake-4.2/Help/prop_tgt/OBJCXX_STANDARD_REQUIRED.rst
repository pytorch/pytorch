OBJCXX_STANDARD_REQUIRED
------------------------

.. versionadded:: 3.16

Boolean describing whether the value of :prop_tgt:`OBJCXX_STANDARD` is a requirement.

If this property is set to ``ON``, then the value of the
:prop_tgt:`OBJCXX_STANDARD` target property is treated as a requirement.  If this
property is ``OFF`` or unset, the :prop_tgt:`OBJCXX_STANDARD` target property is
treated as optional and may "decay" to a previous standard if the requested is
not available.

If the property is not set, and the project has set the :prop_tgt:`CXX_STANDARD_REQUIRED`,
the value of :prop_tgt:`CXX_STANDARD_REQUIRED` is set for ``OBJCXX_STANDARD_REQUIRED``.

See the :manual:`cmake-compile-features(7)` manual for information on
compile features and a list of supported compilers.

This property is initialized by the value of
the :variable:`CMAKE_OBJCXX_STANDARD_REQUIRED` variable if it is set when a
target is created.
