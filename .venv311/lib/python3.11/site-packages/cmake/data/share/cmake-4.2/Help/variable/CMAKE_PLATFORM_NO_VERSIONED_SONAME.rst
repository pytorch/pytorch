CMAKE_PLATFORM_NO_VERSIONED_SONAME
----------------------------------

.. versionadded:: 3.1

This variable is used to globally control whether the
:prop_tgt:`VERSION` and :prop_tgt:`SOVERSION` target
properties should be used for shared libraries.
When set to true, adding version information to each
shared library target is disabled.

By default this variable is set only on platforms where
CMake knows it is needed.   On other platforms, the
specified properties will be used for shared libraries.
