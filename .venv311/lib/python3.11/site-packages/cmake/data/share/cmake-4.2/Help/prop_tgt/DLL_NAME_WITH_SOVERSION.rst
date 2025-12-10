DLL_NAME_WITH_SOVERSION
-----------------------

.. versionadded:: 3.27

This property controls whether the :prop_tgt:`SOVERSION` target
property is added to the filename of generated DLL filenames
for the Windows platform, which is selected when the
:variable:`WIN32` variable is set.

The value of the listed property is appended to the
basename of the runtime component of the shared library
target as ``-<SOVERSION>``.

Please note that setting this property has no effect
if versioned filenames are globally disabled with the
:variable:`CMAKE_PLATFORM_NO_VERSIONED_SONAME` variable.
