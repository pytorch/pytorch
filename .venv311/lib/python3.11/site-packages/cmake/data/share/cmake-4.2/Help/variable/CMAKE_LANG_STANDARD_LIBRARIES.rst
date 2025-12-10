CMAKE_<LANG>_STANDARD_LIBRARIES
-------------------------------

.. versionadded:: 3.6

Libraries linked into every executable and shared library linked
for language ``<LANG>``.  This is meant for specification of system
libraries needed by the language for the current platform.

This variable should not be set by project code.  It is meant to be set by
CMake's platform information modules for the current toolchain, or by a
toolchain file when used with :variable:`CMAKE_TOOLCHAIN_FILE`.

See also :variable:`CMAKE_<LANG>_STANDARD_INCLUDE_DIRECTORIES`.
