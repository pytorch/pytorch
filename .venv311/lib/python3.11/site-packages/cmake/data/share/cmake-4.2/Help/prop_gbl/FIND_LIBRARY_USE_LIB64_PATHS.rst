FIND_LIBRARY_USE_LIB64_PATHS
----------------------------

Whether :command:`find_library` should automatically search lib64
directories.

FIND_LIBRARY_USE_LIB64_PATHS is a boolean specifying whether the
:command:`find_library` command should automatically search the lib64
variant of directories called lib in the search path when building
64-bit binaries.

See also the :variable:`CMAKE_FIND_LIBRARY_CUSTOM_LIB_SUFFIX` variable.
