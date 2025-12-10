CMAKE_CXX_STDLIB_MODULES_JSON
-----------------------------

.. versionadded:: 4.2

This variable may be used to set the path to a metadata file for CMake to
understand how the ``import std`` target for the active CXX compiler should be
constructed.

This should only be used when the compiler does not know how to discover the
relevant module metadata file without such assistance.
