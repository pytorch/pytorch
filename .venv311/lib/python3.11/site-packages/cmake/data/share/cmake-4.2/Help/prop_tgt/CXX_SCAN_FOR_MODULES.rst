CXX_SCAN_FOR_MODULES
--------------------

.. versionadded:: 3.28

``CXX_SCAN_FOR_MODULES`` is a boolean specifying whether CMake will scan C++
sources in the target for module dependencies.  See also the
:prop_sf:`CXX_SCAN_FOR_MODULES` for per-source settings which, if set,
overrides the target-wide settings.

This property is initialized by the value of the
:variable:`CMAKE_CXX_SCAN_FOR_MODULES` variable if it is set when a target is
created.

When this property is set ``ON`` or unset, CMake will scan the target's
``CXX`` sources at build time and add module dependency information to the
compile line as necessary.  When this property is set ``OFF``, CMake will not
scan the target's ``CXX`` sources at build time.

Note that scanning is only performed if C++20 or higher is enabled for the
target.  Scanning for modules in the target's sources belonging to file sets
of type ``CXX_MODULES`` is always performed.
