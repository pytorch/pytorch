CMAKE_HOST_SYSTEM
-----------------

Composite Name of OS CMake is being run on.

This variable is the composite of :variable:`CMAKE_HOST_SYSTEM_NAME` and
:variable:`CMAKE_HOST_SYSTEM_VERSION`, e.g.
``${CMAKE_HOST_SYSTEM_NAME}-${CMAKE_HOST_SYSTEM_VERSION}``.  If
:variable:`CMAKE_HOST_SYSTEM_VERSION` is not set, then this variable is
the same as :variable:`CMAKE_HOST_SYSTEM_NAME`.
