CMAKE_SYSTEM
------------

Composite name of operating system CMake is compiling for.

This variable is the composite of :variable:`CMAKE_SYSTEM_NAME` and
:variable:`CMAKE_SYSTEM_VERSION`, e.g.
``${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_VERSION}``.  If
:variable:`CMAKE_SYSTEM_VERSION` is not set, then this variable is
the same as :variable:`CMAKE_SYSTEM_NAME`.
