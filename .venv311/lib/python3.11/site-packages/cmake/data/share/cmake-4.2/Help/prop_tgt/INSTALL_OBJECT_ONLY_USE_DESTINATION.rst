INSTALL_OBJECT_ONLY_USE_DESTINATION
-----------------------------------

.. versionadded:: 4.2

Controls whether the ``install(DESTINATION)`` for object libraries is used
as-is or supplemented with conflict-avoiding subdirectories.

When installing object files, CMake automatically adds
``objects[-<CONFIG>]/<TARGET_NAME>`` components to the destination to avoid
conflicts. Use this property to suppress these components. Note that when
using a single install prefix for multiple configurations (whether via
multi-config generators or separate build trees), the destination must use
``$<CONFIG>`` to avoid conflicts. Alternatively, the
:prop_sf:`INSTALL_OBJECT_NAME` may be used to avoid configuration-based
conflicts.

This property is initialized by the value of
:variable:`CMAKE_INSTALL_OBJECT_ONLY_USE_DESTINATION` when the target is
created.
