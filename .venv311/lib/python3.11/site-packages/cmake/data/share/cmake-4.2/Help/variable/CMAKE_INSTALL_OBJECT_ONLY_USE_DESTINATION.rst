CMAKE_INSTALL_OBJECT_ONLY_USE_DESTINATION
-----------------------------------------

.. versionadded:: 4.2

Controls whether the ``install(DESTINATION)`` for object libraries is used
as-is or supplemented with conflict-avoiding subdirectories.

``CMAKE_INSTALL_OBJECT_ONLY_USE_DESTINATION`` is used to initialize the
:prop_tgt:`INSTALL_OBJECT_ONLY_USE_DESTINATION` property on all targets.  See
that target property for more information.
