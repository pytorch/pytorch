IMPORTED_OBJECTS_<CONFIG>
-------------------------

.. versionadded:: 3.9

``<CONFIG>``-specific version of :prop_tgt:`IMPORTED_OBJECTS` property.

Configuration names correspond to those provided by the project from
which the target is imported.


Xcode Generator Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Do not use this ``<CONFIG>``-specific property if you need to use
:generator:`Xcode` variables like ``$(CURRENT_ARCH)`` or
``$(EFFECTIVE_PLATFORM_NAME)`` in the value.  The ``<CONFIG>``-specific
properties will be ignored in such cases because CMake cannot determine
whether a file exists at the configuration-specific path at configuration
time.  For such cases, use :prop_tgt:`IMPORTED_OBJECTS` instead.
