CMAKE_XCODE_LINK_BUILD_PHASE_MODE
---------------------------------

.. versionadded:: 3.19

This variable is used to initialize the
:prop_tgt:`XCODE_LINK_BUILD_PHASE_MODE` property on targets.
It affects the methods that the :generator:`Xcode` generator uses to link
different kinds of libraries.  Its default value is ``NONE``.
