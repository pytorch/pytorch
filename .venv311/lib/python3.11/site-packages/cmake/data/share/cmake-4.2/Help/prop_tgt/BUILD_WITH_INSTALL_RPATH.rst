BUILD_WITH_INSTALL_RPATH
------------------------

``BUILD_WITH_INSTALL_RPATH`` is a boolean specifying whether to link the target
in the build tree with the :prop_tgt:`INSTALL_RPATH`.  This takes precedence
over :prop_tgt:`SKIP_BUILD_RPATH` and avoids the need for relinking before
installation.

This property is initialized by the value of the
:variable:`CMAKE_BUILD_WITH_INSTALL_RPATH` variable if it is set when a target
is created.

If policy :policy:`CMP0068` is not ``NEW``, this property also controls use of
:prop_tgt:`INSTALL_NAME_DIR` in the build tree on macOS.  Either way, the
:prop_tgt:`BUILD_WITH_INSTALL_NAME_DIR` target property takes precedence.
