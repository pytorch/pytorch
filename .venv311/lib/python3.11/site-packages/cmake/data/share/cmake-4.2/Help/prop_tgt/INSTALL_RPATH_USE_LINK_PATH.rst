INSTALL_RPATH_USE_LINK_PATH
---------------------------

Add paths to linker search and installed rpath.

``INSTALL_RPATH_USE_LINK_PATH`` is a boolean that if set to ``TRUE``
will append to the runtime search path (rpath) of installed binaries
any directories outside the project that are in the linker search path or
contain linked library files.  The directories are appended after the
value of the :prop_tgt:`INSTALL_RPATH` target property.

This property is initialized by the value of the variable
:variable:`CMAKE_INSTALL_RPATH_USE_LINK_PATH` if it is set when a target is
created.
