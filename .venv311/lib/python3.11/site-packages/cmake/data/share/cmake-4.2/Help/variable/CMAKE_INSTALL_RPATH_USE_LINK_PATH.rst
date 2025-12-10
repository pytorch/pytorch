CMAKE_INSTALL_RPATH_USE_LINK_PATH
---------------------------------

Add paths to linker search and installed rpath.

``CMAKE_INSTALL_RPATH_USE_LINK_PATH`` is a boolean that if set to ``True``
will append to the runtime search path (rpath) of installed binaries
any directories outside the project that are in the linker search path or
contain linked library files.  The directories are appended after the
value of the :prop_tgt:`INSTALL_RPATH` target property.

This variable is used to initialize the target property
:prop_tgt:`INSTALL_RPATH_USE_LINK_PATH` for all targets.
