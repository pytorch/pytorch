CMAKE_BUILD_RPATH
-----------------

.. versionadded:: 3.8

:ref:`Semicolon-separated list <CMake Language Lists>` specifying runtime path (``RPATH``)
entries to add to binaries linked in the build tree (for platforms that
support it).  The entries will *not* be used for binaries in the install
tree.  See also the :variable:`CMAKE_INSTALL_RPATH` variable.

This is used to initialize the :prop_tgt:`BUILD_RPATH` target property
for all targets.
