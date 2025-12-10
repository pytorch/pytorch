BUILD_RPATH
-----------

.. versionadded:: 3.8

A :ref:`semicolon-separated list <CMake Language Lists>` specifying
runtime path (``RPATH``) entries to add to binaries linked in the
build tree (for platforms that support it).  By default, CMake sets
the runtime path of binaries in the build tree to contain search
paths it knows are needed to find the shared libraries they link.
Projects may set ``BUILD_RPATH`` to specify additional search paths.

The build-tree runtime path will *not* be used for binaries in the
install tree.  It will be replaced with the install-tree runtime path
during the installation step.  See also the :prop_tgt:`INSTALL_RPATH`
target property.

This property is initialized by the value of the variable
:variable:`CMAKE_BUILD_RPATH` if it is set when a target is created.

This property supports
:manual:`generator expressions <cmake-generator-expressions(7)>`.

Other settings that affect the build-tree runtime path include:

* The :variable:`CMAKE_SKIP_RPATH` variable completely disables runtime
  paths in both the build tree and install tree.

* The :prop_tgt:`SKIP_BUILD_RPATH` target property disables setting any
  runtime path in the build tree.

* The :prop_tgt:`BUILD_RPATH_USE_ORIGIN` target property causes the
  automatically-generated runtime path to use entries relative to ``$ORIGIN``.

* The :prop_tgt:`BUILD_WITH_INSTALL_RPATH` target property causes binaries
  in the build tree to be built with the install-tree runtime path.
