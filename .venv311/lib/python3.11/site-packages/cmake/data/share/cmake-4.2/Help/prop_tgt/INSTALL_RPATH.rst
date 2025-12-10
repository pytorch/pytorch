INSTALL_RPATH
-------------

The rpath to use for installed targets.

By default, the install rpath is empty. It can be set using this property,
which is a semicolon-separated list specifying the rpath to use in installed
targets (for platforms that support it).  This property is initialized
by the value of the variable :variable:`CMAKE_INSTALL_RPATH` if it is set
when a target is created.
Beside setting the install rpath manually, using the
:prop_tgt:`INSTALL_RPATH_USE_LINK_PATH` target property it can also be
generated automatically by CMake.

Normally CMake uses the build tree for the RPATH when building executables
etc on systems that use RPATH, see the :prop_tgt:`BUILD_RPATH` target
property. When the software is installed
the targets are edited (or relinked) by CMake (see
:variable:`CMAKE_NO_BUILTIN_CHRPATH`) to have the install RPATH.
This editing during installation can be avoided via
the :prop_tgt:`BUILD_WITH_INSTALL_RPATH` target property.

For handling toolchain-dependent RPATH entries the
:prop_tgt:`INSTALL_REMOVE_ENVIRONMENT_RPATH` can be used.
Runtime paths can be disabled completely via the :variable:`CMAKE_SKIP_RPATH`
variable.

Because the rpath may contain ``${ORIGIN}``, which coincides with CMake syntax,
the contents of ``INSTALL_RPATH`` are properly escaped in the
``cmake_install.cmake`` script (see policy :policy:`CMP0095`.)

This property supports
:manual:`generator expressions <cmake-generator-expressions(7)>`.

On Apple platforms, refer to the :prop_tgt:`INSTALL_NAME_DIR` target property.
Under Windows, the :genex:`TARGET_RUNTIME_DLLS` generator expression is related.
