CMAKE_SKIP_BUILD_RPATH
----------------------

Do not include RPATHs in the build tree.

Normally CMake uses the build tree for the RPATH when building
executables etc on systems that use RPATH.  When the software is
installed the executables etc are relinked by CMake to have the
install RPATH.  If this variable is set to ``TRUE`` then the software is
always built with no RPATH.

This is used to initialize the :prop_tgt:`SKIP_BUILD_RPATH` target property
for all targets. For more information on RPATH handling see
the :prop_tgt:`INSTALL_RPATH` and :prop_tgt:`BUILD_RPATH` target properties.

See also the :variable:`CMAKE_SKIP_INSTALL_RPATH` variable.
To omit RPATH in both the build and install steps, use
:variable:`CMAKE_SKIP_RPATH` instead.
