CMAKE_SKIP_INSTALL_RPATH
------------------------

Do not include RPATHs in the install tree.

Normally CMake uses the build tree for the RPATH when building
executables etc on systems that use RPATH.  When the software is
installed the executables etc are relinked by CMake to have the
install RPATH.  If this variable is set to true then the software is
always installed without RPATH, even if RPATH is enabled when
building.  This can be useful for example to allow running tests from
the build directory with RPATH enabled before the installation step.

See also the :variable:`CMAKE_SKIP_BUILD_RPATH` variable.
To omit RPATH in both the build and install steps, use
:variable:`CMAKE_SKIP_RPATH` instead.

For more information on RPATH handling see the :prop_tgt:`INSTALL_RPATH`
and :prop_tgt:`BUILD_RPATH` target properties.
