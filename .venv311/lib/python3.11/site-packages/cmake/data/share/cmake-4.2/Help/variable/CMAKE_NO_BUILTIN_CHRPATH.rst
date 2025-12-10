CMAKE_NO_BUILTIN_CHRPATH
------------------------

Do not use the builtin binary editor to fix runtime library search
paths on installation.

When an ELF or XCOFF binary needs to have a different runtime library
search path after installation than it does in the build tree, CMake uses
a builtin editor to change the runtime search path in the installed copy.
If this variable is set to true then CMake will relink the binary before
installation instead of using its builtin editor.

For more information on RPATH handling see
the :prop_tgt:`INSTALL_RPATH` and :prop_tgt:`BUILD_RPATH` target properties.

.. versionadded:: 3.20

  This variable also applies to XCOFF binaries' LIBPATH.  Prior to the
  addition of the XCOFF editor in CMake 3.20, this variable applied only
  to ELF binaries' RPATH/RUNPATH.
