Output directory in which to build |XXX| target files.

This property specifies the directory into which |xxx| target files
should be built.  The property value may use
:manual:`generator expressions <cmake-generator-expressions(7)>`.
Multi-configuration generators (:ref:`Visual Studio <Visual Studio Generators>`,
:generator:`Xcode`, :generator:`Ninja Multi-Config`) append a
per-configuration subdirectory to the specified directory unless a generator
expression is used.

This property is initialized by the value of the
|CMAKE_XXX_OUTPUT_DIRECTORY| variable if it is set when a target is created.
