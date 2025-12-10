CMAKE_CODEBLOCKS_COMPILER_ID
----------------------------

.. versionadded:: 3.11

Change the compiler id in the generated CodeBlocks project files.

CodeBlocks uses its own compiler id string which differs from
:variable:`CMAKE_<LANG>_COMPILER_ID`.  If this variable is left empty,
CMake tries to recognize the CodeBlocks compiler id automatically.
Otherwise the specified string is used in the CodeBlocks project file.
See the CodeBlocks documentation for valid compiler id strings.

Other IDEs like QtCreator that also use the CodeBlocks generator may ignore
this setting.
