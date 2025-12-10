CMAKE_PROJECT_TOP_LEVEL_INCLUDES
--------------------------------

.. versionadded:: 3.24

:ref:`Semicolon-separated list <CMake Language Lists>` of CMake language
files to include as part of the very first :command:`project` call.
The files will be included immediately after the toolchain file has been read
(if one is specified) and platform variables have been set, but before any
languages have been enabled. Therefore, language-specific variables,
including things like :variable:`CMAKE_<LANG>_COMPILER`, might not be set.
See :ref:`Code Injection` for a more detailed discussion of files potentially
included during a :command:`project` call.

.. versionadded:: 3.29
  This variable can also now refer to module names to be found in
  :variable:`CMAKE_MODULE_PATH` or builtin to CMake.

This variable is intended for specifying files that perform one-time setup
for the build. It provides an injection point for things like configuring
package managers, adding logic the user shares between projects (e.g. defining
their own custom build types), and so on. It is primarily for users to add
things specific to their environment, but not for specifying the toolchain
details (use :variable:`CMAKE_TOOLCHAIN_FILE` for that).

By default, this variable is empty.  It is intended to be set by the user.

See also:

* :variable:`CMAKE_PROJECT_INCLUDE`
* :variable:`CMAKE_PROJECT_INCLUDE_BEFORE`
* :variable:`CMAKE_PROJECT_<PROJECT-NAME>_INCLUDE`
* :variable:`CMAKE_PROJECT_<PROJECT-NAME>_INCLUDE_BEFORE`
* :prop_gbl:`PROPAGATE_TOP_LEVEL_INCLUDES_TO_TRY_COMPILE`
