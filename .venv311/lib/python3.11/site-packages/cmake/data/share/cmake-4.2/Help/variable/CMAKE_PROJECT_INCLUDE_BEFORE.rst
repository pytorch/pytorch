CMAKE_PROJECT_INCLUDE_BEFORE
----------------------------

.. versionadded:: 3.15

A CMake language file to be included as the first step of all
:command:`project` command calls.  This is intended for injecting custom code
into project builds without modifying their source.  See :ref:`Code Injection`
for a more detailed discussion of files potentially included during a
:command:`project` call.

.. versionadded:: 3.29
  This variable can be a :ref:`semicolon-separated list <CMake Language Lists>`
  of CMake language files to be included sequentially. It can also now refer to
  module names to be found in :variable:`CMAKE_MODULE_PATH` or as a builtin
  CMake module.

See also the :variable:`CMAKE_PROJECT_<PROJECT-NAME>_INCLUDE`,
:variable:`CMAKE_PROJECT_<PROJECT-NAME>_INCLUDE_BEFORE`,
:variable:`CMAKE_PROJECT_INCLUDE`, and
:variable:`CMAKE_PROJECT_TOP_LEVEL_INCLUDES` variables.
