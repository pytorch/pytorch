add_compile_definitions
-----------------------

.. versionadded:: 3.12

Add preprocessor definitions to the compilation of source files.

.. code-block:: cmake

  add_compile_definitions(<definition> ...)

Adds preprocessor definitions to the compiler command line.

The preprocessor definitions are added to the :prop_dir:`COMPILE_DEFINITIONS`
directory property for the current ``CMakeLists`` file. They are also added to
the :prop_tgt:`COMPILE_DEFINITIONS` target property for each target in the
current ``CMakeLists`` file.

Definitions are specified using the syntax ``VAR`` or ``VAR=value``.
Function-style definitions are not supported. CMake will automatically
escape the value correctly for the native build system (note that CMake
language syntax may require escapes to specify some values).

.. versionadded:: 3.26
  Any leading ``-D`` on an item will be removed.

.. |command_name| replace:: ``add_compile_definitions``
.. include:: include/GENEX_NOTE.rst

See Also
^^^^^^^^

* The command :command:`target_compile_definitions` adds target-specific definitions.
