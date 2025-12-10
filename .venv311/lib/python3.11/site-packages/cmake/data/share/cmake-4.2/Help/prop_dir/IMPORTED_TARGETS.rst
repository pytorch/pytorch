IMPORTED_TARGETS
----------------

.. versionadded:: 3.21

This read-only directory property contains a
:ref:`semicolon-separated list <CMake Language Lists>` of
:ref:`Imported Targets` added in the directory by calls to the
:command:`add_library` and :command:`add_executable` commands.
Each entry in the list is the logical name of a target, suitable
to pass to the :command:`get_property` command ``TARGET`` option
when called in the same directory.

See also the :prop_dir:`BUILDSYSTEM_TARGETS` directory property.
