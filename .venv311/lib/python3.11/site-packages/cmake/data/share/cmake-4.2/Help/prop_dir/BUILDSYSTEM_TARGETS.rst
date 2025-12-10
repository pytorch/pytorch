BUILDSYSTEM_TARGETS
-------------------

.. versionadded:: 3.7

This read-only directory property contains a
:ref:`semicolon-separated list <CMake Language Lists>` of buildsystem targets added in the
directory by calls to the :command:`add_library`, :command:`add_executable`,
and :command:`add_custom_target` commands.  The list does not include any
:ref:`Imported Targets` or :ref:`Alias Targets`, but does include
:ref:`Interface Libraries`.  Each entry in the list is the logical name
of a target, suitable to pass to the :command:`get_property` command
``TARGET`` option.

See also the :prop_dir:`IMPORTED_TARGETS` directory property.
