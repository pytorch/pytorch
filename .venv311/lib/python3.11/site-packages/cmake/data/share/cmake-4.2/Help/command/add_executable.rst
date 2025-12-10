add_executable
--------------

.. only:: html

  .. contents::

Add an executable to the project using the specified source files.

Normal Executables
^^^^^^^^^^^^^^^^^^

.. signature::
  add_executable(<name> <options>... <sources>...)
  :target: normal

  Add an :ref:`executable <Executables>` target called ``<name>`` to
  be built from the source files listed in the command invocation.

  The options are:

  ``WIN32``
    Set the :prop_tgt:`WIN32_EXECUTABLE` target property automatically.
    See documentation of that target property for details.

  ``MACOSX_BUNDLE``
    Set the :prop_tgt:`MACOSX_BUNDLE` target property automatically.
    See documentation of that target property for details.

  ``EXCLUDE_FROM_ALL``
    Set the :prop_tgt:`EXCLUDE_FROM_ALL` target property automatically.
    See documentation of that target property for details.

The ``<name>`` corresponds to the logical target name and must be globally
unique within a project.  The actual file name of the executable built is
constructed based on conventions of the native platform (such as
``<name>.exe`` or just ``<name>``).

.. versionadded:: 3.1
  Source arguments to ``add_executable`` may use "generator expressions" with
  the syntax ``$<...>``.  See the :manual:`cmake-generator-expressions(7)`
  manual for available expressions.

.. versionadded:: 3.11
  The source files can be omitted if they are added later using
  :command:`target_sources`.

By default the executable file will be created in the build tree
directory corresponding to the source tree directory in which the
command was invoked.  See documentation of the
:prop_tgt:`RUNTIME_OUTPUT_DIRECTORY` target property to change this
location.  See documentation of the :prop_tgt:`OUTPUT_NAME` target property
to change the ``<name>`` part of the final file name.

See the :manual:`cmake-buildsystem(7)` manual for more on defining
buildsystem properties.

See also :prop_sf:`HEADER_FILE_ONLY` on what to do if some sources are
pre-processed, and you want to have the original sources reachable from
within IDE.

Imported Executables
^^^^^^^^^^^^^^^^^^^^

.. signature::
  add_executable(<name> IMPORTED [GLOBAL])
  :target: IMPORTED

  Add an :ref:`IMPORTED executable target <Imported Targets>` to reference
  an executable file located outside the project.  The target name may be
  referenced like any target built within the project, except that by
  default it is visible only in the directory in which it is created,
  and below.

  The options are:

  ``GLOBAL``
    Make the target name globally visible.

No rules are generated to build imported targets, and the :prop_tgt:`IMPORTED`
target property is ``True``.  Imported executables are useful for convenient
reference from commands like :command:`add_custom_command`.

Details about the imported executable are specified by setting properties
whose names begin in ``IMPORTED_``.  The most important such property is
:prop_tgt:`IMPORTED_LOCATION` (and its per-configuration version
:prop_tgt:`IMPORTED_LOCATION_<CONFIG>`) which specifies the location of
the main executable file on disk.  See documentation of the ``IMPORTED_*``
properties for more information.

Alias Executables
^^^^^^^^^^^^^^^^^

.. signature::
  add_executable(<name> ALIAS <target>)
  :target: ALIAS

  Creates an :ref:`Alias Target <Alias Targets>`, such that ``<name>`` can
  be used to refer to ``<target>`` in subsequent commands.  The ``<name>``
  does not appear in the generated buildsystem as a make target.  The
  ``<target>`` may not be an ``ALIAS``.

.. versionadded:: 3.11
  An ``ALIAS`` can target a ``GLOBAL`` :ref:`Imported Target <Imported Targets>`

.. versionadded:: 3.18
  An ``ALIAS`` can target a non-``GLOBAL`` Imported Target. Such alias is
  scoped to the directory in which it is created and subdirectories.
  The :prop_tgt:`ALIAS_GLOBAL` target property can be used to check if the
  alias is global or not.

``ALIAS`` targets can be used as targets to read properties
from, executables for custom commands and custom targets.  They can also be
tested for existence with the regular :command:`if(TARGET)` subcommand.
The ``<name>`` may not be used to modify properties of ``<target>``, that
is, it may not be used as the operand of :command:`set_property`,
:command:`set_target_properties`, :command:`target_link_libraries` etc.
An ``ALIAS`` target may not be installed or exported.

See Also
^^^^^^^^

* :command:`add_library`
