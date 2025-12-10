add_definitions
---------------

Add ``-D`` define flags to the compilation of source files.

.. code-block:: cmake

  add_definitions(-DFOO -DBAR ...)

Adds definitions to the compiler command line for targets in the current
directory, whether added before or after this command is invoked, and for
the ones in sub-directories added after. This command can be used to add any
flags, but it is intended to add preprocessor definitions.

.. note::

  This command has been superseded by alternatives:

  * Use :command:`add_compile_definitions` to add preprocessor definitions.
  * Use :command:`include_directories` to add include directories.
  * Use :command:`add_compile_options` to add other options.

Flags beginning in ``-D`` or ``/D`` that look like preprocessor definitions are
automatically added to the :prop_dir:`COMPILE_DEFINITIONS` directory
property for the current directory.  Definitions with non-trivial values
may be left in the set of flags instead of being converted for reasons of
backwards compatibility.  See documentation of the
:prop_dir:`directory <COMPILE_DEFINITIONS>`,
:prop_tgt:`target <COMPILE_DEFINITIONS>`,
:prop_sf:`source file <COMPILE_DEFINITIONS>` ``COMPILE_DEFINITIONS``
properties for details on adding preprocessor definitions to specific
scopes and configurations.

See Also
^^^^^^^^

* The :manual:`cmake-buildsystem(7)` manual for more on defining
  buildsystem properties.
