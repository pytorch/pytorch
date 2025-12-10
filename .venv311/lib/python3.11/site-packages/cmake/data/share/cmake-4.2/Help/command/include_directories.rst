include_directories
-------------------

Add include directories to the build.

.. code-block:: cmake

  include_directories([AFTER|BEFORE] [SYSTEM] dir1 [dir2 ...])

Add the given directories to those the compiler uses to search for
include files.  Relative paths are interpreted as relative to the
current source directory.

The include directories are added to the :prop_dir:`INCLUDE_DIRECTORIES`
directory property for the current ``CMakeLists`` file.  They are also
added to the :prop_tgt:`INCLUDE_DIRECTORIES` target property for each
target in the current ``CMakeLists`` file.  The target property values
are the ones used by the generators.

By default the directories specified are appended onto the current list of
directories.  This default behavior can be changed by setting
:variable:`CMAKE_INCLUDE_DIRECTORIES_BEFORE` to ``ON``.  By using
``AFTER`` or ``BEFORE`` explicitly, you can select between appending and
prepending, independent of the default.

If the ``SYSTEM`` option is given, the compiler will be told the
directories are meant as system include directories on some platforms.
Signaling this setting might achieve effects such as the compiler
skipping warnings, or these fixed-install system files not being
considered in dependency calculations - see compiler docs.

.. |command_name| replace:: ``include_directories``
.. include:: include/GENEX_NOTE.rst

.. note::

  Prefer the :command:`target_include_directories` command to add include
  directories to individual targets and optionally propagate/export them
  to dependents.

See Also
^^^^^^^^

* :command:`target_include_directories`
