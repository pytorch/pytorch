SUBDIRECTORIES
--------------

.. versionadded:: 3.7

This read-only directory property contains a
:ref:`semicolon-separated list <CMake Language Lists>` of subdirectories processed so far by
the :command:`add_subdirectory` or :command:`subdirs` commands.  Each entry is
the absolute path to the source directory (containing the ``CMakeLists.txt``
file).  This is suitable to pass to the :command:`get_property` command
``DIRECTORY`` option.

.. note::

  The :command:`subdirs` command does not process its arguments until
  after the calling directory is fully processed.  Therefore looking
  up this property in the current directory will not see them.
