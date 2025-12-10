install_programs
----------------

.. deprecated:: 3.0

  Use the :command:`install(PROGRAMS)` command instead.

This command has been superseded by the :command:`install` command.  It is
provided for compatibility with older CMake code.  The ``FILES`` form is
directly replaced by the ``PROGRAMS`` form of the :command:`install`
command.  The regexp form can be expressed more clearly using the ``GLOB``
form of the :command:`file` command.

.. code-block:: cmake

  install_programs(<dir> file1 file2 [file3 ...])
  install_programs(<dir> FILES file1 [file2 ...])

Create rules to install the listed programs into the given directory.
Use the ``FILES`` argument to guarantee that the file list version of the
command will be used even when there is only one argument.

.. code-block:: cmake

  install_programs(<dir> regexp)

In the second form any program in the current source directory that
matches the regular expression will be installed.

This command is intended to install programs that are not built by
cmake, such as shell scripts.  See the ``TARGETS`` form of the
:command:`install` command to create installation rules for targets built
by cmake.

The directory ``<dir>`` is relative to the installation prefix, which is
stored in the variable :variable:`CMAKE_INSTALL_PREFIX`.
