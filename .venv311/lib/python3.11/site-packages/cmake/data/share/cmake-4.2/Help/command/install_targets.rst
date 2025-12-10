install_targets
---------------

.. deprecated:: 3.0

  Use the :command:`install(TARGETS)` command instead.

This command has been superseded by the :command:`install` command.  It is
provided for compatibility with older CMake code.

.. code-block:: cmake

  install_targets(<dir> [RUNTIME_DIRECTORY dir] target target)

Create rules to install the listed targets into the given directory.
The directory ``<dir>`` is relative to the installation prefix, which is
stored in the variable :variable:`CMAKE_INSTALL_PREFIX`.  If
``RUNTIME_DIRECTORY`` is specified, then on systems with special runtime
files (Windows DLL), the files will be copied to that directory.
