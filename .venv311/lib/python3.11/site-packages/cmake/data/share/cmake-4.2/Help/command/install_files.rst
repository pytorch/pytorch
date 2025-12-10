install_files
-------------

.. deprecated:: 3.0

  Use the :command:`install(FILES)` command instead.

This command has been superseded by the :command:`install` command.  It is
provided for compatibility with older CMake code.  The ``FILES`` form is
directly replaced by the ``FILES`` form of the :command:`install`
command.  The regexp form can be expressed more clearly using the ``GLOB``
form of the :command:`file` command.

.. code-block:: cmake

  install_files(<dir> extension file file ...)

Create rules to install the listed files with the given extension into
the given directory.  Only files existing in the current source tree
or its corresponding location in the binary tree may be listed.  If a
file specified already has an extension, that extension will be
removed first.  This is useful for providing lists of source files
such as foo.cxx when you want the corresponding foo.h to be installed.
A typical extension is ``.h``.

.. code-block:: cmake

  install_files(<dir> regexp)

Any files in the current source directory that match the regular
expression will be installed.

.. code-block:: cmake

  install_files(<dir> FILES file file ...)

Any files listed after the ``FILES`` keyword will be installed explicitly
from the names given.  Full paths are allowed in this form.

The directory ``<dir>`` is relative to the installation prefix, which is
stored in the variable :variable:`CMAKE_INSTALL_PREFIX`.
