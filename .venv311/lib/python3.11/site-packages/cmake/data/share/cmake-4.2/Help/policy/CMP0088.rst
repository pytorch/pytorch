CMP0088
-------

.. versionadded:: 3.14

:module:`FindBISON` runs bison in :variable:`CMAKE_CURRENT_BINARY_DIR`
when executing.

The module provides a ``bison_target()`` command which generates BISON output.
In CMake 3.13 and below the command would generate a custom build rule that runs
``bison`` in the source directory.  CMake 3.14 and later prefer to run it
in the build directory and use :variable:`CMAKE_CURRENT_BINARY_DIR` as the
``WORKING_DIRECTORY`` of its :command:`add_custom_command` invocation.
This ensures that any implicitly generated file is written to the build
tree rather than the source.

This policy provides compatibility for projects that have not been updated
to expect the new behavior.

The ``OLD`` behavior for this policy is for ``bison_target()`` to use
the current source directory for the ``WORKING_DIRECTORY`` and where
to generate implicit files. The ``NEW`` behavior of this policy is to
use the current binary directory for the ``WORKING_DIRECTORY`` and where
to generate implicit files.

.. |INTRODUCED_IN_CMAKE_VERSION| replace:: 3.14
.. |WARNS_OR_DOES_NOT_WARN| replace:: does *not* warn
.. include:: include/STANDARD_ADVICE.rst

.. include:: include/DEPRECATED.rst
