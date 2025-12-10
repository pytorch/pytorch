CMP0109
-------

.. versionadded:: 3.19

:command:`find_program` requires permission to execute but not to read.

In CMake 3.18 and below, the :command:`find_program` command on UNIX
would find files that are readable without requiring execute permission,
and would not find files that are executable without read permission.
In CMake 3.19 and above, ``find_program`` now prefers to require execute
permission but not read permission.  This policy provides compatibility
with projects that have not been updated to expect the new behavior.

The ``OLD`` behavior for this policy is for ``find_program`` to require
read permission but not execute permission.
The ``NEW`` behavior for this policy is for ``find_program`` to require
execute permission but not read permission.

.. |INTRODUCED_IN_CMAKE_VERSION| replace:: 3.19
.. |WARNS_OR_DOES_NOT_WARN| replace:: warns
.. include:: include/STANDARD_ADVICE.rst

.. include:: include/DEPRECATED.rst
