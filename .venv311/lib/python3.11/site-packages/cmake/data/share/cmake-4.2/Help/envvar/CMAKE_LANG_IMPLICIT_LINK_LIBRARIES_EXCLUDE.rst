CMAKE_<LANG>_IMPLICIT_LINK_LIBRARIES_EXCLUDE
--------------------------------------------

.. versionadded:: 4.1

.. include:: include/ENV_VAR.rst

A :ref:`semicolon-separated list <CMake Language Lists>` of libraries
to exclude from the :variable:`CMAKE_<LANG>_IMPLICIT_LINK_LIBRARIES`
variable when it is automatically detected from the ``<LANG>`` compiler.

This may be used to work around detection limitations that result in
extraneous implicit link libraries, e.g., when using compiler driver
flags that affect the set of implicitly linked libraries.

See also the :envvar:`CMAKE_<LANG>_IMPLICIT_LINK_DIRECTORIES_EXCLUDE`
environment variable.
