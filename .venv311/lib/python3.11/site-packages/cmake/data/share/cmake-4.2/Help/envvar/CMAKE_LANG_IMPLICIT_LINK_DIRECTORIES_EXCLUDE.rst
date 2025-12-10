CMAKE_<LANG>_IMPLICIT_LINK_DIRECTORIES_EXCLUDE
----------------------------------------------

.. versionadded:: 3.27

.. include:: include/ENV_VAR.rst

A :ref:`semicolon-separated list <CMake Language Lists>` of directories
to exclude from the :variable:`CMAKE_<LANG>_IMPLICIT_LINK_DIRECTORIES`
variable when it is automatically detected from the ``<LANG>`` compiler.

This may be used to work around misconfigured compiler drivers that pass
extraneous implicit link directories to their linker.

See also the :envvar:`CMAKE_<LANG>_IMPLICIT_LINK_LIBRARIES_EXCLUDE`
environment variable.
