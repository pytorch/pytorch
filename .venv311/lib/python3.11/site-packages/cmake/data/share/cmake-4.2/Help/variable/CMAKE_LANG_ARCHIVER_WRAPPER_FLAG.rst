CMAKE_<LANG>_ARCHIVER_WRAPPER_FLAG
----------------------------------

.. versionadded:: 4.0

Defines the syntax of compiler driver option to pass options to the archiver
tool. It will be used to translate the ``ARCHIVER:`` prefix in the static
library options (see :prop_tgt:`STATIC_LIBRARY_OPTIONS`).

This variable holds a :ref:`semicolon-separated list <CMake Language Lists>` of
tokens. If a space (i.e. " ") is specified as last token, flag and
``ARCHIVER:`` arguments will be specified as separate arguments to the compiler
driver. The :variable:`CMAKE_<LANG>_ARCHIVER_WRAPPER_FLAG_SEP` variable can be
specified to manage concatenation of arguments.

See :variable:`CMAKE_<LANG>_LINKER_WRAPPER_FLAG` variable for examples of
definitions because ``CMAKE_<LANG>_ARCHIVER_WRAPPER_FLAG`` use the same syntax.
