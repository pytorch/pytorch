CMAKE_BUILD_PARALLEL_LEVEL
--------------------------

.. versionadded:: 3.12

.. include:: include/ENV_VAR.rst

Specifies the maximum number of concurrent processes to use when building
using the ``cmake --build`` command line
:ref:`Build Tool Mode <Build Tool Mode>`.
For example, if ``CMAKE_BUILD_PARALLEL_LEVEL`` is set to 8, the
underlying build tool will execute up to 8 jobs concurrently as if
``cmake --build`` were invoked with the
:option:`--parallel 8 <cmake--build --parallel>` option.

If this variable is defined empty the native build tool's default number is
used.
