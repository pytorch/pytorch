CTEST_PARALLEL_LEVEL
--------------------

.. include:: include/ENV_VAR.rst

Specify the number of tests for CTest to run in parallel.
For example, if ``CTEST_PARALLEL_LEVEL`` is set to 8, CTest will run
up to 8 tests concurrently as if ``ctest`` were invoked with the
:option:`--parallel 8 <ctest --parallel>` option.

.. versionchanged:: 3.29

  The value may be empty, or ``0``, to let ctest use a default level of
  parallelism, or unbounded parallelism, respectively, as documented by
  the :option:`ctest --parallel` option.

  CTest will interpret a whitespace-only string as empty.

  In CMake 3.28 and earlier, an empty or ``0`` value was equivalent to ``1``.

See :manual:`ctest(1)` for more information on parallel test execution.
