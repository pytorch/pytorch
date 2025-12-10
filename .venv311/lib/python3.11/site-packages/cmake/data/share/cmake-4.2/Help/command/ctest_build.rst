ctest_build
-----------

Perform the :ref:`CTest Build Step` as a :ref:`Dashboard Client`.

.. code-block:: cmake

  ctest_build([BUILD <build-dir>] [APPEND]
              [CONFIGURATION <config>]
              [PARALLEL_LEVEL <parallel>]
              [FLAGS <flags>]
              [PROJECT_NAME <project-name>]
              [TARGET <target-name>]
              [NUMBER_ERRORS <num-err-var>]
              [NUMBER_WARNINGS <num-warn-var>]
              [RETURN_VALUE <result-var>]
              [CAPTURE_CMAKE_ERROR <result-var>]
              )

Build the project and store results in ``Build.xml``
for submission with the :command:`ctest_submit` command.

The :variable:`CTEST_BUILD_COMMAND` variable may be set to explicitly
specify the build command line.  Otherwise the build command line is
computed automatically based on the options given.

The options are:

``BUILD <build-dir>``
  Specify the top-level build directory.  If not given, the
  :variable:`CTEST_BINARY_DIRECTORY` variable is used.

``APPEND``
  Mark ``Build.xml`` for append to results previously submitted to a
  dashboard server since the last :command:`ctest_start` call.
  Append semantics are defined by the dashboard server in use.
  This does *not* cause results to be appended to a ``.xml`` file
  produced by a previous call to this command.

``CONFIGURATION <config>``
  Specify the build configuration (e.g. ``Debug``).  If not
  specified the ``CTEST_BUILD_CONFIGURATION`` variable will be checked.
  Otherwise the :option:`-C \<cfg\> <ctest -C>` option given to the
  :manual:`ctest(1)` command will be used, if any.

``PARALLEL_LEVEL <parallel>``
  .. versionadded:: 3.21

  Specify the parallel level of the underlying build system.  If not
  specified, the :envvar:`CMAKE_BUILD_PARALLEL_LEVEL` environment
  variable will be checked.

``FLAGS <flags>``
  Pass additional arguments to the underlying build command.
  If not specified the ``CTEST_BUILD_FLAGS`` variable will be checked.
  This can, e.g., be used to trigger a parallel build using the
  ``-j`` option of ``make``. See the :module:`ProcessorCount` module
  for an example.

``PROJECT_NAME <project-name>``
  Ignored since CMake 3.0.

  .. versionchanged:: 3.14
    This value is no longer required.

``TARGET <target-name>``
  Specify the name of a target to build.  If not specified the
  ``CTEST_BUILD_TARGET`` variable will be checked.  Otherwise the
  default target will be built.  This is the "all" target
  (called ``ALL_BUILD`` in :ref:`Visual Studio Generators`).

``NUMBER_ERRORS <num-err-var>``
  Store the number of build errors detected in the given variable.

``NUMBER_WARNINGS <num-warn-var>``
  Store the number of build warnings detected in the given variable.

``RETURN_VALUE <result-var>``
  Store the return value of the native build tool in the given variable.

``CAPTURE_CMAKE_ERROR <result-var>``
  .. versionadded:: 3.7

  Store in the ``<result-var>`` variable -1 if there are any errors running
  the command and prevent ctest from returning non-zero if an error occurs.

``QUIET``
  .. versionadded:: 3.3

  Suppress any CTest-specific non-error output that would have been
  printed to the console otherwise.  The summary of warnings / errors,
  as well as the output from the native build tool is unaffected by
  this option.
