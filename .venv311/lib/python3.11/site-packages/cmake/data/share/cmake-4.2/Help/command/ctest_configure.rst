ctest_configure
---------------

Perform the :ref:`CTest Configure Step` as a :ref:`Dashboard Client`.

.. code-block:: cmake

  ctest_configure([BUILD <build-dir>] [SOURCE <source-dir>] [APPEND]
                  [OPTIONS <options>] [RETURN_VALUE <result-var>] [QUIET]
                  [CAPTURE_CMAKE_ERROR <result-var>])

Configure the project build tree and record results in ``Configure.xml``
for submission with the :command:`ctest_submit` command.

The options are:

``BUILD <build-dir>``
  Specify the top-level build directory.  If not given, the
  :variable:`CTEST_BINARY_DIRECTORY` variable is used.

``SOURCE <source-dir>``
  Specify the source directory.  If not given, the
  :variable:`CTEST_SOURCE_DIRECTORY` variable is used.

``APPEND``
  Mark ``Configure.xml`` for append to results previously submitted to a
  dashboard server since the last :command:`ctest_start` call.
  Append semantics are defined by the dashboard server in use.
  This does *not* cause results to be appended to a ``.xml`` file
  produced by a previous call to this command.

``OPTIONS <options>``
  Specify command-line arguments to pass to the configuration tool.

``RETURN_VALUE <result-var>``
  Store in the ``<result-var>`` variable the return value of the native
  configuration tool.

``CAPTURE_CMAKE_ERROR <result-var>``
  .. versionadded:: 3.7

  Store in the ``<result-var>`` variable -1 if there are any errors running
  the command and prevent ctest from returning non-zero if an error occurs.

``QUIET``
  .. versionadded:: 3.3

  Suppress any CTest-specific non-error messages that would have
  otherwise been printed to the console.  Output from the underlying
  configure command is not affected.
