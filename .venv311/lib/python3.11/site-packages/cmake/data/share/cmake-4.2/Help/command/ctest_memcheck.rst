ctest_memcheck
--------------

Perform the :ref:`CTest MemCheck Step` as a :ref:`Dashboard Client`.

.. code-block:: cmake

  ctest_memcheck([BUILD <build-dir>] [APPEND]
                 [START <start-number>]
                 [END <end-number>]
                 [STRIDE <stride-number>]
                 [EXCLUDE <exclude-regex>]
                 [INCLUDE <include-regex>]
                 [EXCLUDE_LABEL <label-exclude-regex>]
                 [INCLUDE_LABEL <label-include-regex>]
                 [EXCLUDE_FIXTURE <regex>]
                 [EXCLUDE_FIXTURE_SETUP <regex>]
                 [EXCLUDE_FIXTURE_CLEANUP <regex>]
                 [PARALLEL_LEVEL <level>]
                 [RESOURCE_SPEC_FILE <file>]
                 [TEST_LOAD <threshold>]
                 [SCHEDULE_RANDOM <ON|OFF>]
                 [STOP_ON_FAILURE]
                 [STOP_TIME <time-of-day>]
                 [RETURN_VALUE <result-var>]
                 [CAPTURE_CMAKE_ERROR <result-var>]
                 [REPEAT <mode>:<n>]
                 [OUTPUT_JUNIT <file>]
                 [DEFECT_COUNT <defect-count-var>]
                 [QUIET]
                 )


Run tests with a dynamic analysis tool and store results in
``MemCheck.xml`` for submission with the :command:`ctest_submit`
command.

Most options are the same as those for the :command:`ctest_test` command.

The options unique to this command are:

``DEFECT_COUNT <defect-count-var>``
  .. versionadded:: 3.8

  Store in the ``<defect-count-var>`` the number of defects found.
