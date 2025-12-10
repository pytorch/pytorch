ctest_test
----------

Perform the :ref:`CTest Test Step` as a :ref:`Dashboard Client`.

.. code-block:: cmake

  ctest_test([BUILD <build-dir>] [APPEND]
             [START <start-number>]
             [END <end-number>]
             [STRIDE <stride-number>]
             [EXCLUDE <exclude-regex>]
             [INCLUDE <include-regex>]
             [EXCLUDE_LABEL <label-exclude-regex>]
             [INCLUDE_LABEL <label-include-regex>]
             [EXCLUDE_FROM_FILE <filename>]
             [INCLUDE_FROM_FILE <filename>]
             [EXCLUDE_FIXTURE <regex>]
             [EXCLUDE_FIXTURE_SETUP <regex>]
             [EXCLUDE_FIXTURE_CLEANUP <regex>]
             [PARALLEL_LEVEL [<level>]]
             [RESOURCE_SPEC_FILE <file>]
             [TEST_LOAD <threshold>]
             [SCHEDULE_RANDOM <ON|OFF>]
             [STOP_ON_FAILURE]
             [STOP_TIME <time-of-day>]
             [RETURN_VALUE <result-var>]
             [CAPTURE_CMAKE_ERROR <result-var>]
             [REPEAT <mode>:<n>]
             [OUTPUT_JUNIT <file>]
             [QUIET]
             )

..
   NOTE If updating the argument list here, please also update the argument
   list documentation for :command:`ctest_memcheck` as well.

Run tests in the project build tree and store results in
``Test.xml`` for submission with the :command:`ctest_submit` command.

The options are:

``BUILD <build-dir>``
  Specify the top-level build directory.  If not given, the
  :variable:`CTEST_BINARY_DIRECTORY` variable is used.

``APPEND``
  Mark ``Test.xml`` for append to results previously submitted to a
  dashboard server since the last :command:`ctest_start` call.
  Append semantics are defined by the dashboard server in use.
  This does *not* cause results to be appended to a ``.xml`` file
  produced by a previous call to this command.

``START <start-number>``
  Specify the beginning of a range of test numbers.

``END <end-number>``
  Specify the end of a range of test numbers.

``STRIDE <stride-number>``
  Specify the stride by which to step across a range of test numbers.

``EXCLUDE <exclude-regex>``
  Specify a regular expression matching test names to exclude.

``INCLUDE <include-regex>``
  Specify a regular expression matching test names to include.
  Tests not matching this expression are excluded.

``EXCLUDE_LABEL <label-exclude-regex>``
  Specify a regular expression matching test labels to exclude.

``INCLUDE_LABEL <label-include-regex>``
  Specify a regular expression matching test labels to include.
  Tests not matching this expression are excluded.

``EXCLUDE_FROM_FILE <filename>``
  .. versionadded:: 3.29

  Do NOT run tests listed with their exact name in the given file.

``INCLUDE_FROM_FILE <filename>``
  .. versionadded:: 3.29

  Only run the tests listed with their exact name in the given file.

``EXCLUDE_FIXTURE <regex>``
  .. versionadded:: 3.7

  If a test in the set of tests to be executed requires a particular fixture,
  that fixture's setup and cleanup tests would normally be added to the test
  set automatically. This option prevents adding setup or cleanup tests for
  fixtures matching the ``<regex>``. Note that all other fixture behavior is
  retained, including test dependencies and skipping tests that have fixture
  setup tests that fail.

``EXCLUDE_FIXTURE_SETUP <regex>``
  .. versionadded:: 3.7

  Same as ``EXCLUDE_FIXTURE`` except only matching setup tests are excluded.

``EXCLUDE_FIXTURE_CLEANUP <regex>``
  .. versionadded:: 3.7

  Same as ``EXCLUDE_FIXTURE`` except only matching cleanup tests are excluded.

``PARALLEL_LEVEL [<level>]``
  Run tests in parallel, limited to a given level of parallelism.

  .. versionadded:: 3.29

    The ``<level>`` may be omitted, or ``0``, to let ctest use a default
    level of parallelism, or unbounded parallelism, respectively, as
    documented by the :option:`ctest --parallel` option.

``RESOURCE_SPEC_FILE <file>``
  .. versionadded:: 3.16

  Specify a
  :ref:`resource specification file <ctest-resource-specification-file>`. See
  :ref:`ctest-resource-allocation` for more information.

``TEST_LOAD <threshold>``
  .. versionadded:: 3.4

  While running tests in parallel, try not to start tests when they
  may cause the CPU load to pass above a given threshold.  If not
  specified the :variable:`CTEST_TEST_LOAD` variable will be checked,
  and then the :option:`--test-load <ctest --test-load>` command-line
  argument to :manual:`ctest(1)`. See also the ``TestLoad`` setting
  in the :ref:`CTest Test Step`.

``REPEAT <mode>:<n>``
  .. versionadded:: 3.17

  Run tests repeatedly based on the given ``<mode>`` up to ``<n>`` times.
  The modes are:

  ``UNTIL_FAIL``
    Require each test to run ``<n>`` times without failing in order to pass.
    This is useful in finding sporadic failures in test cases.

  ``UNTIL_PASS``
    Allow each test to run up to ``<n>`` times in order to pass.
    Repeats tests if they fail for any reason.
    This is useful in tolerating sporadic failures in test cases.

  ``AFTER_TIMEOUT``
    Allow each test to run up to ``<n>`` times in order to pass.
    Repeats tests only if they timeout.
    This is useful in tolerating sporadic timeouts in test cases
    on busy machines.

``SCHEDULE_RANDOM <ON|OFF>``
  Launch tests in a random order.  This may be useful for detecting
  implicit test dependencies.

``STOP_ON_FAILURE``
  .. versionadded:: 3.18

  Stop the execution of the tests once one has failed.

``STOP_TIME <time-of-day>``
  Specify a time of day at which the tests should all stop running.

``RETURN_VALUE <result-var>``
  Store in the ``<result-var>`` variable ``0`` if all tests passed.
  Store non-zero if anything went wrong.

``CAPTURE_CMAKE_ERROR <result-var>``
  .. versionadded:: 3.7

  Store in the ``<result-var>`` variable -1 if there are any errors running
  the command and prevent ctest from returning non-zero if an error occurs.

``OUTPUT_JUNIT <file>``
  .. versionadded:: 3.21

  Write test results to ``<file>`` in JUnit XML format. If ``<file>`` is a
  relative path, it will be placed in the build directory. If ``<file>``
  already exists, it will be overwritten. Note that the resulting JUnit XML
  file is **not** uploaded to CDash because it would be redundant with
  CTest's ``Test.xml`` file.

``QUIET``
  .. versionadded:: 3.3

  Suppress any CTest-specific non-error messages that would have otherwise
  been printed to the console.  Output from the underlying test command is not
  affected.  Summary info detailing the percentage of passing tests is also
  unaffected by the ``QUIET`` option.

See also the :variable:`CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE`,
:variable:`CTEST_CUSTOM_MAXIMUM_FAILED_TEST_OUTPUT_SIZE` and
:variable:`CTEST_CUSTOM_TEST_OUTPUT_TRUNCATION` variables, along with their
corresponding :manual:`ctest(1)` command line options
:option:`--test-output-size-passed <ctest --test-output-size-passed>`,
:option:`--test-output-size-failed <ctest --test-output-size-failed>`, and
:option:`--test-output-truncation <ctest --test-output-truncation>`.

.. _`Additional Test Measurements`:

Additional Test Measurements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CTest can parse the output of your tests for extra measurements to report
to CDash.

When run as a :ref:`Dashboard Client`, CTest will include these custom
measurements in the ``Test.xml`` file that gets uploaded to CDash.

Check the `CDash test measurement documentation
<https://github.com/Kitware/CDash/blob/master/docs/test_measurements.md>`_
for more information on the types of test measurements that CDash recognizes.

.. versionadded:: 3.22
  CTest can parse custom measurements from tags named
  ``<CTestMeasurement>`` or ``<CTestMeasurementFile>``. The older names
  ``<DartMeasurement>`` and ``<DartMeasurementFile>`` are still supported.

The following example demonstrates how to output a variety of custom test
measurements.

.. code-block:: c++

   std::cout <<
     "<CTestMeasurement type=\"numeric/double\" name=\"score\">28.3</CTestMeasurement>"
     << std::endl;

   std::cout <<
     "<CTestMeasurement type=\"text/string\" name=\"color\">red</CTestMeasurement>"
     << std::endl;

   std::cout <<
     "<CTestMeasurement type=\"text/link\" name=\"CMake URL\">https://cmake.org</CTestMeasurement>"
     << std::endl;

   std::cout <<
     "<CTestMeasurement type=\"text/preformatted\" name=\"Console Output\">" <<
     "line 1.\n" <<
     "  \033[31;1m line 2. Bold red, and indented!\033[0;0ml\n" <<
     "line 3. Not bold or indented...\n" <<
     "</CTestMeasurement>" << std::endl;

Image Measurements
""""""""""""""""""

The following example demonstrates how to upload test images to CDash.

.. code-block:: c++

   std::cout <<
     "<CTestMeasurementFile type=\"image/jpg\" name=\"TestImage\">" <<
     "/dir/to/test_img.jpg</CTestMeasurementFile>" << std::endl;

   std::cout <<
     "<CTestMeasurementFile type=\"image/gif\" name=\"ValidImage\">" <<
     "/dir/to/valid_img.gif</CTestMeasurementFile>" << std::endl;

   std::cout <<
     "<CTestMeasurementFile type=\"image/png\" name=\"AlgoResult\">" <<
     "/dir/to/img.png</CTestMeasurementFile>"
     << std::endl;

Images will be displayed together in an interactive comparison mode on CDash
if they are provided with two or more of the following names.

* ``TestImage``
* ``ValidImage``
* ``BaselineImage``
* ``DifferenceImage2``

By convention, ``TestImage`` is the image generated by your test, and
``ValidImage`` (or ``BaselineImage``) is basis of comparison used to determine
if the test passed or failed.

If another image name is used it will be displayed by CDash as a static image
separate from the interactive comparison UI.

Attached Files
""""""""""""""

.. versionadded:: 3.21

The following example demonstrates how to upload non-image files to CDash.

.. code-block:: c++

   std::cout <<
     "<CTestMeasurementFile type=\"file\" name=\"TestInputData1\">" <<
     "/dir/to/data1.csv</CTestMeasurementFile>\n"                   <<
     "<CTestMeasurementFile type=\"file\" name=\"TestInputData2\">" <<
     "/dir/to/data2.csv</CTestMeasurementFile>"                     << std::endl;

If the name of the file to upload is known at configure time, you can use the
:prop_test:`ATTACHED_FILES` or :prop_test:`ATTACHED_FILES_ON_FAIL` test
properties instead.

Custom Details
""""""""""""""

.. versionadded:: 3.21

The following example demonstrates how to specify a custom value for the
``Test Details`` field displayed on CDash.

.. code-block:: c++

   std::cout <<
     "<CTestDetails>My Custom Details Value</CTestDetails>" << std::endl;

.. _`Additional Labels`:

Additional Labels
"""""""""""""""""

.. versionadded:: 3.22

The following example demonstrates how to add additional labels to a test
at runtime.

.. code-block:: c++

   std::cout <<
     "<CTestLabel>Custom Label 1</CTestLabel>\n" <<
     "<CTestLabel>Custom Label 2</CTestLabel>"   << std::endl;

Use the :prop_test:`LABELS` test property instead for labels that can be
determined at configure time.
