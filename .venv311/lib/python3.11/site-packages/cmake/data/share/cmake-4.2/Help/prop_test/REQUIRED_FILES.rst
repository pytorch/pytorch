REQUIRED_FILES
--------------

List of files required to run the test.  The filenames are relative to the
test :prop_test:`WORKING_DIRECTORY` unless an absolute path is specified.

If set to a list of files, the test will not be run unless all of the
files exist.

Examples
~~~~~~~~

Suppose that ``test.txt`` is created by test ``baseTest`` and ``none.txt``
does not exist:

.. code-block:: cmake

  add_test(NAME baseTest ...)   # Assumed to create test.txt
  add_test(NAME fileTest ...)

  # The following ensures that if baseTest is successful, test.txt will
  # have been created before fileTest is run
  set_tests_properties(fileTest PROPERTIES
    DEPENDS baseTest
    REQUIRED_FILES test.txt
  )

  add_test(NAME notRunTest ...)

  # The following makes notRunTest depend on two files. Nothing creates
  # the none.txt file, so notRunTest will fail with status "Not Run".
  set_tests_properties(notRunTest PROPERTIES
    REQUIRED_FILES "test.txt;none.txt"
  )

The above example demonstrates how ``REQUIRED_FILES`` works, but it is not the
most robust way to implement test ordering with failure detection.  For that,
test fixtures are a better alternative (see :prop_test:`FIXTURES_REQUIRED`).
