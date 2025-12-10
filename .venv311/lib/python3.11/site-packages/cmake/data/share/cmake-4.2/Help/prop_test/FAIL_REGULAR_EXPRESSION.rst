FAIL_REGULAR_EXPRESSION
-----------------------

If the test output (stdout or stderr) matches this regular expression the test
will fail, regardless of the process exit code. Tests that exceed the timeout
specified by :prop_test:`TIMEOUT` fail regardless of
``FAIL_REGULAR_EXPRESSION``. Any non-zero return code or system-level test
failures including segmentation faults, signal abort, or heap errors fail the
test even if the regular expression does not match.

If set, if the output matches one of specified regular expressions, the test
will fail.  Example:

.. code-block:: cmake

  # test would pass, except for FAIL_REGULAR_EXPRESSION
  add_test(NAME mytest COMMAND ${CMAKE_COMMAND} -E echo "Failed")

  set_property(TEST mytest PROPERTY
    FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed"
  )

``FAIL_REGULAR_EXPRESSION`` expects a list of regular expressions.

See also the :prop_test:`PASS_REGULAR_EXPRESSION` and
:prop_test:`SKIP_REGULAR_EXPRESSION` test properties.
