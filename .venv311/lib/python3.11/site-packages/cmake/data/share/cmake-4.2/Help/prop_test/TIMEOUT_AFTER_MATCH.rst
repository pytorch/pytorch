TIMEOUT_AFTER_MATCH
-------------------

.. versionadded:: 3.6

Change a test's timeout duration after a matching line is encountered
in its output.

Usage
^^^^^

.. code-block:: cmake

 add_test(mytest ...)
 set_property(TEST mytest PROPERTY TIMEOUT_AFTER_MATCH "${seconds}" "${regex}")

Description
^^^^^^^^^^^

Allow a test ``seconds`` to complete after ``regex`` is encountered in
its output.

When the test outputs a line that matches ``regex`` its start time is
reset to the current time and its timeout duration is changed to
``seconds``.  Prior to this, the timeout duration is determined by the
:prop_test:`TIMEOUT` property or the :variable:`CTEST_TEST_TIMEOUT`
variable if either of these are set.  Because the test's start time is
reset, its execution time will not include any time that was spent
waiting for the matching output.

``TIMEOUT_AFTER_MATCH`` is useful for avoiding spurious
timeouts when your test must wait for some system resource to become
available before it can execute.  Set :prop_test:`TIMEOUT` to a longer
duration that accounts for resource acquisition and use
``TIMEOUT_AFTER_MATCH`` to control how long the actual test
is allowed to run.

If the required resource can be controlled by CTest you should use
:prop_test:`RESOURCE_LOCK` instead of ``TIMEOUT_AFTER_MATCH``.
This property should be used when only the test itself can determine
when its required resources are available.

See also :prop_test:`TIMEOUT_SIGNAL_NAME`.
