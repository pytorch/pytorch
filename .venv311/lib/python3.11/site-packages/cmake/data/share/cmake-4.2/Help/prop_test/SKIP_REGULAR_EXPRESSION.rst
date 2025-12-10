SKIP_REGULAR_EXPRESSION
-----------------------

.. versionadded:: 3.16

If the test output (stderr or stdout) matches this regular expression the test
will be marked as skipped, regardless of the process exit code. Tests that
exceed the timeout specified by :prop_test:`TIMEOUT` still fail regardless of
``SKIP_REGULAR_EXPRESSION``. System-level test failures including segmentation
faults, signal abort, or heap errors may fail the test even if the regular
expression matches.

Example:

.. code-block:: cmake

  add_test(NAME mytest COMMAND ${CMAKE_COMMAND} -E echo "Skipped this test")

  set_property(TEST mytest PROPERTY
    SKIP_REGULAR_EXPRESSION "[^a-z]Skip" "SKIP" "Skipped"
  )

``SKIP_REGULAR_EXPRESSION`` expects a list of regular expressions.

To run a test that may have a system-level failure, but still skip if
``SKIP_REGULAR_EXPRESSION`` matches, use a CMake command to wrap the
executable run. Note that this will prevent automatic handling of the
:prop_tgt:`CROSSCOMPILING_EMULATOR` and :prop_tgt:`TEST_LAUNCHER`
target property.

.. code-block:: cmake

    add_executable(main main.c)

    add_test(NAME sigabrt COMMAND ${CMAKE_COMMAND} -E env $<TARGET_FILE:main>)

    set_property(TEST sigabrt PROPERTY SKIP_REGULAR_EXPRESSION "SIGABRT;[aA]bort")

.. code-block:: c

    #include <signal.h>

    int main(void){ raise(SIGABRT); return 0; }

See also the :prop_test:`SKIP_RETURN_CODE`,
:prop_test:`PASS_REGULAR_EXPRESSION`, and :prop_test:`FAIL_REGULAR_EXPRESSION`
test properties.
