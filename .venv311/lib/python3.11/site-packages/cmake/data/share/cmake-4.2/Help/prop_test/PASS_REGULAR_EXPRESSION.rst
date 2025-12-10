PASS_REGULAR_EXPRESSION
-----------------------

The test output (stdout or stderr) must match this regular expression
for the test to pass. The process exit code is ignored. Tests that exceed
the timeout specified by :prop_test:`TIMEOUT` still fail regardless of
``PASS_REGULAR_EXPRESSION``. System-level test failures including
segmentation faults, signal abort, or heap errors may fail the test even
if ``PASS_REGULAR_EXPRESSION`` is matched.

Example:

.. code-block:: cmake

  add_test(NAME mytest COMMAND ${CMAKE_COMMAND} -E echo "Passed this test")

  set_property(TEST mytest PROPERTY
    PASS_REGULAR_EXPRESSION "pass;Passed"
  )

``PASS_REGULAR_EXPRESSION`` expects a list of regular expressions.

To run a test that may have a system-level failure, but still pass if
``PASS_REGULAR_EXPRESSION`` matches, use a CMake command to wrap the
executable run. Note that this will prevent automatic handling of the
:prop_tgt:`CROSSCOMPILING_EMULATOR` and :prop_tgt:`TEST_LAUNCHER`
target property.

.. code-block:: cmake

    add_executable(main main.c)

    add_test(NAME sigabrt COMMAND ${CMAKE_COMMAND} -E env $<TARGET_FILE:main>)

    set_property(TEST sigabrt PROPERTY PROPERTY_REGULAR_EXPRESSION "pass;Passed")

.. code-block:: c

    #include <signal.h>
    #include <stdio.h>

    int main(void){
        fprintf(stdout, "Passed\n");
        fflush(stdout);  /* ensure the output buffer is seen */
        raise(SIGABRT);
        return 0;
    }

See also the :prop_test:`FAIL_REGULAR_EXPRESSION` and
:prop_test:`SKIP_REGULAR_EXPRESSION` test properties.
