SKIP_RETURN_CODE
----------------

Return code to mark a test as skipped.

Sometimes only a test itself can determine if all requirements for the
test are met. If such a situation should not be considered a hard failure
a return code of the process can be specified that will mark the test as
``Not Run`` if it is encountered. Valid values are in the range of
0 to 255, inclusive.

Tests that exceed the timeout specified by :prop_test:`TIMEOUT` still fail
regardless of ``SKIP_RETURN_CODE``.
System-level test failures including segmentation faults,
signal abort, or heap errors may fail the test even if the return code matches.

.. code-block:: cmake

    # cmake (1) defines this to return code 1
    add_test(NAME r1 COMMAND ${CMAKE_COMMAND} -E false)

    set_tests_properties(r1 PROPERTIES SKIP_RETURN_CODE 1)


To run a test that may have a system-level failure, but still skip if
``SKIP_RETURN_CODE`` matches, use a CMake command to wrap the executable run.
Note that this will prevent automatic handling of the
:prop_tgt:`CROSSCOMPILING_EMULATOR` and :prop_tgt:`TEST_LAUNCHER` target
property.

.. code-block:: cmake

    add_executable(main main.c)

    # cmake -E env <command> returns 1 if the command fails in any way
    add_test(NAME sigabrt COMMAND ${CMAKE_COMMAND} -E env $<TARGET_FILE:main>)

    set_property(TEST sigabrt PROPERTY SKIP_RETURN_CODE 1)

.. code-block:: c

    #include <signal.h>

    int main(void){ raise(SIGABRT); return 0; }


To handle multiple types of cases that may need to be skipped, consider the
:prop_test:`SKIP_REGULAR_EXPRESSION` property.
