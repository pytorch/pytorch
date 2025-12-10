WILL_FAIL
---------

If ``true``, inverts the pass / fail test criteria. Tests for which
``WILL_FAIL`` is ``true`` fail with return code 0 and pass with non-zero
return code. Tests that exceed the timeout specified by :prop_test:`TIMEOUT`
still fail regardless of ``WILL_FAIL``.
System-level test failures including segmentation faults,
signal abort, or heap errors may fail the test even if ``WILL_FAIL`` is true.

Example of a test that would ordinarily pass, but fails because ``WILL_FAIL``
is ``true``:

.. code-block:: cmake

    add_test(NAME failed COMMAND ${CMAKE_COMMAND} -E true)
    set_property(TEST failed PROPERTY WILL_FAIL true)

To run a test that may have a system-level failure, but still pass if
``WILL_FAIL`` is set, use a CMake command to wrap the executable run.
Note that this will prevent automatic handling of the
:prop_tgt:`CROSSCOMPILING_EMULATOR` and :prop_tgt:`TEST_LAUNCHER`
target property.

.. code-block:: cmake

    add_executable(main main.c)

    add_test(NAME sigabrt COMMAND ${CMAKE_COMMAND} -E env $<TARGET_FILE:main>)

    set_property(TEST sigabrt PROPERTY WILL_FAIL TRUE)

.. code-block:: c

    #include <signal.h>

    int main(void){ raise(SIGABRT); return 0; }
