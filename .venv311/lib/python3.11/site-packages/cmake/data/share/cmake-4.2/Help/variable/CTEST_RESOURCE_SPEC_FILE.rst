CTEST_RESOURCE_SPEC_FILE
------------------------

.. versionadded:: 3.18

Specify the CTest ``ResourceSpecFile`` setting in a :manual:`ctest(1)`
dashboard client script.

This can also be used to specify the resource spec file from a CMake build. If
no ``RESOURCE_SPEC_FILE`` is passed to :command:`ctest_test`, and
``CTEST_RESOURCE_SPEC_FILE`` is not specified in the dashboard script, the
value of this variable from the build is used.
