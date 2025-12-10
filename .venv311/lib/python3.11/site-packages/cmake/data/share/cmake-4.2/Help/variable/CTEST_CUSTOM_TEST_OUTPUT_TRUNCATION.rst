CTEST_CUSTOM_TEST_OUTPUT_TRUNCATION
-----------------------------------

.. versionadded:: 3.24

Set the test output truncation mode in case a maximum size is configured
via the :variable:`CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE` or
:variable:`CTEST_CUSTOM_MAXIMUM_FAILED_TEST_OUTPUT_SIZE` variables.
By default the ``tail`` of the output will be truncated. Other possible
values are ``middle`` and ``head``.

.. include:: include/CTEST_CUSTOM_XXX.rst
