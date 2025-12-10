CTEST_MEMORYCHECK_SANITIZER_OPTIONS
-----------------------------------

.. versionadded:: 3.1

Specify the CTest ``MemoryCheckSanitizerOptions`` setting
in a :manual:`ctest(1)` dashboard client script.

CTest prepends correct sanitizer options ``*_OPTIONS``
environment variable to executed command. CTests adds
its own ``log_path`` to sanitizer options, don't provide your
own ``log_path``.
