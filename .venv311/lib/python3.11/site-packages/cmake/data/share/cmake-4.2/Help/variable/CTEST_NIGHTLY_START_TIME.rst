CTEST_NIGHTLY_START_TIME
------------------------

.. versionadded:: 3.1

Specify the CTest ``NightlyStartTime`` setting in a :manual:`ctest(1)`
dashboard client script.

Note that this variable must always be set for a nightly build in a
dashboard script. It is needed so that nightly builds can be properly grouped
together in CDash.
