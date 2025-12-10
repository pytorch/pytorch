CTEST_NO_TESTS_ACTION
---------------------

.. versionadded:: 3.26

.. include:: include/ENV_VAR.rst

Environment variable that controls how :manual:`ctest <ctest(1)>` handles
cases when there are no tests to run. Possible values are: ``error``,
``ignore``, empty or unset.

The :option:`--no-tests=\<action\> <ctest --no-tests>` option to
:manual:`ctest <ctest(1)>` overrides this environment variable if both
are given.
