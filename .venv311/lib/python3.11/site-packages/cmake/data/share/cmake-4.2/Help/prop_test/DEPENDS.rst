DEPENDS
-------

Specifies that this test should only be run after the specified list of tests.

Set this to a list of tests that must finish before this test is run. The
results of those tests are not considered, the dependency relationship is
purely for order of execution (i.e. it is really just a *run after*
relationship). Consider using test fixtures with setup tests if a dependency
with successful completion is required (see :prop_test:`FIXTURES_REQUIRED`).

Examples
~~~~~~~~

.. code-block:: cmake

  add_test(NAME baseTest1 ...)
  add_test(NAME baseTest2 ...)
  add_test(NAME dependsTest12 ...)

  set_tests_properties(dependsTest12 PROPERTIES DEPENDS "baseTest1;baseTest2")
  # dependsTest12 runs after baseTest1 and baseTest2, even if they fail
