FIXTURES_CLEANUP
----------------

.. versionadded:: 3.7

Specifies a list of fixtures for which the test is to be treated as a cleanup
test. These fixture names are distinct from test case names and are not
required to have any similarity to the names of tests associated with them.

Fixture cleanup tests are ordinary tests with all of the usual test
functionality. Setting the ``FIXTURES_CLEANUP`` property for a test has two
primary effects:

- CTest will ensure the test executes after all other tests which list any of
  the fixtures in its :prop_test:`FIXTURES_REQUIRED` property.

- If CTest is asked to run only a subset of tests (e.g. using regular
  expressions or the ``--rerun-failed`` option) and the cleanup test is not in
  the set of tests to run, it will automatically be added if any tests in the
  set require any fixture listed in ``FIXTURES_CLEANUP``.

A cleanup test can have multiple fixtures listed in its ``FIXTURES_CLEANUP``
property. It will execute only once for the whole CTest run, not once for each
fixture. A fixture can also have more than one cleanup test defined. If there
are multiple cleanup tests for a fixture, projects can control their order with
the usual :prop_test:`DEPENDS` test property if necessary.

A cleanup test is allowed to require other fixtures, but not any fixture listed
in its ``FIXTURES_CLEANUP`` property. For example:

.. code-block:: cmake

  # Ok: Dependent fixture is different to cleanup
  set_tests_properties(cleanupFoo PROPERTIES
    FIXTURES_CLEANUP  Foo
    FIXTURES_REQUIRED Bar
  )

  # Error: cannot require same fixture as cleanup
  set_tests_properties(cleanupFoo PROPERTIES
    FIXTURES_CLEANUP  Foo
    FIXTURES_REQUIRED Foo
  )

Cleanup tests will execute even if setup or regular tests for that fixture fail
or are skipped.

See :prop_test:`FIXTURES_REQUIRED` for a more complete discussion of how to use
test fixtures.
