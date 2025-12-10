FIXTURES_REQUIRED
-----------------

.. versionadded:: 3.7

Specifies a list of fixtures the test requires. Fixture names are case
sensitive and they are not required to have any similarity to test names.

Fixtures are a way to attach setup and cleanup tasks to a set of tests. If a
test requires a given fixture, then all tests marked as setup tasks for that
fixture will be executed first (once for the whole set of tests, not once per
test requiring the fixture). After all tests requiring a particular fixture
have completed, CTest will ensure all tests marked as cleanup tasks for that
fixture are then executed. Tests are marked as setup tasks with the
:prop_test:`FIXTURES_SETUP` property and as cleanup tasks with the
:prop_test:`FIXTURES_CLEANUP` property. If any of a fixture's setup tests fail,
all tests listing that fixture in their ``FIXTURES_REQUIRED`` property will not
be executed. The cleanup tests for the fixture will always be executed, even if
some setup tests fail.

When CTest is asked to execute only a subset of tests (e.g. by the use of
regular expressions or when run with the :option:`--rerun-failed <ctest --rerun-failed>`
command line option), it will automatically add any setup or cleanup tests for
fixtures required by any of the tests that are in the execution set. This
behavior can be overridden with the :option:`-FS <ctest -FS>`,
:option:`-FC <ctest -FC>` and :option:`-FA <ctest -FA>` command line options to
:manual:`ctest(1)` if desired.

Since setup and cleanup tasks are also tests, they can have an ordering
specified by the :prop_test:`DEPENDS` test property just like any other tests.
This can be exploited to implement setup or cleanup using multiple tests for a
single fixture to modularise setup or cleanup logic.

The concept of a fixture is different to that of a resource specified by
:prop_test:`RESOURCE_LOCK`, but they may be used together. A fixture defines a
set of tests which share setup and cleanup requirements, whereas a resource
lock has the effect of ensuring a particular set of tests do not run in
parallel. Some situations may need both, such as setting up a database,
serializing test access to that database and deleting the database again at the
end. For such cases, tests would populate both ``FIXTURES_REQUIRED`` and
:prop_test:`RESOURCE_LOCK` to combine the two behaviors. Names used for
:prop_test:`RESOURCE_LOCK` have no relationship with names of fixtures, so note
that a resource lock does not imply a fixture and vice versa.

Consider the following example which represents a database test scenario
similar to that mentioned above:

.. code-block:: cmake

  add_test(NAME testsDone   COMMAND emailResults)
  add_test(NAME fooOnly     COMMAND testFoo)
  add_test(NAME dbOnly      COMMAND testDb)
  add_test(NAME dbWithFoo   COMMAND testDbWithFoo)
  add_test(NAME createDB    COMMAND initDB)
  add_test(NAME setupUsers  COMMAND userCreation)
  add_test(NAME cleanupDB   COMMAND deleteDB)
  add_test(NAME cleanupFoo  COMMAND removeFoos)

  set_tests_properties(setupUsers PROPERTIES DEPENDS createDB)

  set_tests_properties(createDB   PROPERTIES FIXTURES_SETUP    DB)
  set_tests_properties(setupUsers PROPERTIES FIXTURES_SETUP    DB)
  set_tests_properties(cleanupDB  PROPERTIES FIXTURES_CLEANUP  DB)
  set_tests_properties(cleanupFoo PROPERTIES FIXTURES_CLEANUP  Foo)
  set_tests_properties(testsDone  PROPERTIES FIXTURES_CLEANUP  "DB;Foo")

  set_tests_properties(fooOnly    PROPERTIES FIXTURES_REQUIRED Foo)
  set_tests_properties(dbOnly     PROPERTIES FIXTURES_REQUIRED DB)
  set_tests_properties(dbWithFoo  PROPERTIES FIXTURES_REQUIRED "DB;Foo")

  set_tests_properties(dbOnly dbWithFoo createDB setupUsers cleanupDB
                       PROPERTIES RESOURCE_LOCK DbAccess)

Key points from this example:

- Two fixtures are defined: ``DB`` and ``Foo``. Tests can require a single
  fixture as ``fooOnly`` and ``dbOnly`` do, or they can depend on multiple
  fixtures like ``dbWithFoo`` does.

- A ``DEPENDS`` relationship is set up to ensure ``setupUsers`` happens after
  ``createDB``, both of which are setup tests for the ``DB`` fixture and will
  therefore be executed before the ``dbOnly`` and ``dbWithFoo`` tests
  automatically.

- No explicit ``DEPENDS`` relationships were needed to make the setup tests run
  before or the cleanup tests run after the regular tests.

- The ``Foo`` fixture has no setup tests defined, only a single cleanup test.

- ``testsDone`` is a cleanup test for both the ``DB`` and ``Foo`` fixtures.
  Therefore, it will only execute once regular tests for both fixtures have
  finished (i.e. after ``fooOnly``, ``dbOnly`` and ``dbWithFoo``). No
  ``DEPENDS`` relationship was specified for ``testsDone``, so it is free to
  run before, after or concurrently with other cleanup tests for either
  fixture.

- The setup and cleanup tests never list the fixtures they are for in their own
  ``FIXTURES_REQUIRED`` property, as that would result in a dependency on
  themselves and be considered an error.
