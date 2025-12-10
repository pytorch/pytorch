DISABLED
--------

.. versionadded:: 3.9

If set to ``True``, the test will be skipped and its status will be 'Not Run'. A
``DISABLED`` test will not be counted in the total number of tests and its
completion status will be reported to CDash as ``Disabled``.

A ``DISABLED`` test does not participate in test fixture dependency resolution.
If a ``DISABLED`` test has fixture requirements defined in its
:prop_test:`FIXTURES_REQUIRED` property, it will not cause setup or cleanup
tests for those fixtures to be added to the test set.

If a test with the :prop_test:`FIXTURES_SETUP` property set is ``DISABLED``,
the fixture behavior will be as though that setup test was passing and any test
case requiring that fixture will still run.
