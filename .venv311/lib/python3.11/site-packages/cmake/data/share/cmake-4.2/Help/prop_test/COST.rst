COST
----

This property describes the cost of a test.  When parallel testing is
enabled, tests in the test set will be run in descending order of cost.
Projects can explicitly define the cost of a test by setting this property
to a floating point value.

When the cost of a test is not defined by the project,
:manual:`ctest <ctest(1)>` will initially use a default cost of ``0``.
It computes a weighted average of the cost each time a test is run and
uses that as an improved estimate of the cost for the next run.  The more
a test is re-run in the same build directory, the more representative the
cost should become.
