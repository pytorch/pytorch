PROCESSORS
----------

Set to specify how many process slots this test requires.
If not set, the default is ``1`` processor.

Denotes the number of processors that this test will require.  This is
typically used for MPI tests, and should be used in conjunction with
the :command:`ctest_test` ``PARALLEL_LEVEL`` option.

This will also be used to display a weighted test timing result in label and
subproject summaries in the command line output of :manual:`ctest(1)`. The wall
clock time for the test run will be multiplied by this property to give a
better idea of how much cpu resource CTest allocated for the test.

See also the :prop_test:`PROCESSOR_AFFINITY` test property.
