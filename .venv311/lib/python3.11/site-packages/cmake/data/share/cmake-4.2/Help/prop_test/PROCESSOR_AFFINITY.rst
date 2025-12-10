PROCESSOR_AFFINITY
------------------

.. versionadded:: 3.12

Set to a true value to ask CTest to launch the test process with CPU affinity
for a fixed set of processors.  If enabled and supported for the current
platform, CTest will choose a set of processors to place in the CPU affinity
mask when launching the test process.  The number of processors in the set is
determined by the :prop_test:`PROCESSORS` test property or the number of
processors available to CTest, whichever is smaller.  The set of processors
chosen will be disjoint from the processors assigned to other concurrently
running tests that also have the ``PROCESSOR_AFFINITY`` property enabled.
