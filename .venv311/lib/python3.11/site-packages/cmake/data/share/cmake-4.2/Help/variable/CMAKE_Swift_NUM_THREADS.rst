CMAKE_Swift_NUM_THREADS
-----------------------

.. versionadded:: 3.15.1

Number of threads for parallel compilation for Swift targets.

This variable controls the number of parallel jobs that the swift driver creates
for building targets.  If not specified, it will default to the number of
logical CPUs on the host.
