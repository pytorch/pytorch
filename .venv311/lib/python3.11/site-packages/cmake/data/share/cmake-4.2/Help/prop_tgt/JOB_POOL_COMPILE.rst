JOB_POOL_COMPILE
----------------

Ninja only: Pool used for compiling.

The number of parallel compile processes could be limited by defining
pools with the global :prop_gbl:`JOB_POOLS`
property and then specifying here the pool name.

For instance:

.. code-block:: cmake

  set_property(TARGET myexe PROPERTY JOB_POOL_COMPILE ten_jobs)

This property is initialized by the value of
:variable:`CMAKE_JOB_POOL_COMPILE`.
