JOB_POOL_LINK
-------------

Ninja only: Pool used for linking.

The number of parallel link processes could be limited by defining
pools with the global :prop_gbl:`JOB_POOLS`
property and then specifying here the pool name.

For instance:

.. code-block:: cmake

  set_property(TARGET myexe PROPERTY JOB_POOL_LINK two_jobs)

This property is initialized by the value of :variable:`CMAKE_JOB_POOL_LINK`.
