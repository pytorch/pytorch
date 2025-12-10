JOB_POOL_COMPILE
----------------

.. versionadded:: 4.2

Ninja only: Pool used for compiling.

The number of parallel compile processes could be limited by defining
pools with the global :prop_gbl:`JOB_POOLS`
property and then specifying here the pool name.

This allows to override the :prop_tgt:`JOB_POOL_COMPILE`
value for specific source files within a same target.

For instance:

.. code-block:: cmake

  set_property(SOURCE main.cc PROPERTY JOB_POOL_COMPILE two_jobs)

This property is undefined by default.
