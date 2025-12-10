JOB_POOLS
---------

Ninja only: List of available pools.

A pool is a named integer property and defines the maximum number
of concurrent jobs which can be started by a rule assigned to the pool.
The ``JOB_POOLS`` property is a semicolon-separated list of
pairs using the syntax ``NAME=integer`` (without a space after the equality sign).

For instance:

.. code-block:: cmake

  set_property(GLOBAL PROPERTY JOB_POOLS two_jobs=2 ten_jobs=10)

Defined pools can be used at different levels:

* Globally, by setting :variable:`CMAKE_JOB_POOL_COMPILE` and
  :variable:`CMAKE_JOB_POOL_LINK`.
* Per target, by setting the target properties :prop_tgt:`JOB_POOL_COMPILE`
  and :prop_tgt:`JOB_POOL_LINK`.
* :command:`Custom commands <add_custom_command>` and
  :command:`custom targets <add_custom_target>` can specify pools using the
  option ``JOB_POOL``.
* Per-source, by setting the source file property :prop_sf:`JOB_POOL_COMPILE`,
  in case some specific source files require to override their global or
  target assigned pool.

Using a pool that is not defined by ``JOB_POOLS`` causes an error by ninja
at build time.

If not set, this property uses the value of the :variable:`CMAKE_JOB_POOLS`
variable.

Build targets provided by CMake that are meant for individual interactive
use, such as ``install``, are placed in the ``console`` pool automatically.
