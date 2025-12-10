INSTALL_PARALLEL
----------------

.. versionadded:: 3.30

Enables parallel installation option for a project. The install code for each
subdirectory added with ``add_subdirectory`` can run independently.

When using the :ref:`Ninja Generators`, enabling this property causes
``install/local`` targets have the console pool disabled, allowing them to run
concurrently.

This property also provides the target ``install/parallel``, which has an
explicit dependency on the ``install/local`` target for each subdirectory.

  .. versionadded:: 3.31

  When this property is enabled, ``cmake --install`` can be given the ``-j <jobs>``
  or ``--parallel <jobs>`` option to specify a maximum number of jobs.
  The :envvar:`CMAKE_INSTALL_PARALLEL_LEVEL` environment variable specifies a
  default parallel level if this option is not provided.

Calls to :command:`install(CODE)` or :command:`install(SCRIPT)` might depend
on actions performed by an earlier :command:`install` command in a different
directory such as files installed or variable settings. If the project has
such order-dependent installation logic, parallel installation should
not be enabled, in order to prevent possible race conditions.
