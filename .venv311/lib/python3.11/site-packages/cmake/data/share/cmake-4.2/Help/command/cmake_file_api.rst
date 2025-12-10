cmake_file_api
--------------

.. versionadded:: 3.27

Enables interacting with the :manual:`CMake file API <cmake-file-api(7)>`.

.. signature::
  cmake_file_api(QUERY ...)

  The ``QUERY`` subcommand adds a file API query for the current CMake
  invocation.

  .. code-block:: cmake

    cmake_file_api(
      QUERY
      API_VERSION <version>
      [CODEMODEL <versions>...]
      [CACHE <versions>...]
      [CMAKEFILES <versions>...]
      [TOOLCHAINS <versions>...]
    )

  The ``API_VERSION`` must always be given.  Currently, the only supported
  value for ``<version>`` is 1.  See :ref:`file-api v1` for details of the
  reply content and location.

  Each of the optional keywords ``CODEMODEL``, ``CACHE``, ``CMAKEFILES`` and
  ``TOOLCHAINS`` correspond to one of the object kinds that can be requested
  by the project.  The ``configureLog`` object kind cannot be set with this
  command, since it must be set before CMake starts reading the top level
  ``CMakeLists.txt`` file.

  For each of the optional keywords, the ``<versions>`` list must contain one
  or more version values of the form ``major`` or ``major.minor``, where
  ``major`` and ``minor`` are integers.  Projects should list the versions they
  accept in their preferred order, as only the first supported value from the
  list will be selected.  The command will ignore versions with a ``major``
  version higher than any major version it supports for that object kind.
  It will raise an error if it encounters an invalid version number, or if none
  of the requested versions is supported.

  For each type of object kind requested, a query equivalent to a shared,
  stateless query will be added internally.  No query file will be created in
  the file system.  The reply *will* be written to the file system at
  generation time.

  It is not an error to add a query for the same thing more than once, whether
  from query files or from multiple calls to ``cmake_file_api(QUERY)``.
  The final set of queries will be a merged combination of all queries
  specified on disk and queries submitted by the project.

Example
^^^^^^^

A project may want to use replies from the file API at build time to implement
some form of verification task.  Instead of relying on something outside of
CMake to create a query file, the project can use ``cmake_file_api(QUERY)``
to request the required information for the current run.  It can then create
a custom command to run at build time, knowing that the requested information
should always be available.

.. code-block:: cmake

  cmake_file_api(
    QUERY
    API_VERSION 1
    CODEMODEL 2.3
    TOOLCHAINS 1
  )

  add_custom_target(verify_project
    COMMAND ${CMAKE_COMMAND}
      -D BUILD_DIR=${CMAKE_BINARY_DIR}
      -D CONFIG=$<CONFIG>
      -P ${CMAKE_CURRENT_SOURCE_DIR}/verify_project.cmake
  )
