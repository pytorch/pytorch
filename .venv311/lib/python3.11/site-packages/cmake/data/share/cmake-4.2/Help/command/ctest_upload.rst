ctest_upload
------------

Upload files to a dashboard server as a :ref:`Dashboard Client`.

.. code-block:: cmake

  ctest_upload(FILES <file>... [QUIET] [CAPTURE_CMAKE_ERROR <result-var>])

The options are:

``FILES <file>...``
  Specify a list of files to be sent along with the build results to the
  dashboard server.

``QUIET``
  .. versionadded:: 3.3

  Suppress any CTest-specific non-error output that would have been
  printed to the console otherwise.

``CAPTURE_CMAKE_ERROR <result-var>``
  .. versionadded:: 3.7

  Store in the ``<result-var>`` variable -1 if there are any errors running
  the command and prevent ctest from returning non-zero if an error occurs.
