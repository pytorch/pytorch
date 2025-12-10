ctest_update
------------

Perform the :ref:`CTest Update Step` as a :ref:`Dashboard Client`.

.. code-block:: cmake

  ctest_update([SOURCE <source-dir>]
               [RETURN_VALUE <result-var>]
               [CAPTURE_CMAKE_ERROR <result-var>]
               [QUIET])

Update the source tree from version control and record results in
``Update.xml`` for submission with the :command:`ctest_submit` command.

The options are:

``SOURCE <source-dir>``
  Specify the source directory.  If not given, the
  :variable:`CTEST_SOURCE_DIRECTORY` variable is used.

``RETURN_VALUE <result-var>``
  Store in the ``<result-var>`` variable the number of files
  updated or ``-1`` on error.

``CAPTURE_CMAKE_ERROR <result-var>``
  .. versionadded:: 3.13

  Store in the ``<result-var>`` variable -1 if there are any errors running
  the command and prevent ctest from returning non-zero if an error occurs.

``QUIET``
  .. versionadded:: 3.3

  Tell CTest to suppress most non-error messages that it would
  have otherwise printed to the console.  CTest will still report
  the new revision of the repository and any conflicting files
  that were found.

The update always follows the version control branch currently checked
out in the source directory.  See the :ref:`CTest Update Step`
documentation for information about variables that change the behavior
of ``ctest_update()``.
