ctest_submit
------------

Perform the :ref:`CTest Submit Step` as a :ref:`Dashboard Client`.

.. code-block:: cmake

  ctest_submit([PARTS <part>...] [FILES <file>...]
               [SUBMIT_URL <url>]
               [BUILD_ID <result-var>]
               [HTTPHEADER <header>]
               [RETRY_COUNT <count>]
               [RETRY_DELAY <delay>]
               [RETURN_VALUE <result-var>]
               [CAPTURE_CMAKE_ERROR <result-var>]
               [QUIET]
               )

Submit results to a dashboard server.
By default all available parts are submitted.

The options are:

``PARTS <part>...``
  Specify a subset of parts to submit.  Valid part names are:

  * ``Start`` - nothing.
  * ``Update`` - :command:`ctest_update` results, in ``Update.xml``.
  * ``Configure`` - :command:`ctest_configure` results, in ``Configure.xml``.
  * ``Build`` - :command:`ctest_build` results, in ``Build.xml``.
  * ``Test`` - :command:`ctest_test` results, in ``Test.xml``.
  * ``Coverage`` - :command:`ctest_coverage` results, in ``Coverage.xml``.
  * ``MemCheck`` - :command:`ctest_memcheck` results, in
    ``DynamicAnalysis.xml`` and ``DynamicAnalysis-Test.xml``.
  * ``Notes`` - Files listed by :variable:`CTEST_NOTES_FILES`, in ``Notes.xml``.
  * ``ExtraFiles`` - Files listed by :variable:`CTEST_EXTRA_SUBMIT_FILES`.
  * ``Upload`` - Files prepared for upload by :command:`ctest_upload`, in
    ``Upload.xml``.
  * ``Submit`` - nothing.
  * ``Done`` - Build is complete, in ``Done.xml``.

``FILES <file>...``
  Specify an explicit list of specific files to be submitted.
  Each individual file must exist at the time of the call.

``SUBMIT_URL <url>``
  .. versionadded:: 3.14

  The ``http`` or ``https`` URL of the dashboard server to send the submission
  to.  If not given, the :variable:`CTEST_SUBMIT_URL` variable is used.

``BUILD_ID <result-var>``
  .. versionadded:: 3.15

  Store in the ``<result-var>`` variable the ID assigned to this build by
  CDash.

``HTTPHEADER <HTTP-header>``
  .. versionadded:: 3.9

  Specify HTTP header to be included in the request to CDash during submission.
  For example, CDash can be configured to only accept submissions from
  authenticated clients. In this case, you should provide a bearer token in your
  header:

  .. code-block:: cmake

    ctest_submit(HTTPHEADER "Authorization: Bearer <auth-token>")

  This suboption can be repeated several times for multiple headers.

``RETRY_COUNT <count>``
  Specify how many times to retry a timed-out submission.

``RETRY_DELAY <delay>``
  Specify how long (in seconds) to wait after a timed-out submission
  before attempting to re-submit.

``RETURN_VALUE <result-var>``
  Store in the ``<result-var>`` variable ``0`` for success and
  non-zero on failure.

``CAPTURE_CMAKE_ERROR <result-var>``
  .. versionadded:: 3.13

  Store in the ``<result-var>`` variable -1 if there are any errors running
  the command and prevent ctest from returning non-zero if an error occurs.

``QUIET``
  .. versionadded:: 3.3

  Suppress all non-error messages that would have otherwise been
  printed to the console.

Submit to CDash Upload API
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.2

.. code-block:: cmake

  ctest_submit(CDASH_UPLOAD <file> [CDASH_UPLOAD_TYPE <type>]
               [SUBMIT_URL <url>]
               [BUILD_ID <result-var>]
               [HTTPHEADER <header>]
               [RETRY_COUNT <count>]
               [RETRY_DELAY <delay>]
               [RETURN_VALUE <result-var>]
               [QUIET])

This second signature is used to upload files to CDash via the CDash
file upload API. The API first sends a request to upload to CDash along
with a content hash of the file. If CDash does not already have the file,
then it is uploaded. Along with the file, a CDash type string is specified
to tell CDash which handler to use to process the data.

This signature interprets options in the same way as the first one.

.. versionadded:: 3.8
  Added the ``RETRY_COUNT``, ``RETRY_DELAY``, ``QUIET`` options.

.. versionadded:: 3.9
  Added the ``HTTPHEADER`` option.

.. versionadded:: 3.13
  Added the ``RETURN_VALUE`` option.

.. versionadded:: 3.14
  Added the ``SUBMIT_URL`` option.

.. versionadded:: 3.15
  Added the ``BUILD_ID`` option.
