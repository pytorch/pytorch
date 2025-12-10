CMAKE_SKIP_TEST_ALL_DEPENDENCY
------------------------------

.. versionadded:: 3.29

Control whether the ``test`` target depends on the ``all`` target.

If this variable is not defined, or is set to ``TRUE``, then the
``test`` (or ``RUN_TESTS``) target does not depend on the
``all`` (or ``ALL_BUILD``) target.  When the ``test`` target is built,
e.g., via ``make test``, the test process will start immediately,
regardless of whether the project has been completely built or not.

If ``CMAKE_SKIP_TEST_ALL_DEPENDENCY`` is explicitly set to ``FALSE``,
then the ``test`` target will depend on the ``all`` target.  When the
``test`` target is built, e.g., via ``make test``, the ``all`` target
will be built first, and then the tests will run.

See also :variable:`CMAKE_SKIP_INSTALL_ALL_DEPENDENCY`.
