CMAKE_SKIP_INSTALL_ALL_DEPENDENCY
---------------------------------

Don't make the ``install`` target depend on the ``all`` target.

By default, the ``install`` target depends on the ``all`` target.  This
has the effect, that when ``make install`` is invoked or ``INSTALL`` is
built, first the ``all`` target is built, then the installation starts.
If ``CMAKE_SKIP_INSTALL_ALL_DEPENDENCY`` is set to ``TRUE``, this
dependency is not created, so the installation process will start immediately,
independent from whether the project has been completely built or not.

See also :variable:`CMAKE_SKIP_TEST_ALL_DEPENDENCY`.
