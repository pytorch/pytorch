CTEST_TLS_VERIFY
----------------

.. versionadded:: 3.30

Specify the CTest ``TLSVerify`` setting in a :manual:`ctest(1)`
:ref:`Dashboard Client` script or in project ``CMakeLists.txt`` code
before including the :module:`CTest` module.  The value is a boolean
indicating whether to  verify the server certificate when submitting
to a dashboard via ``https://`` URLs.

If ``CTEST_TLS_VERIFY`` is not set, the :variable:`CMAKE_TLS_VERIFY` variable
or :envvar:`CMAKE_TLS_VERIFY` environment variable is used instead.
If neither is set, the default is *on*.

.. versionchanged:: 3.31
  The default is on.  Previously, the default was off.
  Users may set the :envvar:`CMAKE_TLS_VERIFY` environment
  variable to ``0`` to restore the old default.
