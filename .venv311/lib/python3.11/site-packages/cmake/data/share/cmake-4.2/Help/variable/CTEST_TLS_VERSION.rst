CTEST_TLS_VERSION
-----------------

.. versionadded:: 3.30

Specify the CTest ``TLSVersion`` setting in a :manual:`ctest(1)`
:ref:`Dashboard Client` script or in project ``CMakeLists.txt`` code
before including the :module:`CTest` module.  The value is a minimum
TLS version allowed when submitting to a dashboard via ``https://`` URLs.

The value may be one of:

.. include:: include/CMAKE_TLS_VERSION-VALUES.rst

If ``CTEST_TLS_VERSION`` is not set, the :variable:`CMAKE_TLS_VERSION` variable
or :envvar:`CMAKE_TLS_VERSION` environment variable is used instead.
