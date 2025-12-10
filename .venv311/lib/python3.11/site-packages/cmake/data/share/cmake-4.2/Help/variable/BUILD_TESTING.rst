BUILD_TESTING
-------------

Control whether the :module:`CTest` module invokes :command:`enable_testing`.

The :module:`CTest` module, when loaded by ``include(CTest)``,
runs code of the form:

.. code-block:: cmake

  option(BUILD_TESTING "..." ON)
  if (BUILD_TESTING)
     # ...
     enable_testing()
     # ...
  endif()

This creates a ``BUILD_TESTING`` option that controls whether the
:command:`enable_testing` command is invoked to enable generation
of tests to run using :manual:`ctest(1)`.  See the :command:`add_test`
command to create tests.

.. note::

  Call ``include(CTest)`` in the top-level source directory since
  :manual:`ctest(1)` expects to find a test file in the top-level
  build directory.
