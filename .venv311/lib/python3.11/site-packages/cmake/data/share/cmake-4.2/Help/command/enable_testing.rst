enable_testing
--------------

Enable testing for current directory and below.

.. code-block:: cmake

  enable_testing()

Enables testing for this directory and below.

This command should be in the top-level source directory because
:manual:`ctest(1)` expects to find a test file in the top-level
build directory.

This command is automatically invoked when the :module:`CTest`
module is included, except if the :variable:`BUILD_TESTING`
option is turned off.

See also the :command:`add_test` command.
