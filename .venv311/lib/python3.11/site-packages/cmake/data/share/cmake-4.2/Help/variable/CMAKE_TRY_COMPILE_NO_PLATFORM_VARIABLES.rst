CMAKE_TRY_COMPILE_NO_PLATFORM_VARIABLES
---------------------------------------

.. versionadded:: 3.24

Set to a true value to tell the :command:`try_compile` command not
to propagate any platform variables into the test project.

The :command:`try_compile` command normally passes some CMake variables
that configure the platform and toolchain behavior into test projects.
See policy :policy:`CMP0137`.  This variable may be set to disable
that behavior.
