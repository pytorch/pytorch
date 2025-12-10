CMAKE_TRY_COMPILE_CONFIGURATION
-------------------------------

Build configuration used for :command:`try_compile` and :command:`try_run`
projects.

Projects built by :command:`try_compile` and :command:`try_run` are built
synchronously during the CMake configuration step.  Therefore a specific build
configuration must be chosen even if the generated build system
supports multiple configurations.
