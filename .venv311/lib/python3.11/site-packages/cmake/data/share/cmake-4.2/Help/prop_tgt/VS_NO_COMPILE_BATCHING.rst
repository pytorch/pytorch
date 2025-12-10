VS_NO_COMPILE_BATCHING
----------------------

.. versionadded:: 3.24

Turn off compile batching for the target. Usually MSBuild calls the compiler
with multiple c/cpp files and compiler starts subprocesses for each file to
make the build parallel. If you want compiler to be invoked with one file at
a time set ``VS_NO_COMPILE_BATCHING`` to ON. If this flag is set MSBuild will
call compiler with one c/cpp file at a time. Useful when you want to use tool
that replaces the compiler, for example some build caching tool.

This property is initialized by the :variable:`CMAKE_VS_NO_COMPILE_BATCHING`
variable if it is set when a target is created.

Example
^^^^^^^

This shows setting the property for the target ``foo``.

.. code-block:: cmake

  add_library(foo SHARED foo.cpp)
  set_property(TARGET foo PROPERTY VS_NO_COMPILE_BATCHING ON)
