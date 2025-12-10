:variable:`CMAKE_TRY_COMPILE_TARGET_TYPE`
  Internally, the :command:`try_compile` command is used to perform the
  check, and this variable controls the type of target it creates.  If this
  variable is set to ``EXECUTABLE`` (the default), the check compiles and
  links the test source code as an executable program.  If set to
  ``STATIC_LIBRARY``, the test source code is compiled but not linked.
