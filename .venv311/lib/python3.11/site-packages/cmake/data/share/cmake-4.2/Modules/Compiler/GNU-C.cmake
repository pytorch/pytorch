include(Compiler/GNU)
__compiler_gnu(C)
__compiler_gnu_c_standards(C)


if((NOT DEFINED CMAKE_DEPENDS_USE_COMPILER OR CMAKE_DEPENDS_USE_COMPILER)
    AND CMAKE_GENERATOR MATCHES "Makefiles|WMake"
    AND CMAKE_DEPFILE_FLAGS_C)
  # dependencies are computed by the compiler itself
  set(CMAKE_C_DEPFILE_FORMAT gcc)
  set(CMAKE_C_DEPENDS_USE_COMPILER TRUE)
endif()

set(CMAKE_C_COMPILE_OPTIONS_EXPLICIT_LANGUAGE -x c)

__compiler_check_default_language_standard(C 3.4 90 5.0 11 8.1 17)
