include(Compiler/GNU)
__compiler_gnu(OBJC)
__compiler_gnu_c_standards(OBJC)


if((NOT DEFINED CMAKE_DEPENDS_USE_COMPILER OR CMAKE_DEPENDS_USE_COMPILER)
    AND CMAKE_GENERATOR MATCHES "Makefiles|WMake"
    AND CMAKE_DEPFILE_FLAGS_OBJC)
  # dependencies are computed by the compiler itself
  set(CMAKE_OBJC_DEPFILE_FORMAT gcc)
  set(CMAKE_OBJC_DEPENDS_USE_COMPILER TRUE)
endif()
