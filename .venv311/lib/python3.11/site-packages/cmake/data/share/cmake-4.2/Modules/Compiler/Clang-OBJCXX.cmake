include(Compiler/Clang)
__compiler_clang(OBJCXX)
__compiler_clang_cxx_standards(OBJCXX)

if((NOT DEFINED CMAKE_DEPENDS_USE_COMPILER OR CMAKE_DEPENDS_USE_COMPILER)
    AND CMAKE_GENERATOR MATCHES "Makefiles|WMake"
    AND CMAKE_DEPFILE_FLAGS_OBJCXX)
  # dependencies are computed by the compiler itself
  set(CMAKE_OBJCXX_DEPFILE_FORMAT gcc)
  set(CMAKE_OBJCXX_DEPENDS_USE_COMPILER TRUE)
endif()
