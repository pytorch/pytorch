include(Compiler/IBMClang)

set(_ibmclang_version_cxx "${CMAKE_CXX_COMPILER_VERSION}")
set(CMAKE_CXX_COMPILER_VERSION "${CMAKE_CXX_COMPILER_VERSION_INTERNAL}")
include(Compiler/Clang-CXX)
set(CMAKE_CXX_COMPILER_VERSION "${_ibmclang_version_cxx}")
unset(_ibmclang_version_cxx)

__compiler_ibmclang(CXX)

__compiler_check_default_language_standard(CXX 17.1.0 17)

set(CMAKE_CXX_COMPILE_OBJECT
  "<CMAKE_CXX_COMPILER> -x c++ <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")
