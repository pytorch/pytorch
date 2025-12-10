include(Compiler/FujitsuClang)

set(_fjclang_ver "${CMAKE_CXX_COMPILER_VERSION_INTERNAL}")
set(CMAKE_CXX_COMPILER_VERSION "${CMAKE_CXX_COMPILER_VERSION_INTERNAL}")
include(Compiler/Clang-CXX)
set(CMAKE_CXX_COMPILER_VERSION "${_fjclang_ver}")
