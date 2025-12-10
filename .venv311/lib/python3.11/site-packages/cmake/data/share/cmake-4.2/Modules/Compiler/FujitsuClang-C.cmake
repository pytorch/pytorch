include(Compiler/FujitsuClang)

set(_fjclang_ver "${CMAKE_C_COMPILER_VERSION_INTERNAL}")
set(CMAKE_C_COMPILER_VERSION "${CMAKE_C_COMPILER_VERSION_INTERNAL}")
include(Compiler/Clang-C)
set(CMAKE_C_COMPILER_VERSION "${_fjclang_ver}")
