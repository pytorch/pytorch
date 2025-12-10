include(Compiler/IBMClang)

set(_ibmclang_version_asm "${CMAKE_ASM_COMPILER_VERSION}")
set(CMAKE_ASM_COMPILER_VERSION "${CMAKE_ASM_COMPILER_VERSION_INTERNAL}")
include(Compiler/Clang-ASM)
set(CMAKE_ASM_COMPILER_VERSION "${_ibmclang_version_asm}")
unset(_ibmclang_version_asm)

set(CMAKE_ASM_SOURCE_FILE_EXTENSIONS s;S;asm)

__compiler_ibmclang(ASM)
