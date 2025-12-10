# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# Find the MS ARM assembler (marmasm or marmasm64)

set(ASM_DIALECT "_MARMASM")

# if we are using the 64bit cl compiler, assume we also want the 64bit assembler
if(";${CMAKE_VS_PLATFORM_NAME};${CMAKE_C_COMPILER_ARCHITECTURE_ID};${CMAKE_CXX_COMPILER_ARCHITECTURE_ID};"
    MATCHES ";(ARM64);")
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_INIT armasm64)
else()
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_INIT armasm)
endif()

include(CMakeDetermineASMCompiler)
set(ASM_DIALECT)
