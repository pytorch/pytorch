# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# Find the MS assembler (masm or masm64)

set(ASM_DIALECT "_MASM")

# if we are using the 64bit cl compiler, assume we also want the 64bit assembler
if(";${CMAKE_VS_PLATFORM_NAME};${MSVC_C_ARCHITECTURE_ID};${MSVC_CXX_ARCHITECTURE_ID};"
  MATCHES ";(Win64|Itanium|x64|IA64);")
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_INIT ml64)
else()
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_INIT ml)
endif()

include(CMakeDetermineASMCompiler)
set(ASM_DIALECT)
