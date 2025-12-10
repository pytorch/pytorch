# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# Find the nasm assembler. yasm (http://www.tortall.net/projects/yasm/) is nasm compatible

set(CMAKE_ASM_NASM_COMPILER_LIST nasm yasm)

if(NOT CMAKE_ASM_NASM_COMPILER)
  set(_CMAKE_ENV_VARX86 "ProgramFiles(x86)")
  set(_CMAKE_ASM_NASM_COMPILER_PATHS
    "[HKEY_CURRENT_USER\\SOFTWARE\\nasm]"
    "$ENV{ProgramFiles}/NASM"
    "$ENV{${_CMAKE_ENV_VARX86}}/NASM"
    "$ENV{LOCALAPPDATA}/NASM"
  )
  unset(_CMAKE_ENV_VARX86)
endif()

# Load the generic DetermineASM compiler file with the DIALECT set properly:
set(ASM_DIALECT "_NASM")
include(CMakeDetermineASMCompiler)
set(ASM_DIALECT)
unset(_CMAKE_ASM_NASM_COMPILER_PATHS)
