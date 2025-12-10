# This file is processed when the IAR Assembler is used

include(Compiler/IAR)

# Architecture specific
if("${CMAKE_ASM${ASM_DIALECT}_COMPILER_ARCHITECTURE_ID}" STREQUAL "ARM")
  __compiler_iar_ilink(ASM)
  __assembler_iar_deps("-y" 9.30)
  set(_CMAKE_IAR_SILENCER_FLAG " -S")
  set(CMAKE_ASM_SOURCE_FILE_EXTENSIONS s;asm;msa;S)

elseif("${CMAKE_ASM${ASM_DIALECT}_COMPILER_ARCHITECTURE_ID}" STREQUAL "RX")
  __compiler_iar_ilink(ASM)
  __assembler_iar_deps("--dependencies=ns" 2.50.1)
  set(_CMAKE_IAR_SILENCER_FLAG " --silent")
  set(CMAKE_ASM_SOURCE_FILE_EXTENSIONS s;asm;msa;S)

elseif("${CMAKE_ASM${ASM_DIALECT}_COMPILER_ARCHITECTURE_ID}" STREQUAL "RH850")
  __compiler_iar_ilink(ASM)
  __assembler_iar_deps("--dependencies=ns" 2)
  set(_CMAKE_IAR_SILENCER_FLAG " --silent")
  set(CMAKE_ASM_SOURCE_FILE_EXTENSIONS s;asm;msa;S)

elseif("${CMAKE_ASM${ASM_DIALECT}_COMPILER_ARCHITECTURE_ID}" STREQUAL "RL78")
  __compiler_iar_ilink(ASM)
  __assembler_iar_deps("--dependencies=ns" 2)
  set(_CMAKE_IAR_SILENCER_FLAG " --silent")
  set(CMAKE_ASM_SOURCE_FILE_EXTENSIONS s;asm;msa;S)

elseif("${CMAKE_ASM${ASM_DIALECT}_COMPILER_ARCHITECTURE_ID}" MATCHES "(RISCV|RISC-V)")
  __compiler_iar_ilink(ASM)
  __assembler_iar_deps("--dependencies=ns" 1)
  set(_CMAKE_IAR_SILENCER_FLAG " --silent")
  set(CMAKE_ASM_SOURCE_FILE_EXTENSIONS s;asm;msa;S)

elseif("${CMAKE_ASM${ASM_DIALECT}_COMPILER_ARCHITECTURE_ID}" STREQUAL "AVR")
  __compiler_iar_xlink(ASM)
  __assembler_iar_deps("-y" 8)
  set(_CMAKE_IAR_SILENCER_FLAG " -S")
  set(CMAKE_ASM_SOURCE_FILE_EXTENSIONS s90;asm;msa)
  set(CMAKE_ASM_OUTPUT_EXTENSION ".r90")

elseif("${CMAKE_ASM${ASM_DIALECT}_COMPILER_ARCHITECTURE_ID}" STREQUAL "MSP430")
  __compiler_iar_xlink(ASM)
  __assembler_iar_deps("-y" 8)
  set(_CMAKE_IAR_SILENCER_FLAG " -S")
  set(CMAKE_ASM_SOURCE_FILE_EXTENSIONS s43;asm;msa)
  set(CMAKE_ASM_OUTPUT_EXTENSION ".r43")

elseif("${CMAKE_ASM${ASM_DIALECT}_COMPILER_ARCHITECTURE_ID}" STREQUAL "V850")
  __compiler_iar_xlink(ASM)
  set(_CMAKE_IAR_SILENCER_FLAG " -S")
  set(CMAKE_ASM_SOURCE_FILE_EXTENSIONS s85;asm;msa)
  set(CMAKE_ASM_OUTPUT_EXTENSION ".r85")

elseif("${CMAKE_ASM${ASM_DIALECT}_COMPILER_ARCHITECTURE_ID}" STREQUAL "8051")
  __compiler_iar_xlink(ASM)
  set(_CMAKE_IAR_SILENCER_FLAG " -S")
  set(CMAKE_ASM_SOURCE_FILE_EXTENSIONS s51;asm;msa)
  set(CMAKE_ASM_OUTPUT_EXTENSION ".r51")

elseif("${CMAKE_ASM${ASM_DIALECT}_COMPILER_ARCHITECTURE_ID}" STREQUAL "STM8")
  __compiler_iar_ilink(ASM)
  __assembler_iar_deps("--dependencies=ns" 2)
  set(_CMAKE_IAR_SILENCER_FLAG " --silent")
  set(CMAKE_ASM_SOURCE_FILE_EXTENSIONS s;asm;msa;S)

else()
  message(FATAL_ERROR "CMAKE_ASM${ASM_DIALECT}_COMPILER_ARCHITECTURE_ID not detected. This should be automatic." )
endif()

string(APPEND CMAKE_ASM_FLAGS_DEBUG_INIT " -r")
string(APPEND CMAKE_ASM_FLAGS_MINSIZEREL_INIT " -DNDEBUG")
string(APPEND CMAKE_ASM_FLAGS_RELEASE_INIT " -DNDEBUG")
string(APPEND CMAKE_ASM_FLAGS_RELWITHDEBINFO_INIT " -r -DNDEBUG")

set(CMAKE_ASM_COMPILE_OBJECT "<CMAKE_ASM_COMPILER> ${_CMAKE_IAR_SILENCER_FLAG} <SOURCE> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT>")

unset(_CMAKE_IAR_SILENCER_FLAG)
