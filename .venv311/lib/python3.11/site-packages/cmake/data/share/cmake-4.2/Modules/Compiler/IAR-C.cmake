# This file is processed when the IAR C Compiler is used
#
# C Language Specification support
#  - Newer versions of the IAR C Compiler require the --c89 flag to build a file under the C90 standard.
#  - Earlier versions of the compiler had C90 by default, not requiring the backward-compatibility flag.
#
# The IAR Language Extensions
#  - The IAR Language Extensions can be enabled by -e flag
#
include(Compiler/IAR)
include(Compiler/CMakeCommonCompilerMacros)

if(NOT CMAKE_C_COMPILER_VERSION)
  message(FATAL_ERROR "Could not detect CMAKE_C_COMPILER_VERSION. This should be automatic. Check your product license.\n")
endif()

# Unused after CMP0128
set(CMAKE_C_EXTENSION_COMPILE_OPTION -e)

if(CMAKE_C_COMPILER_VERSION_INTERNAL VERSION_GREATER 7)
  set(CMAKE_C90_STANDARD_COMPILE_OPTION --c89)
  set(CMAKE_C90_EXTENSION_COMPILE_OPTION --c89 -e)
else()
  set(CMAKE_C90_STANDARD_COMPILE_OPTION "")
  set(CMAKE_C90_EXTENSION_COMPILE_OPTION -e)
endif()

set(CMAKE_C${CMAKE_C_STANDARD_COMPUTED_DEFAULT}_STANDARD_COMPILE_OPTION "")
set(CMAKE_C${CMAKE_C_STANDARD_COMPUTED_DEFAULT}_EXTENSION_COMPILE_OPTION -e)

# Architecture specific
if("${CMAKE_C_COMPILER_ARCHITECTURE_ID}" STREQUAL "ARM")
  if (CMAKE_C_COMPILER_VERSION VERSION_LESS 5)
    # IAR C Compiler for Arm prior version 5.xx uses XLINK. Support in CMake is not implemented.
    message(FATAL_ERROR "IAR C Compiler for Arm version ${CMAKE_C_COMPILER_VERSION} not supported by CMake.")
  endif()
  __compiler_iar_ilink(C)
  __compiler_check_default_language_standard(C 5.10 90 6.10 99 8.10 11 8.40 17)

elseif("${CMAKE_C_COMPILER_ARCHITECTURE_ID}" STREQUAL "RX")
  __compiler_iar_ilink(C)
  __compiler_check_default_language_standard(C 1.10 90 2.10 99 4.10 11 4.20 17)

elseif("${CMAKE_C_COMPILER_ARCHITECTURE_ID}" STREQUAL "RH850")
  __compiler_iar_ilink(C)
  __compiler_check_default_language_standard(C 1.10 90 1.10 99 2.10 11 2.21 17)

elseif("${CMAKE_C_COMPILER_ARCHITECTURE_ID}" STREQUAL "RL78")
  if(CMAKE_C_COMPILER_VERSION VERSION_LESS 2)
    # IAR C Compiler for RL78 prior version 2.xx uses XLINK. Support in CMake is not implemented.
    message(FATAL_ERROR "IAR C Compiler for RL78 version ${CMAKE_C_COMPILER_VERSION} not supported by CMake.")
  endif()
  __compiler_iar_ilink(C)
  __compiler_check_default_language_standard(C 2.10 90 2.10 99 4.10 11 4.20 17)

elseif("${CMAKE_C_COMPILER_ARCHITECTURE_ID}" STREQUAL "RISCV")
  __compiler_iar_ilink(C)
  __compiler_check_default_language_standard(C 1.10 90 1.10 99 1.10 11 1.21 17)

elseif("${CMAKE_C_COMPILER_ARCHITECTURE_ID}" STREQUAL "AVR")
  __compiler_iar_xlink(C)
  __compiler_check_default_language_standard(C 7.10 99 8.10 17)
  set(CMAKE_C_OUTPUT_EXTENSION ".r90")

elseif("${CMAKE_C_COMPILER_ARCHITECTURE_ID}" STREQUAL "MSP430")
  __compiler_iar_xlink(C)
  __compiler_check_default_language_standard(C 1.10 90 5.10 99)
  set(CMAKE_C_OUTPUT_EXTENSION ".r43")

elseif("${CMAKE_C_COMPILER_ARCHITECTURE_ID}" STREQUAL "V850")
  __compiler_iar_xlink(C)
  __compiler_check_default_language_standard(C 1.10 90 4.10 99)
  set(CMAKE_C_OUTPUT_EXTENSION ".r85")

elseif("${CMAKE_C_COMPILER_ARCHITECTURE_ID}" STREQUAL "8051")
  __compiler_iar_xlink(C)
  __compiler_check_default_language_standard(C 6.10 90 8.10 99)
  set(CMAKE_C_OUTPUT_EXTENSION ".r51")

elseif("${CMAKE_C_COMPILER_ARCHITECTURE_ID}" STREQUAL "STM8")
  __compiler_iar_ilink(C)
  __compiler_check_default_language_standard(C 3.11 90 3.11 99)

else()
  message(FATAL_ERROR "CMAKE_C_COMPILER_ARCHITECTURE_ID not detected. This should be automatic.")
endif()
