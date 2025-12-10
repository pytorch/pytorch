# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This file is used by EnableLanguage in cmGlobalGenerator to
# determine that the selected ASM compiler works.
# For assembler this can only check whether the compiler has been found,
# because otherwise there would have to be a separate assembler source file
# for each assembler on every architecture.


set(_ASM_COMPILER_WORKS 0)

if(CMAKE_ASM${ASM_DIALECT}_COMPILER)
  set(_ASM_COMPILER_WORKS 1)
endif()

# when using generic "ASM" support, we must have detected the compiler ID, fail otherwise:
if("ASM${ASM_DIALECT}" STREQUAL "ASM")
  if(NOT CMAKE_ASM${ASM_DIALECT}_COMPILER_ID)
    set(_ASM_COMPILER_WORKS 0)
  endif()
endif()

set(CMAKE_ASM${ASM_DIALECT}_COMPILER_WORKS ${_ASM_COMPILER_WORKS} CACHE INTERNAL "")
