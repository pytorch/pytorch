# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This file is included in CMakeSystemSpecificInformation.cmake if
# the CodeBlocks extra generator has been selected.

find_program(CMAKE_CODEBLOCKS_EXECUTABLE NAMES codeblocks DOC "The CodeBlocks executable")

if(CMAKE_CODEBLOCKS_EXECUTABLE)
  set(CMAKE_OPEN_PROJECT_COMMAND "${CMAKE_CODEBLOCKS_EXECUTABLE} <PROJECT_FILE>" )
endif()

# Determine builtin macros and include dirs:
include(${CMAKE_CURRENT_LIST_DIR}/CMakeExtraGeneratorDetermineCompilerMacrosAndIncludeDirs.cmake)

# Try to find out how many CPUs we have and set the -j argument for make accordingly
set(_CMAKE_CODEBLOCKS_INITIAL_MAKE_ARGS "")

include(ProcessorCount)
ProcessorCount(_CMAKE_CODEBLOCKS_PROCESSOR_COUNT)

# Only set -j if we are under UNIX and if the make-tool used actually has "make" in the name
# (we may also get here in the future e.g. for ninja)
if("${_CMAKE_CODEBLOCKS_PROCESSOR_COUNT}" GREATER 1  AND  CMAKE_HOST_UNIX  AND  "${CMAKE_MAKE_PROGRAM}" MATCHES make)
  set(_CMAKE_CODEBLOCKS_INITIAL_MAKE_ARGS "-j${_CMAKE_CODEBLOCKS_PROCESSOR_COUNT}")
endif()

# This variable is used by the CodeBlocks generator and appended to the make invocation commands.
set(CMAKE_CODEBLOCKS_MAKE_ARGUMENTS "${_CMAKE_CODEBLOCKS_INITIAL_MAKE_ARGS}" CACHE STRING "Additional command line arguments when CodeBlocks invokes make. Enter e.g. -j<some_number> to get parallel builds")

# This variable is used by the CodeBlocks generator and allows the user to overwrite the autodetected CodeBlocks compiler id
set(CMAKE_CODEBLOCKS_COMPILER_ID "" CACHE STRING "Id string of the compiler for the CodeBlocks IDE. Automatically detected when left empty")
