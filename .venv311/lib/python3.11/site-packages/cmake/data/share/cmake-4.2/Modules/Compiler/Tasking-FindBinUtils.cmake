# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# Find the archiver for the compiler architecture, which is always in the same
# directory as the compiler.
if(NOT DEFINED _CMAKE_PROCESSING_LANGUAGE OR _CMAKE_PROCESSING_LANGUAGE STREQUAL "")
  message(FATAL_ERROR "Internal error: _CMAKE_PROCESSING_LANGUAGE is not set")
endif()

get_filename_component(__tasking_hints "${CMAKE_${_CMAKE_PROCESSING_LANGUAGE}_COMPILER}" DIRECTORY)

find_program(CMAKE_${_CMAKE_PROCESSING_LANGUAGE}_COMPILER_AR
  NAMES artc ararm armcs ar51 ararc arpcp
  HINTS ${__tasking_hints}
  NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH
  DOC "Tasking Archiver"
)
mark_as_advanced(CMAKE_${_CMAKE_PROCESSING_LANGUAGE}_COMPILER_AR)
