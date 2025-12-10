# Find the archiver for the compiler architecture, which is always in the same
# directory as the compiler.
if(NOT DEFINED _CMAKE_PROCESSING_LANGUAGE OR _CMAKE_PROCESSING_LANGUAGE STREQUAL "")
  message(FATAL_ERROR "Internal error: _CMAKE_PROCESSING_LANGUAGE is not set")
endif()

get_filename_component(__diab_path "${CMAKE_${_CMAKE_PROCESSING_LANGUAGE}_COMPILER}" DIRECTORY)

find_program(CMAKE_${_CMAKE_PROCESSING_LANGUAGE}_COMPILER_AR
  NAMES dar
  HINTS ${__diab_path}
  NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH
  DOC "Diab Archiver"
)
mark_as_advanced(CMAKE_${_CMAKE_PROCESSING_LANGUAGE}_COMPILER_AR)
