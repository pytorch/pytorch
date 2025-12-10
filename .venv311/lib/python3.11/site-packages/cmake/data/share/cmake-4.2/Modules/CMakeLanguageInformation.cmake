# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This file contains common code blocks used by all the language information
# files

# load any compiler-wrapper specific information
macro(__cmake_include_compiler_wrapper lang)
  set(_INCLUDED_WRAPPER_FILE 0)
  if (CMAKE_${lang}_COMPILER_ID)
    include(Platform/${CMAKE_EFFECTIVE_SYSTEM_NAME}-${CMAKE_${lang}_COMPILER_WRAPPER}-${CMAKE_${lang}_COMPILER_ID}-${lang} OPTIONAL RESULT_VARIABLE _INCLUDED_WRAPPER_FILE)
  endif()
  if (NOT _INCLUDED_WRAPPER_FILE)
    include(Platform/${CMAKE_EFFECTIVE_SYSTEM_NAME}-${CMAKE_${lang}_COMPILER_WRAPPER}-${lang} OPTIONAL RESULT_VARIABLE _INCLUDED_WRAPPER_FILE)
  endif ()

  # No platform - wrapper - lang information so maybe there's just wrapper - lang information
  if(NOT _INCLUDED_WRAPPER_FILE)
    if (CMAKE_${lang}_COMPILER_ID)
      include(Compiler/${CMAKE_${lang}_COMPILER_WRAPPER}-${CMAKE_${lang}_COMPILER_ID}-${lang} OPTIONAL RESULT_VARIABLE _INCLUDED_WRAPPER_FILE)
    endif()
    if (NOT _INCLUDED_WRAPPER_FILE)
      include(Compiler/${CMAKE_${lang}_COMPILER_WRAPPER}-${lang} OPTIONAL RESULT_VARIABLE _INCLUDED_WRAPPER_FILE)
    endif ()
  endif ()
endmacro ()
