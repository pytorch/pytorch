# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

function(cmake_cuda_architectures_validate lang)
  if(DEFINED CMAKE_${lang}_ARCHITECTURES)
    if(CMAKE_${lang}_ARCHITECTURES STREQUAL "")
      message(FATAL_ERROR "CMAKE_${lang}_ARCHITECTURES must be non-empty if set.")
    elseif(CMAKE_${lang}_ARCHITECTURES MATCHES [["]])
      message(FATAL_ERROR
        "CMAKE_${lang}_ARCHITECTURES contains literal quotes:\n"
        "  ${CMAKE_${lang}_ARCHITECTURES}\n"
      )
    elseif(CMAKE_${lang}_ARCHITECTURES AND NOT CMAKE_${lang}_ARCHITECTURES MATCHES "^([0-9]+(a|f)?(-real|-virtual)?(;[0-9]+(a|f)?(-real|-virtual)?|;)*|all|all-major|native)$")
      message(FATAL_ERROR
        "CMAKE_${lang}_ARCHITECTURES:\n"
        "  ${CMAKE_${lang}_ARCHITECTURES}\n"
        "is not one of the following:\n"
        "  * a semicolon-separated list of integers, each optionally\n"
        "    followed by '-real' or '-virtual'\n"
        "  * a special value: all, all-major, native\n"
        )
    endif()
  endif()
endfunction()
