# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

include_guard(GLOBAL)
include(Internal/CheckFlagCommonConfig)
include(Internal/CheckSourceCompiles)
include(CMakeCheckCompilerFlagCommonPatterns)

function(CMAKE_CHECK_LINKER_FLAG _lang _flag _var)
  # link step supports less languages than the compiler
  # so do a first check about the requested language
  if (_lang STREQUAL "ISPC")
    message (SEND_ERROR "check_linker_flag: ${_lang}: unsupported language.")
    return()
  endif()

  # Parse extra arguments
  cmake_parse_arguments(PARSE_ARGV 3 CHECK_LINKER_FLAG "" "OUTPUT_VARIABLE" "")

  cmake_check_flag_common_init("check_linker_flag" ${_lang} _lang_src _lang_fail_regex)

  set(CMAKE_REQUIRED_LINK_OPTIONS "${_flag}")

  check_compiler_flag_common_patterns(_common_patterns)

  # Match linker warnings if the exact flag is ignored.
  foreach(flag IN LISTS _flag)
    string(REGEX REPLACE "([][+.*?()^$])" [[\\\1]] _flag_regex "${flag}")
    list(APPEND _common_patterns
      FAIL_REGEX "warning: [^\n]*${_flag_regex}[^\n]* ignored"
      )
  endforeach()

  cmake_check_source_compiles(${_lang}
    "${_lang_src}"
    ${_var}
    ${_lang_fail_regex}
    ${_common_patterns}
    OUTPUT_VARIABLE _output
    )

  if (CHECK_LINKER_FLAG_OUTPUT_VARIABLE)
    set(${CHECK_LINKER_FLAG_OUTPUT_VARIABLE} "${_output}" PARENT_SCOPE)
  endif()

  cmake_check_flag_common_finish()
endfunction()
