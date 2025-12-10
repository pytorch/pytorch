# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

include_guard(GLOBAL)
include(Internal/CheckFlagCommonConfig)
include(Internal/CheckSourceCompiles)
include(CMakeCheckCompilerFlagCommonPatterns)

function(CMAKE_CHECK_COMPILER_FLAG _lang _flag _var)
  # Parse extra arguments
  cmake_parse_arguments(PARSE_ARGV 3 CHECK_COMPILER_FLAG "" "OUTPUT_VARIABLE" "")

  cmake_check_flag_common_init("check_compiler_flag" ${_lang} _lang_src _lang_fail_regex)

  set(CMAKE_REQUIRED_DEFINITIONS ${_flag})

  check_compiler_flag_common_patterns(_common_patterns)
  cmake_check_source_compiles(${_lang}
    "${_lang_src}"
    ${_var}
    ${_lang_fail_regex}
    ${_common_patterns}
    OUTPUT_VARIABLE _output
    )

  if (CHECK_COMPILER_FLAG_OUTPUT_VARIABLE)
    set(${CHECK_COMPILER_FLAG_OUTPUT_VARIABLE} "${_output}" PARENT_SCOPE)
  endif()

  cmake_check_flag_common_finish()
endfunction()
