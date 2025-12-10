# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=[

NOTE: This function is used internally by CMake. Projects should not include
      this file directly.

The cmake_try_compiler_or_linker_flag() function can be used to compile and link a
source file to check whether a specific compiler or linker flag is supported.
The function does not use the try_compile() command so as to avoid infinite
recursion.  It may not work for all platforms or toolchains, the caller is
responsible for ensuring it is only called in valid situations.

  cmake_try_compiler_or_linker_flag(<lang> <flag> <result>
                            [SRC_EXT <ext>] [COMMAND_PATTERN <pattern>]
                            [FAIL_REGEX <regex> ...]
                            [OUTPUT_VARIABLE <output>])

Parameters:
  <lang>   - Language to check.
  <flag>   - The flag to add to the compile/link command line.
  <result> - Boolean output variable.  It will be stored in the cache as an
             internal variable and if true, will cause future tests that assign
             to that variable to be bypassed.

Optional parameters:
  SRC_EXT         - Overrides the extension of the source file used for the
                    check.  Defaults are 'c' (C), 'cxx' (CXX), 'F' (Fortran).
  COMMAND_PATTERN - Pattern to be used for the command line. The default is
                    '<FLAG> -o <OUTPUT> <SOURCE>'
  FAIL_REGEX      - List of additional regular expressions that, if matched by
                    the output, give a failed result for the check.  A common
                    set of regular expressions will be included in addition to
                    those given by FAIL_REGEX.
  OUTPUT_VARIABLE - Set <output> variable with details about any error.
#]=]

include_guard(GLOBAL)
include(CMakeCheckCompilerFlagCommonPatterns)

function(CMAKE_TRY_COMPILER_OR_LINKER_FLAG lang flag result)
  # Cache results between runs similar to check_<lang>_source_compiles()
  if(DEFINED ${result})
    return()
  endif()

  set(comment "Is the '${flag}' option(s) supported")
  string(REPLACE ";" " " comment "${comment}")

  if (NOT lang MATCHES "^(C|CXX|Fortran|ASM)$")
    # other possible languages are not supported
    # log message to keep trace of this problem...
    message(CONFIGURE_LOG
      "Function 'CMAKE_CHECK_COMPILER_FLAG' called with unsupported language: ${lang}\n")
    set(${result} FALSE CACHE INTERNAL ${comment})
    return()
  endif()
  if (lang STREQUAL "ASM")
    # assume ASM compiler is a multi-language compiler, so supports C language as well
    set(check_lang C)
  else()
    set(check_lang ${lang})
  endif()

  cmake_parse_arguments(CCCF "" "SRC_EXT;COMMAND_PATTERN;OUTPUT_VARIABLE" "FAIL_REGEX" ${ARGN})

  if (NOT CCCF_COMMAND_PATTERN)
    set (CCCF_COMMAND_PATTERN "<FLAG> -o <OUTPUT> <SOURCE>")
  endif()

  list (APPEND CCCF_FAIL_REGEX "argument unused during compilation") # clang
  if (check_lang STREQUAL "C")
    list(APPEND CCCF_FAIL_REGEX
      "command line option .* is valid for .* but not for C") # GNU
  elseif(check_lang STREQUAL "CXX")
    list(APPEND CCCF_FAIL_REGEX
      "command line option .* is valid for .* but not for C\\+\\+") # GNU
  elseif(check_lang STREQUAL "Fortran")
    list(APPEND CCCF_FAIL_REGEX
      "command line option .* is valid for .* but not for Fortran") # GNU
  endif()

  # Add patterns for common errors
  check_compiler_flag_common_patterns(COMPILER_FLAG_COMMON_PATTERNS)
  foreach(arg IN LISTS COMPILER_FLAG_COMMON_PATTERNS)
    if(arg MATCHES "^FAIL_REGEX$")
      continue()
    endif()
    list(APPEND CCCF_FAIL_REGEX "${arg}")
  endforeach()

  if(NOT CCCF_SRC_EXT)
    if (check_lang STREQUAL "C")
      set(CCCF_SRC_EXT c)
    elseif(check_lang STREQUAL "CXX")
      set(CCCF_SRC_EXT cxx)
    elseif(check_lang STREQUAL "Fortran")
      set(CCCF_SRC_EXT F)
    endif()
  endif()

  if (CCCF_OUTPUT_VARIABLE)
    unset(${CCCF_OUTPUT_VARIABLE} PARENT_SCOPE)
  endif()

  # Compute the directory in which to run the test.
  set(COMPILER_FLAG_DIR "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp")
  # Compute source and output files.
  set(COMPILER_FLAG_SRC
    "${COMPILER_FLAG_DIR}/CompilerFlag${lang}.${CCCF_SRC_EXT}")
  if(check_lang STREQUAL "Fortran")
    file(WRITE "${COMPILER_FLAG_SRC}"
      "      program simple\n      end program simple\n")
  else()
    file(WRITE "${COMPILER_FLAG_SRC}" "int main (void)\n{ return 0; }\n")
  endif()
  get_filename_component(COMPILER_FLAG_EXE "${COMPILER_FLAG_SRC}" NAME_WE)
  string(APPEND COMPILER_FLAG_EXE "${CMAKE_EXECUTABLE_SUFFIX}")

  # Build command line
  separate_arguments(CCCF_COMMAND_PATTERN UNIX_COMMAND
    "${CCCF_COMMAND_PATTERN}")
  list(TRANSFORM CCCF_COMMAND_PATTERN REPLACE "<SOURCE>" "${COMPILER_FLAG_SRC}")
  list(TRANSFORM CCCF_COMMAND_PATTERN REPLACE "<OUTPUT>" "${COMPILER_FLAG_EXE}")
  list(TRANSFORM CCCF_COMMAND_PATTERN REPLACE "<FLAG>" "${flag}")

  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env LC_ALL=C LC_MESSAGES=C LANG=C
            "${CMAKE_${lang}_COMPILER}" ${CCCF_COMMAND_PATTERN}
    WORKING_DIRECTORY "${COMPILER_FLAG_DIR}"
    OUTPUT_VARIABLE COMPILER_FLAG_OUTPUT
    ERROR_VARIABLE COMPILER_FLAG_OUTPUT
    RESULT_VARIABLE COMPILER_FLAG_RESULT)

  # Record result in the cache so we can avoid re-testing every CMake run
  if (COMPILER_FLAG_RESULT)
    set(${result} FALSE CACHE INTERNAL ${comment})
  else()
    foreach(regex IN LISTS CCCF_FAIL_REGEX)
      if(COMPILER_FLAG_OUTPUT MATCHES "${regex}")
        set(${result} FALSE CACHE INTERNAL ${comment})
      endif()
    endforeach()
  endif()
  if (DEFINED ${result})
    message(CONFIGURE_LOG
        "Determining if the ${flag} option "
        "is supported for ${lang} language failed with the following output:\n"
        "${COMPILER_FLAG_OUTPUT}\n")
    if (CCCF_OUTPUT_VARIABLE)
      set(${CCCF_OUTPUT_VARIABLE} "${COMPILER_FLAG_OUTPUT}" PARENT_SCOPE)
    endif()
    return()
  endif()

  set(${result} TRUE CACHE INTERNAL ${comment})
endfunction()
