# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

if(CMAKE_Swift_COMPILER_FORCED)
  # The compiler configuration was forced by the user.
  # Assume the user has configured all compiler information.
  set(CMAKE_Swift_COMPILER_WORKS TRUE)
  return()
endif()

include(CMakeTestCompilerCommon)

# Remove any cached result from an older CMake version.
# We now store this in CMakeSwiftCompiler.cmake.
unset(CMAKE_Swift_COMPILER_WORKS CACHE)

# This file is used by EnableLanguage in cmGlobalGenerator to
# determine that the selected C++ compiler can actually compile
# and link the most basic of programs.   If not, a fatal error
# is set and cmake stops processing commands and will not generate
# any makefiles or projects.
if(NOT CMAKE_Swift_COMPILER_WORKS)
  PrintTestCompilerStatus("Swift")
  # Clear result from normal variable.
  unset(CMAKE_Swift_COMPILER_WORKS)
  # Puts test result in cache variable.
  string(CONCAT __CMAKE_Swift_TEST_SOURCE
  "public struct CMakeStruct {"
  "  let x: Int"
  "}")
  try_compile(CMAKE_Swift_COMPILER_WORKS
    SOURCE_FROM_VAR main.swift __CMAKE_Swift_TEST_SOURCE
    OUTPUT_VARIABLE __CMAKE_Swift_COMPILER_OUTPUT)
  # Move result from cache to normal variable.
  set(CMAKE_Swift_COMPILER_WORKS ${CMAKE_Swift_COMPILER_WORKS})
  unset(CMAKE_Swift_COMPILER_WORKS CACHE)
  set(Swift_TEST_WAS_RUN 1)
endif()

if(NOT CMAKE_Swift_COMPILER_WORKS)
  PrintTestCompilerResult(CHECK_FAIL "broken")
  string(REPLACE "\n" "\n  " _output "${__CMAKE_Swift_COMPILER_OUTPUT}")
  message(FATAL_ERROR "The Swift compiler\n  \"${CMAKE_Swift_COMPILER}\"\n"
    "is not able to compile a simple test program.\nIt fails "
    "with the following output:\n  ${_output}\n\n"
    "CMake will not be able to correctly generate this project.")
else()
  if(Swift_TEST_WAS_RUN)
    PrintTestCompilerResult(CHECK_PASS "works")
  endif()

  # Unlike C and CXX we do not yet detect any information about the Swift ABI.
  # However, one of the steps done for C and CXX as part of that detection is
  # to initialize the implicit include directories.  That is relevant here.
  set(CMAKE_Swift_IMPLICIT_INCLUDE_DIRECTORIES "${_CMAKE_Swift_IMPLICIT_INCLUDE_DIRECTORIES_INIT}")

  # Re-configure to save learned information.
  configure_file(${CMAKE_ROOT}/Modules/CMakeSwiftCompiler.cmake.in
                 ${CMAKE_PLATFORM_INFO_DIR}/CMakeSwiftCompiler.cmake @ONLY)
  include(${CMAKE_PLATFORM_INFO_DIR}/CMakeSwiftCompiler.cmake)
endif()

unset(__CMAKE_Swift_TEST_SOURCE)
unset(__CMAKE_Swift_COMPILER_OUTPUT)
