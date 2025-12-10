# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


if(CMAKE_OBJC_COMPILER_FORCED)
  # The compiler configuration was forced by the user.
  # Assume the user has configured all compiler information.
  set(CMAKE_OBJC_COMPILER_WORKS TRUE)
  return()
endif()

include(CMakeTestCompilerCommon)

# work around enforced code signing and / or missing executable target type
set(__CMAKE_SAVED_TRY_COMPILE_TARGET_TYPE ${CMAKE_TRY_COMPILE_TARGET_TYPE})
if(_CMAKE_FEATURE_DETECTION_TARGET_TYPE)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE ${_CMAKE_FEATURE_DETECTION_TARGET_TYPE})
endif()

# Remove any cached result from an older CMake version.
# We now store this in CMakeCCompiler.cmake.
unset(CMAKE_OBJC_COMPILER_WORKS CACHE)

# Try to identify the ABI and configure it into CMakeOBJCCompiler.cmake
include(${CMAKE_ROOT}/Modules/CMakeDetermineCompilerABI.cmake)
CMAKE_DETERMINE_COMPILER_ABI(OBJC ${CMAKE_ROOT}/Modules/CMakeOBJCCompilerABI.m)
if(CMAKE_OBJC_ABI_COMPILED)
  # The compiler worked so skip dedicated test below.
  set(CMAKE_OBJC_COMPILER_WORKS TRUE)
  message(STATUS "Check for working OBJC compiler: ${CMAKE_OBJC_COMPILER} - skipped")
endif()

# This file is used by EnableLanguage in cmGlobalGenerator to
# determine that that selected Objective-C compiler can actually compile
# and link the most basic of programs.   If not, a fatal error
# is set and cmake stops processing commands and will not generate
# any makefiles or projects.
if(NOT CMAKE_OBJC_COMPILER_WORKS)
  PrintTestCompilerStatus("OBJC")
  __TestCompiler_setTryCompileTargetType()
  string(CONCAT __TestCompiler_testObjCCompilerSource
    "#ifdef __cplusplus\n"
    "# error \"The CMAKE_OBJC_COMPILER is set to a C++ compiler\"\n"
    "#endif\n"
    "#ifndef __OBJC__\n"
    "# error \"The CMAKE_OBJC_COMPILER is not an Objective-C compiler\"\n"
    "#endif\n"
    "int main(int argc, char* argv[])\n"
    "{ (void)argv; return argc-1;}\n")
  # Clear result from normal variable.
  unset(CMAKE_OBJC_COMPILER_WORKS)
  # Puts test result in cache variable.
  try_compile(CMAKE_OBJC_COMPILER_WORKS
    SOURCE_FROM_VAR testObjCCompiler.m __TestCompiler_testObjCCompilerSource
    OUTPUT_VARIABLE __CMAKE_OBJC_COMPILER_OUTPUT)
  unset(__TestCompiler_testObjCCompilerSource)
  # Move result from cache to normal variable.
  set(CMAKE_OBJC_COMPILER_WORKS ${CMAKE_OBJC_COMPILER_WORKS})
  unset(CMAKE_OBJC_COMPILER_WORKS CACHE)
  __TestCompiler_restoreTryCompileTargetType()
  if(NOT CMAKE_OBJC_COMPILER_WORKS)
    PrintTestCompilerResult(CHECK_FAIL "broken")
    string(REPLACE "\n" "\n  " _output "${__CMAKE_OBJC_COMPILER_OUTPUT}")
    message(FATAL_ERROR "The Objective-C compiler\n  \"${CMAKE_OBJC_COMPILER}\"\n"
      "is not able to compile a simple test program.\nIt fails "
      "with the following output:\n  ${_output}\n\n"
      "CMake will not be able to correctly generate this project.")
  endif()
  PrintTestCompilerResult(CHECK_PASS "works")
endif()

# Try to identify the compiler features
include(${CMAKE_ROOT}/Modules/CMakeDetermineCompilerSupport.cmake)
CMAKE_DETERMINE_COMPILER_SUPPORT(OBJC)

# Re-configure to save learned information.
configure_file(
  ${CMAKE_ROOT}/Modules/CMakeOBJCCompiler.cmake.in
  ${CMAKE_PLATFORM_INFO_DIR}/CMakeOBJCCompiler.cmake
  @ONLY
  )
include(${CMAKE_PLATFORM_INFO_DIR}/CMakeOBJCCompiler.cmake)

if(CMAKE_OBJC_SIZEOF_DATA_PTR)
  foreach(f ${CMAKE_OBJC_ABI_FILES})
    include(${f})
  endforeach()
  unset(CMAKE_OBJC_ABI_FILES)
endif()

set(CMAKE_TRY_COMPILE_TARGET_TYPE ${__CMAKE_SAVED_TRY_COMPILE_TARGET_TYPE})
unset(__CMAKE_SAVED_TRY_COMPILE_TARGET_TYPE)
unset(__CMAKE_OBJC_COMPILER_OUTPUT)
