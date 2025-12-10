# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


if(CMAKE_CXX_COMPILER_FORCED)
  # The compiler configuration was forced by the user.
  # Assume the user has configured all compiler information.
  set(CMAKE_CXX_COMPILER_WORKS TRUE)
  return()
endif()

include(CMakeTestCompilerCommon)

# work around enforced code signing and / or missing executable target type
set(__CMAKE_SAVED_TRY_COMPILE_TARGET_TYPE ${CMAKE_TRY_COMPILE_TARGET_TYPE})
if(_CMAKE_FEATURE_DETECTION_TARGET_TYPE)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE ${_CMAKE_FEATURE_DETECTION_TARGET_TYPE})
endif()

# Remove any cached result from an older CMake version.
# We now store this in CMakeCXXCompiler.cmake.
unset(CMAKE_CXX_COMPILER_WORKS CACHE)

# Try to identify the ABI and configure it into CMakeCXXCompiler.cmake
include(${CMAKE_ROOT}/Modules/CMakeDetermineCompilerABI.cmake)
CMAKE_DETERMINE_COMPILER_ABI(CXX ${CMAKE_ROOT}/Modules/CMakeCXXCompilerABI.cpp)
if(CMAKE_CXX_ABI_COMPILED)
  # The compiler worked so skip dedicated test below.
  set(CMAKE_CXX_COMPILER_WORKS TRUE)
  message(STATUS "Check for working CXX compiler: ${CMAKE_CXX_COMPILER} - skipped")
endif()

# This file is used by EnableLanguage in cmGlobalGenerator to
# determine that the selected C++ compiler can actually compile
# and link the most basic of programs.   If not, a fatal error
# is set and cmake stops processing commands and will not generate
# any makefiles or projects.
if(NOT CMAKE_CXX_COMPILER_WORKS)
  PrintTestCompilerStatus("CXX")
  # FIXME: Use a block() to isolate the variables we set/unset here.
  if(DEFINED CMAKE_CXX_SCAN_FOR_MODULES)
    set(__CMAKE_SAVED_CXX_SCAN_FOR_MODULES "${CMAKE_CXX_SCAN_FOR_MODULES}")
  else()
    unset(__CMAKE_SAVED_CXX_SCAN_FOR_MODULES)
  endif()
  set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
  __TestCompiler_setTryCompileTargetType()
  string(CONCAT __TestCompiler_testCXXCompilerSource
    "#ifndef __cplusplus\n"
    "# error \"The CMAKE_CXX_COMPILER is set to a C compiler\"\n"
    "#endif\n"
    "int main(){return 0;}\n")
  # Clear result from normal variable.
  unset(CMAKE_CXX_COMPILER_WORKS)
  # Puts test result in cache variable.
  try_compile(CMAKE_CXX_COMPILER_WORKS
    SOURCE_FROM_VAR testCXXCompiler.cxx __TestCompiler_testCXXCompilerSource
    OUTPUT_VARIABLE __CMAKE_CXX_COMPILER_OUTPUT)
  if(DEFINED __CMAKE_SAVED_CXX_SCAN_FOR_MODULES)
    set(CMAKE_CXX_SCAN_FOR_MODULES "${__CMAKE_SAVED_CXX_SCAN_FOR_MODULES}")
    unset(__CMAKE_SAVED_CXX_SCAN_FOR_MODULES)
  else()
    unset(CMAKE_CXX_SCAN_FOR_MODULES)
  endif()
  unset(__TestCompiler_testCXXCompilerSource)
  # Move result from cache to normal variable.
  set(CMAKE_CXX_COMPILER_WORKS ${CMAKE_CXX_COMPILER_WORKS})
  unset(CMAKE_CXX_COMPILER_WORKS CACHE)
  __TestCompiler_restoreTryCompileTargetType()
  if(NOT CMAKE_CXX_COMPILER_WORKS)
    PrintTestCompilerResult(CHECK_FAIL "broken")
    string(REPLACE "\n" "\n  " _output "${__CMAKE_CXX_COMPILER_OUTPUT}")
    message(FATAL_ERROR "The C++ compiler\n  \"${CMAKE_CXX_COMPILER}\"\n"
      "is not able to compile a simple test program.\nIt fails "
      "with the following output:\n  ${_output}\n\n"
      "CMake will not be able to correctly generate this project.")
  endif()
  PrintTestCompilerResult(CHECK_PASS "works")
endif()

# Try to identify the compiler features
include(${CMAKE_ROOT}/Modules/CMakeDetermineCompilerSupport.cmake)
CMAKE_DETERMINE_COMPILER_SUPPORT(CXX)

# Re-configure to save learned information.
configure_file(
  ${CMAKE_ROOT}/Modules/CMakeCXXCompiler.cmake.in
  ${CMAKE_PLATFORM_INFO_DIR}/CMakeCXXCompiler.cmake
  @ONLY
  )
include(${CMAKE_PLATFORM_INFO_DIR}/CMakeCXXCompiler.cmake)

if(CMAKE_CXX_SIZEOF_DATA_PTR)
  foreach(f ${CMAKE_CXX_ABI_FILES})
    include(${f})
  endforeach()
  unset(CMAKE_CXX_ABI_FILES)
endif()

set(CMAKE_TRY_COMPILE_TARGET_TYPE ${__CMAKE_SAVED_TRY_COMPILE_TARGET_TYPE})
unset(__CMAKE_SAVED_TRY_COMPILE_TARGET_TYPE)
unset(__CMAKE_CXX_COMPILER_OUTPUT)
