# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

if(CMAKE_CUDA_COMPILER_FORCED)
  # The compiler configuration was forced by the user.
  # Assume the user has configured all compiler information.
  set(CMAKE_CUDA_COMPILER_WORKS TRUE)
  return()
endif()

include(CMakeTestCompilerCommon)

# Remove any cached result from an older CMake version.
# We now store this in CMakeCUDACompiler.cmake.
unset(CMAKE_CUDA_COMPILER_WORKS CACHE)

# Try to identify the ABI and configure it into CMakeCUDACompiler.cmake
include(${CMAKE_ROOT}/Modules/CMakeDetermineCompilerABI.cmake)
CMAKE_DETERMINE_COMPILER_ABI(CUDA ${CMAKE_ROOT}/Modules/CMakeCUDACompilerABI.cu)
if(CMAKE_CUDA_ABI_COMPILED)
  # The compiler worked so skip dedicated test below.
  set(CMAKE_CUDA_COMPILER_WORKS TRUE)
  message(STATUS "Check for working CUDA compiler: ${CMAKE_CUDA_COMPILER} - skipped")

  include(Internal/CMakeCUDAArchitecturesNative)
  # Run the test binary to get:
  # - CMAKE_CUDA_ARCHITECTURES_NATIVE
  cmake_cuda_architectures_native(CUDA)
endif()

# This file is used by EnableLanguage in cmGlobalGenerator to
# determine that the selected cuda compiler can actually compile
# and link the most basic of programs.   If not, a fatal error
# is set and cmake stops processing commands and will not generate
# any makefiles or projects.
if(NOT CMAKE_CUDA_COMPILER_WORKS)
  PrintTestCompilerStatus("CUDA")
  string(CONCAT __TestCompiler_testCudaCompilerSource
    "#ifndef __CUDACC__\n"
    "# error \"The CMAKE_CUDA_COMPILER is set to an invalid CUDA compiler\"\n"
    "#endif\n"
    "int main(){return 0;}\n")

  # Clear result from normal variable.
  unset(CMAKE_CUDA_COMPILER_WORKS)

  # Puts test result in cache variable.
  try_compile(CMAKE_CUDA_COMPILER_WORKS
    SOURCE_FROM_VAR main.cu __TestCompiler_testCudaCompilerSource
    OUTPUT_VARIABLE __CMAKE_CUDA_COMPILER_OUTPUT)
  unset(__TestCompiler_testCudaCompilerSource)

  # Move result from cache to normal variable.
  set(CMAKE_CUDA_COMPILER_WORKS ${CMAKE_CUDA_COMPILER_WORKS})
  unset(CMAKE_CUDA_COMPILER_WORKS CACHE)
  if(NOT CMAKE_CUDA_COMPILER_WORKS)
    PrintTestCompilerResult(CHECK_FAIL "broken")
    string(REPLACE "\n" "\n  " _output "${__CMAKE_CUDA_COMPILER_OUTPUT}")
    message(FATAL_ERROR "The CUDA compiler\n  \"${CMAKE_CUDA_COMPILER}\"\n"
      "is not able to compile a simple test program.\nIt fails "
      "with the following output:\n  ${_output}\n\n"
      "CMake will not be able to correctly generate this project.")
  endif()
  PrintTestCompilerResult(CHECK_PASS "works")
endif()

# Try to identify the compiler features
include(${CMAKE_ROOT}/Modules/CMakeDetermineCompilerSupport.cmake)
CMAKE_DETERMINE_COMPILER_SUPPORT(CUDA)

if("x${CMAKE_CUDA_SIMULATE_ID}" STREQUAL "xMSVC")
  set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "${CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES}")
  set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES}")
endif()

include(Internal/CMakeCUDAFilterImplicitLibs)
# Filter out implicit link libraries that should not be passed unconditionally.
cmake_cuda_filter_implicit_libs(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES)

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  include(Internal/CMakeNVCCFilterImplicitInfo)
  # Match arguments with cmake_nvcc_parse_implicit_info call in CMakeDetermineCUDACompiler.
  cmake_nvcc_filter_implicit_info(CUDA CMAKE_CUDA_)
endif()

# Re-configure to save learned information.
configure_file(
  ${CMAKE_ROOT}/Modules/CMakeCUDACompiler.cmake.in
  ${CMAKE_PLATFORM_INFO_DIR}/CMakeCUDACompiler.cmake
  @ONLY
  )
include(${CMAKE_PLATFORM_INFO_DIR}/CMakeCUDACompiler.cmake)

unset(__CMAKE_CUDA_COMPILER_OUTPUT)
