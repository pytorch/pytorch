cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Modules_CUDA_fix
  /usr/lib/x86_64-linux-gnu/
  ${CMAKE_CURRENT_SOURCE_DIR}/src/TH/cmake
  ${CMAKE_MODULE_PATH})
SET(CMAKE_LIBRARY_PATH /usr/lib/x86_64-linux-gnu/ ${CMAKE_LIBRARY_PATH})
project(ATen)

cmake_policy(SET CMP0012 NEW)

# Polyfill for upstream FindCUDA
include(CMakeInitializeConfigs)

# RPATH stuff
# see https://cmake.org/Wiki/CMake_RPATH_handling
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
endif()
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()

IF(NOT MSVC)
  set(CMAKE_CXX_FLAGS "--std=c++11 -Wall -Wextra -Wno-missing-field-initializers -Wno-type-limits -Wno-unused-parameter -Wno-unknown-pragmas -Wno-vla -fexceptions ${CMAKE_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "-fexceptions ${CMAKE_C_FLAGS}")
  if ($ENV{WERROR})
    set(CMAKE_CXX_FLAGS "-Werror ${CMAKE_CXX_FLAGS}")
  endif()
ENDIF(NOT MSVC)

INCLUDE(CheckCXXSourceCompiles)

# disable some verbose warnings
IF (MSVC)
  set(CMAKE_CXX_FLAGS "/wd4267 /wd4251 /wd4522 /wd4522 /wd4838 /wd4305 /wd4244 /wd4190 /wd4101 /wd4996 /wd4275 ${CMAKE_CXX_FLAGS}")
ENDIF(MSVC)

# windef.h will define max/min macros if NOMINMAX is not defined
IF(MSVC)
  add_definitions(/DNOMINMAX)
ENDIF(MSVC)

#Check if certain std functions are supported. Sometimes
#_GLIBCXX_USE_C99 macro is not defined and some functions are missing.
CHECK_CXX_SOURCE_COMPILES("
#include <cmath>
#include <string>

int main() {
  int a = std::isinf(3.0);
  int b = std::isnan(0.0);
  std::string s = std::to_string(1);

  return 0;
  }" SUPPORT_GLIBCXX_USE_C99)

if(NOT SUPPORT_GLIBCXX_USE_C99)
  message(FATAL_ERROR
          "The C++ compiler does not support required functions. "
          "This is very likely due to a known bug in GCC 5 "
          "(and maybe other versions) on Ubuntu 17.10 and newer. "
          "For more information, see: "
          "https://github.com/pytorch/pytorch/issues/5229"
         )
endif()

# Top-level build config
############################################
# Flags
# When using MSVC

# Detect CUDA architecture and get best NVCC flags
# finding cuda must be first because other things depend on the result
#
# NB: We MUST NOT run this find_package if NO_CUDA is set, because upstream
# FindCUDA has a bug where it will still attempt to make use of NOTFOUND
# compiler variables to run various probe tests.  We could try to fix
# this, but since FindCUDA upstream is subsumed by first-class support
# for CUDA language, it seemed not worth fixing.
IF(NOT CUDA_FOUND AND NOT NO_CUDA)
  FIND_PACKAGE(CUDA 5.5)
ENDIF()

IF(MSVC)
  # we want to respect the standard, and we are bored of those **** .
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE=1)
  LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler /wd4819 -Xcompiler /wd4503 -Xcompiler /wd4190 -Xcompiler /wd4244 -Xcompiler /wd4251 -Xcompiler /wd4275 -Xcompiler /wd4522")
  ADD_DEFINITIONS(-DTH_EXPORTS)
  IF (NOT NO_CUDA)
    ADD_DEFINITIONS(-DTHC_EXPORTS)
  ENDIF()
ENDIF(MSVC)

IF (NOT MSVC)
  IF (CMAKE_VERSION VERSION_LESS "3.1")
    SET(CMAKE_C_FLAGS "-std=c11 ${CMAKE_C_FLAGS}")
  ELSE ()
    SET(CMAKE_C_STANDARD 11)
  ENDIF ()
ENDIF(NOT MSVC)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9")
    if(CUDA_VERSION VERSION_LESS "8.0")
      MESSAGE(STATUS "Found gcc >=5 and CUDA <= 7.5, adding workaround C++ flags")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__")
    endif(CUDA_VERSION VERSION_LESS "8.0")
  endif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9")
endif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

LIST(APPEND CUDA_NVCC_FLAGS -Wno-deprecated-gpu-targets)
LIST(APPEND CUDA_NVCC_FLAGS --expt-extended-lambda)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  SET(CMAKE_CXX_STANDARD 11)
endif()

IF(NOT COMMAND CUDA_SELECT_NVCC_ARCH_FLAGS)
  INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/select_compute_arch.cmake)
ENDIF()
LIST(APPEND CUDA_NVCC_FLAGS $ENV{TORCH_NVCC_FLAGS})
CUDA_SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_EXTRA $ENV{TORCH_CUDA_ARCH_LIST})
LIST(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})
IF(CMAKE_POSITION_INDEPENDENT_CODE AND NOT MSVC)
  LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
ENDIF()

IF(CUDA_HAS_FP16 OR NOT ${CUDA_VERSION} LESS 7.5)
  MESSAGE(STATUS "Found CUDA with FP16 support, compiling with torch.CudaHalfTensor")
  LIST(APPEND CUDA_NVCC_FLAGS "-DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__")
  SET(CMAKE_C_FLAGS "-DCUDA_HAS_FP16=1 ${CMAKE_C_FLAGS}")
ELSE(CUDA_HAS_FP16 OR NOT ${CUDA_VERSION} LESS 7.5)
  MESSAGE(STATUS "Could not find CUDA with FP16 support, compiling without torch.CudaHalfTensor")
ENDIF(CUDA_HAS_FP16 OR NOT ${CUDA_VERSION} LESS 7.5)

OPTION(NDEBUG "disable asserts (WARNING: this may result in silent UB e.g. with out-of-bound indices)")
IF(NOT NDEBUG)
  MESSAGE(STATUS "Removing -DNDEBUG from compile flags")
  STRING(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS "" ${CMAKE_C_FLAGS})
  STRING(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_DEBUG "" ${CMAKE_C_FLAGS_DEBUG})
  STRING(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELEASE "" ${CMAKE_C_FLAGS_RELEASE})
  STRING(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS "" ${CMAKE_CXX_FLAGS})
  STRING(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_DEBUG "" ${CMAKE_CXX_FLAGS_DEBUG})
  STRING(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "" ${CMAKE_CXX_FLAGS_RELEASE})
ENDIF()

# OpenMP support?
SET(WITH_OPENMP ON CACHE BOOL "OpenMP support if available?")
IF (APPLE AND CMAKE_COMPILER_IS_GNUCC)
  EXEC_PROGRAM (uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
  STRING (REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
  MESSAGE (STATUS "MAC OS Darwin Version: ${DARWIN_VERSION}")
  IF (DARWIN_VERSION GREATER 9)
    SET(APPLE_OPENMP_SUCKS 1)
  ENDIF (DARWIN_VERSION GREATER 9)
  EXECUTE_PROCESS (COMMAND ${CMAKE_C_COMPILER} -dumpversion
    OUTPUT_VARIABLE GCC_VERSION)
  IF (APPLE_OPENMP_SUCKS AND GCC_VERSION VERSION_LESS 4.6.2)
    MESSAGE(STATUS "Warning: Disabling OpenMP (unstable with this version of GCC)")
    MESSAGE(STATUS " Install GCC >= 4.6.2 or change your OS to enable OpenMP")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unknown-pragmas")
    SET(WITH_OPENMP OFF CACHE BOOL "OpenMP support if available?" FORCE)
  ENDIF ()
ENDIF ()

IF (WITH_OPENMP AND NOT CHECKED_OPENMP)
  FIND_PACKAGE(OpenMP)
  SET(CHECKED_OPENMP ON CACHE BOOL "already checked for OpenMP")

  # OPENMP_FOUND is not cached in FindOpenMP.cmake (all other variables are cached)
  # see https://github.com/Kitware/CMake/blob/master/Modules/FindOpenMP.cmake
  SET(OPENMP_FOUND ${OPENMP_FOUND} CACHE BOOL "OpenMP Support found")
ENDIF (WITH_OPENMP AND NOT CHECKED_OPENMP)

IF(OPENMP_FOUND)
  MESSAGE(STATUS "Compiling with OpenMP support")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF(OPENMP_FOUND)


SET(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)

FIND_PACKAGE(MAGMA)
IF(CUDA_FOUND AND MAGMA_FOUND)
  INCLUDE_DIRECTORIES("${MAGMA_INCLUDE_DIR}")
  SET(CMAKE_REQUIRED_INCLUDES "${MAGMA_INCLUDE_DIR};${CUDA_INCLUDE_DIRS}")
  INCLUDE(CheckPrototypeDefinition)
  check_prototype_definition(magma_get_sgeqrf_nb
   "magma_int_t magma_get_sgeqrf_nb( magma_int_t m, magma_int_t n );"
   "0"
   "magma.h"
    MAGMA_V2)
  IF (MAGMA_V2)
    add_definitions(-DMAGMA_V2)
  ENDIF (MAGMA_V2)

  SET(USE_MAGMA 1)
  MESSAGE(STATUS "Compiling with MAGMA support")
  MESSAGE(STATUS "MAGMA INCLUDE DIRECTORIES: ${MAGMA_INCLUDE_DIR}")
  MESSAGE(STATUS "MAGMA LIBRARIES: ${MAGMA_LIBRARIES}")
  MESSAGE(STATUS "MAGMA V2 check: ${MAGMA_V2}")
ELSE()
  MESSAGE(STATUS "MAGMA not found. Compiling without MAGMA support")
ENDIF()

# ARM specific flags
FIND_PACKAGE(ARM)
IF (ASIMD_FOUND)
  MESSAGE(STATUS "asimd/Neon found with compiler flag : -D__NEON__")
  SET(CMAKE_C_FLAGS "-D__NEON__ ${CMAKE_C_FLAGS}")
ELSEIF (NEON_FOUND)
  MESSAGE(STATUS "Neon found with compiler flag : -mfpu=neon -D__NEON__")
  SET(CMAKE_C_FLAGS "-mfpu=neon -D__NEON__ ${CMAKE_C_FLAGS}")
ENDIF (ASIMD_FOUND)
IF (CORTEXA8_FOUND)
  MESSAGE(STATUS "Cortex-A8 Found with compiler flag : -mcpu=cortex-a8")
  SET(CMAKE_C_FLAGS "-mcpu=cortex-a8 -fprefetch-loop-arrays ${CMAKE_C_FLAGS}")
ENDIF (CORTEXA8_FOUND)
IF (CORTEXA9_FOUND)
  MESSAGE(STATUS "Cortex-A9 Found with compiler flag : -mcpu=cortex-a9")
  SET(CMAKE_C_FLAGS "-mcpu=cortex-a9 ${CMAKE_C_FLAGS}")
ENDIF (CORTEXA9_FOUND)

IF(UNIX)
  # prevent Unknown CMake command "check_function_exists".
  INCLUDE(CheckFunctionExists)
ENDIF(UNIX)

INCLUDE (CheckIncludeFile)
INCLUDE (CheckCSourceCompiles)
INCLUDE (CheckCSourceRuns)

# Check that our programs run.  This is different from the native CMake compiler
# check, which just tests if the program compiles and links.  This is important
# because with ASAN you might need to help the compiled library find some
# dynamic libraries.
CHECK_C_SOURCE_RUNS("
int main() { return 0; }
" COMPILER_WORKS)
IF(NOT COMPILER_WORKS)
  # Force cmake to retest next time around
  unset(COMPILER_WORKS CACHE)
  MESSAGE(FATAL_ERROR
      "Could not run a simple program built with your compiler. "
      "If you are trying to use -fsanitize=address, make sure "
      "libasan is properly installed on your system (you can confirm "
      "if the problem is this by attempting to build and run a "
      "small program.)")
ENDIF()

CHECK_INCLUDE_FILE(cpuid.h HAVE_CPUID_H)
# Check for a cpuid intrinsic
IF(HAVE_CPUID_H)
    CHECK_C_SOURCE_COMPILES("#include <cpuid.h>
        int main()
        {
            unsigned int eax, ebx, ecx, edx;
            return __get_cpuid(0, &eax, &ebx, &ecx, &edx);
        }" HAVE_GCC_GET_CPUID)
ENDIF()
IF(HAVE_GCC_GET_CPUID)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHAVE_GCC_GET_CPUID")
ENDIF(HAVE_GCC_GET_CPUID)

CHECK_C_SOURCE_COMPILES("#include <stdint.h>
    static inline void cpuid(uint32_t *eax, uint32_t *ebx,
    			 uint32_t *ecx, uint32_t *edx)
    {
      uint32_t a = *eax, b, c = *ecx, d;
      asm volatile ( \"cpuid\" : \"+a\"(a), \"=b\"(b), \"+c\"(c), \"=d\"(d) );
      *eax = a; *ebx = b; *ecx = c; *edx = d;
    }
    int main() {
      uint32_t a,b,c,d;
      cpuid(&a, &b, &c, &d);
      return 0;
    }" NO_GCC_EBX_FPIC_BUG)

IF(NOT NO_GCC_EBX_FPIC_BUG)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_GCC_GET_CPUID")
ENDIF(NOT NO_GCC_EBX_FPIC_BUG)

FIND_PACKAGE(SSE) # checks SSE, AVX and AVX2
IF(C_SSE2_FOUND)
  MESSAGE(STATUS "SSE2 Found")
  SET(CMAKE_C_FLAGS "${C_SSE2_FLAGS} -DUSE_SSE2 ${CMAKE_C_FLAGS}")
ENDIF(C_SSE2_FOUND)
IF(C_SSE4_1_FOUND AND C_SSE4_2_FOUND)
  SET(CMAKE_C_FLAGS "${C_SSE4_1_FLAGS} -DUSE_SSE4_1 ${C_SSE4_2_FLAGS} -DUSE_SSE4_2 ${CMAKE_C_FLAGS}")
ENDIF()
IF(C_SSE3_FOUND)
  MESSAGE(STATUS "SSE3 Found")
  SET(CMAKE_C_FLAGS "${C_SSE3_FLAGS} -DUSE_SSE3 ${CMAKE_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${C_SSE3_FLAGS} -DUSE_SSE3 ${CMAKE_CXX_FLAGS}")
ENDIF(C_SSE3_FOUND)

# we don't set -mavx and -mavx2 flags globally, but only for specific files
# however, we want to enable the AVX codepaths, so we still need to
# add USE_AVX and USE_AVX2 macro defines
IF(C_AVX_FOUND)
  MESSAGE(STATUS "AVX Found")
  SET(CMAKE_C_FLAGS "-DUSE_AVX ${CMAKE_C_FLAGS}")
ENDIF(C_AVX_FOUND)
IF(C_AVX2_FOUND)
  MESSAGE(STATUS "AVX2 Found")
  SET(CMAKE_C_FLAGS "-DUSE_AVX2 ${CMAKE_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "-DUSE_AVX2 ${CMAKE_CXX_FLAGS}")
ENDIF(C_AVX2_FOUND)

CHECK_C_SOURCE_RUNS("
#include <stdatomic.h>
// ATOMIC_INT_LOCK_FREE is flaky on some older gcc versions
// so if this define is not usable a preprocessor definition
// we fail this check and fall back to GCC atomics
#if ATOMIC_INT_LOCK_FREE == 2
#define TH_ATOMIC_IPC_REFCOUNT 1
#endif
int main()
{
  int a;
  int oa;
  atomic_store(&a, 1);
  atomic_fetch_add(&a, 1);
  oa = atomic_load(&a);
  if(!atomic_compare_exchange_strong(&a, &oa, 3))
    return -1;
  return 0;
}
" HAS_C11_ATOMICS)

IF(NOT HAS_C11_ATOMICS)
  CHECK_C_SOURCE_RUNS("
#include <intrin.h>
int main()
{
  long a;
  _InterlockedExchange(&a, 1);
  _InterlockedExchangeAdd(&a, 1);
  if(_InterlockedCompareExchange(&a, 3, 2) != 2)
    return -1;
  return 0;
}
" HAS_MSC_ATOMICS)

  CHECK_C_SOURCE_RUNS("
int main()
{
  int a;
  __sync_lock_test_and_set(&a, 1);
  __sync_fetch_and_add(&a, 1);
  if(!__sync_bool_compare_and_swap(&a, 2, 3))
    return -1;
  return 0;
}
" HAS_GCC_ATOMICS)
ENDIF()

IF(HAS_C11_ATOMICS)
  ADD_DEFINITIONS(-DUSE_C11_ATOMICS=1)
  MESSAGE(STATUS "Atomics: using C11 intrinsics")
ELSEIF(HAS_MSC_ATOMICS)
  ADD_DEFINITIONS(-DUSE_MSC_ATOMICS=1)
  MESSAGE(STATUS "Atomics: using MSVC intrinsics")
ELSEIF(HAS_GCC_ATOMICS)
  ADD_DEFINITIONS(-DUSE_GCC_ATOMICS=1)
    MESSAGE(STATUS "Atomics: using GCC intrinsics")
ELSE()
  SET(CMAKE_THREAD_PREFER_PTHREAD TRUE)
  FIND_PACKAGE(Threads)
  IF(THREADS_FOUND)
    ADD_DEFINITIONS(-DUSE_PTHREAD_ATOMICS=1)
    TARGET_LINK_LIBRARIES(TH ${CMAKE_THREAD_LIBS_INIT})
    MESSAGE(STATUS "Atomics: using pthread")
  ENDIF()
ENDIF()

IF (WIN32 AND NOT CYGWIN)
  SET(BLAS_INSTALL_LIBRARIES "OFF"
    CACHE BOOL "Copy the required BLAS DLLs into the TH install dirs")
ENDIF (WIN32 AND NOT CYGWIN)

MACRO(Install_Required_Library ln)
    get_filename_component(libpath ${ln} PATH)
    get_filename_component(libname ${ln} NAME_WE)
    file(GLOB libdlls "${libpath}/${libname}*.dll")
    install(PROGRAMS ${libdlls}
      DESTINATION "${TH_INSTALL_BIN_SUBDIR}")
ENDMACRO(Install_Required_Library libname)

FIND_PACKAGE(BLAS)
SET(AT_MKL_ENABLED 0)
SET(AT_MKL_MT 0)
IF(BLAS_FOUND)
  SET(USE_BLAS 1)
  IF(BLAS_INFO STREQUAL "mkl")
    ADD_DEFINITIONS(-DTH_BLAS_MKL)
    IF(NOT BLAS_INCLUDE_DIR)
      MESSAGE(FATAL_ERROR "MKL is used, but MKL header files are not found. \
        You can get them by `conda install mkl-include` if using conda (if \
        it is missing, run `conda upgrade -n root conda` first), and \
        `pip install mkl-devel` if using pip. If build fails with header files \
        available in the system, please make sure that CMake will search the \
        directory containing them, e.g., by setting CMAKE_INCLUDE_PATH.")
    ENDIF()
    IF(MSVC AND MKL_LIBRARIES MATCHES ".*libiomp5md\\.lib.*")
      ADD_DEFINITIONS(-D_OPENMP_NOFORCE_MANIFEST)
      SET(AT_MKL_MT 1)
    ENDIF()
    INCLUDE_DIRECTORIES(${BLAS_INCLUDE_DIR})  # include MKL headers
    SET(AT_MKL_ENABLED 1)
  ENDIF()
ENDIF(BLAS_FOUND)

FIND_PACKAGE(LAPACK)
IF(LAPACK_FOUND)
  SET(USE_LAPACK 1)
ENDIF(LAPACK_FOUND)

#############################################

set(ATen_CPU_SRCS)
set(ATen_CPU_INCLUDE)
set(ATen_CUDA_SRCS)
set(ATen_CUDA_INCLUDE)
SET(ATEN_INSTALL_BIN_SUBDIR "bin" CACHE PATH "ATen install binary subdirectory")
SET(ATEN_INSTALL_LIB_SUBDIR "lib" CACHE PATH "ATen install library subdirectory")
SET(ATEN_INSTALL_INCLUDE_SUBDIR "include" CACHE PATH "ATen install include subdirectory")

add_definitions(-DTH_INDEX_BASE=0)
set(TH_LINK_STYLE STATIC)
add_subdirectory(src/TH)
include_directories(
  # dense
  ${CMAKE_CURRENT_SOURCE_DIR}/src/TH
  ${CMAKE_CURRENT_SOURCE_DIR}/src/THC
  ${CMAKE_CURRENT_BINARY_DIR}/src/TH
  ${CMAKE_CURRENT_BINARY_DIR}/src/THC
  # sparse
  ${CMAKE_CURRENT_SOURCE_DIR}/src/THS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/THCS
  ${CMAKE_CURRENT_BINARY_DIR}/src/THS
  ${CMAKE_CURRENT_BINARY_DIR}/src/THCS

  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_BINARY_DIR}/src)
add_subdirectory(src/THNN)
add_subdirectory(src/THS)

if(NO_CUDA)
  message("disabling CUDA because NO_CUDA is set")
  SET(CUDA_FLAG -n)
  SET(AT_CUDA_ENABLED 0)
else()
  SET(AT_CUDA_ENABLED 1)
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
  find_package(CUDA 5.5 REQUIRED)
  add_subdirectory(src/THC)
  add_subdirectory(src/THCUNN)
  add_subdirectory(src/THCS)
endif()

find_package(CuDNN)
IF(NOT AT_CUDA_ENABLED OR NOT CUDNN_FOUND)
  MESSAGE(STATUS "CuDNN not found. Compiling without CuDNN support")
  set(AT_CUDNN_ENABLED 0)
ELSE()
  INCLUDE_DIRECTORIES(BEFORE ${CUDNN_INCLUDE_DIRS})
  set(AT_CUDNN_ENABLED 1)
ENDIF()

if(NO_MKLDNN)
  message("disabling MKLDNN because NO_MKLDNN is set")
  set(AT_MKLDNN_ENABLED 0)
else()
  find_package(MKLDNN)
  if(NOT MKLDNN_FOUND)
    message(STATUS "MKLDNN not found. Compiling without MKLDNN support")
    set(AT_MKLDNN_ENABLED 0)
  else()
    INCLUDE_DIRECTORIES(${MKLDNN_INCLUDE_DIRS})
    set(AT_MKLDNN_ENABLED 1)
  endif()
endif()

set(cwrap_files
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ATen/Declarations.cwrap
  ${CMAKE_CURRENT_SOURCE_DIR}/src/THNN/generic/THNN.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/THCUNN/generic/THCUNN.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ATen/nn.yaml
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ATen/native/native_functions.yaml
)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}/src/THNN
${CMAKE_CURRENT_SOURCE_DIR}/src/THCUNN)

add_subdirectory(src/ATen)
include_directories(
${CMAKE_CURRENT_SOURCE_DIR}/src
${CMAKE_CURRENT_SOURCE_DIR}/src/ATen/utils/catch/single_include
${CMAKE_CURRENT_BINARY_DIR}/src/ATen)
if(NOT NO_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
endif()
add_subdirectory(src/ATen/test)

if(ATEN_NO_CONTRIB)
  message("disable contrib because ATEN_NO_CONTRIB is set")
else()
  add_subdirectory(contrib/data)
  add_subdirectory(contrib/meter)
endif()
