# RPATH stuff
# see https://cmake.org/Wiki/CMake_RPATH_handling
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
  set(_rpath_portable_origin "@loader_path")
else()
  set(_rpath_portable_origin $ORIGIN)
endif(APPLE)
# Use separate rpaths during build and install phases
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
# Don't use the install-rpath during the build phase
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${_rpath_portable_origin}")
# Automatically add all linked folders that are NOT in the build directory to
# the rpath (per library?)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

 # UBSAN triggers when compiling protobuf, so we need to disable it.
set(UBSAN_FLAG "-fsanitize=undefined")

macro(disable_ubsan)
  if(CMAKE_C_FLAGS MATCHES ${UBSAN_FLAG} OR CMAKE_CXX_FLAGS MATCHES ${UBSAN_FLAG})
    set(CAFFE2_UBSAN_ENABLED ON)
    string(REPLACE ${UBSAN_FLAG} "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
    string(REPLACE ${UBSAN_FLAG} "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  endif()
endmacro()

macro(enable_ubsan)
  if(CAFFE2_UBSAN_ENABLED)
    set(CMAKE_C_FLAGS "${UBSAN_FLAG} ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${UBSAN_FLAG} ${CMAKE_CXX_FLAGS}")
  endif()
endmacro()

# ---[ CUDA
if(USE_CUDA)
  # public/*.cmake uses CAFFE2_USE_*
  set(CAFFE2_USE_CUDA ${USE_CUDA})
  set(CAFFE2_USE_CUDNN ${USE_CUDNN})
  set(CAFFE2_USE_NVRTC ${USE_NVRTC})
  set(CAFFE2_USE_TENSORRT ${USE_TENSORRT})
  include(${CMAKE_CURRENT_LIST_DIR}/public/cuda.cmake)
  if(CAFFE2_USE_CUDA)
    # A helper variable recording the list of Caffe2 dependent libraries
    # torch::cudart is dealt with separately, due to CUDA_ADD_LIBRARY
    # design reason (it adds CUDA_LIBRARIES itself).
    set(Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS
      caffe2::cufft caffe2::curand caffe2::cublas)
    if(CAFFE2_USE_NVRTC)
      list(APPEND Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS caffe2::cuda caffe2::nvrtc)
    else()
      caffe2_update_option(USE_NVRTC OFF)
    endif()
    if(CAFFE2_USE_CUDNN)
      list(APPEND Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS caffe2::cudnn-public)
    else()
      caffe2_update_option(USE_CUDNN OFF)
    endif()
    if(CAFFE2_USE_TENSORRT)
      list(APPEND Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS caffe2::tensorrt)
    else()
      caffe2_update_option(USE_TENSORRT OFF)
    endif()
  else()
    message(WARNING
      "Not compiling with CUDA. Suppress this warning with "
      "-DUSE_CUDA=OFF.")
    caffe2_update_option(USE_CUDA OFF)
    caffe2_update_option(USE_CUDNN OFF)
    caffe2_update_option(USE_NVRTC OFF)
    caffe2_update_option(USE_TENSORRT OFF)
    set(CAFFE2_USE_CUDA OFF)
    set(CAFFE2_USE_CUDNN OFF)
    set(CAFFE2_USE_NVRTC OFF)
    set(CAFFE2_USE_TENSORRT OFF)
  endif()
endif()

# ---[ Custom Protobuf
if(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO AND NOT INTERN_BUILD_MOBILE)
  disable_ubsan()
  include(${CMAKE_CURRENT_LIST_DIR}/ProtoBuf.cmake)
  enable_ubsan()
endif()

# For MSVC,
# 1. Remove /Zi, /ZI and /Z7 for Release, MinSizeRel and Default builds
# 2. Switch off incremental linking in debug builds
# 3. If MSVC_Z7_OVERRIDE is ON, then /Zi and /ZI will be replaced with /Z7
#    for Debug and RelWithDebInfo builds
if(MSVC)
  # skip unwanted includes from windows.h
  add_definitions(-DWIN32_LEAN_AND_MEAN)

  # Windows SDK broke compatibility since version 25131, but introduced this define for backward compatibility.
  add_definitions(-D_UCRT_LEGACY_INFINITY)

  foreach(flag_var
      CMAKE_C_FLAGS CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_MINSIZEREL
      CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL)
    if(${flag_var} MATCHES "/Z[iI7]")
      string(REGEX REPLACE "/Z[iI7]" "" ${flag_var} "${${flag_var}}")
    endif()
  endforeach(flag_var)
  if(MSVC_Z7_OVERRIDE)
    foreach(flag_var
        CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/Z[iI]")
        string(REGEX REPLACE "/Z[iI]" "/Z7" ${flag_var} "${${flag_var}}")
      endif()
    endforeach(flag_var)
  endif(MSVC_Z7_OVERRIDE)
  foreach(flag_var
      CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO CMAKE_STATIC_LINKER_FLAGS_RELWITHDEBINFO
      CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO
      CMAKE_SHARED_LINKER_FLAGS_DEBUG CMAKE_STATIC_LINKER_FLAGS_DEBUG
      CMAKE_EXE_LINKER_FLAGS_DEBUG CMAKE_MODULE_LINKER_FLAGS_DEBUG)
    if(${flag_var} MATCHES "/INCREMENTAL" AND NOT ${flag_var} MATCHES "/INCREMENTAL:NO")
      string(REGEX REPLACE "/INCREMENTAL" "/INCREMENTAL:NO" ${flag_var} "${${flag_var}}")
    endif()
  endforeach(flag_var)
endif(MSVC)

# ---[ Threads
include(${CMAKE_CURRENT_LIST_DIR}/public/threads.cmake)
if(TARGET caffe2::Threads)
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS caffe2::Threads)
else()
  message(FATAL_ERROR
      "Cannot find threading library. Caffe2 requires Threads to compile.")
endif()

if(USE_TBB)
  if(USE_SYSTEM_TBB)
    find_package(TBB 2018.0 REQUIRED CONFIG COMPONENTS tbb)

    get_target_property(TBB_INCLUDE_DIR TBB::tbb INTERFACE_INCLUDE_DIRECTORIES)
  else()
    message(STATUS "Compiling TBB from source")
    # Unset our restrictive C++ flags here and reset them later.
    # Remove this once we use proper target_compile_options.
    set(OLD_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    set(CMAKE_CXX_FLAGS)

    set(TBB_ROOT_DIR "${PROJECT_SOURCE_DIR}/third_party/tbb")
    set(TBB_BUILD_STATIC OFF CACHE BOOL " " FORCE)
    set(TBB_BUILD_SHARED ON CACHE BOOL " " FORCE)
    set(TBB_BUILD_TBBMALLOC OFF CACHE BOOL " " FORCE)
    set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL " " FORCE)
    set(TBB_BUILD_TESTS OFF CACHE BOOL " " FORCE)
    add_subdirectory(${PROJECT_SOURCE_DIR}/aten/src/ATen/cpu/tbb)
    set_property(TARGET tbb tbb_def_files PROPERTY FOLDER "dependencies")

    set(CMAKE_CXX_FLAGS ${OLD_CMAKE_CXX_FLAGS})

    set(TBB_INCLUDE_DIR "${TBB_ROOT_DIR}/include")

    add_library(TBB::tbb ALIAS tbb)
  endif()
endif()

# ---[ protobuf
if(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  if(USE_LITE_PROTO)
    set(CAFFE2_USE_LITE_PROTO 1)
  endif()
endif()

# ---[ BLAS

# setting default preferred BLAS options if not already present.
if(NOT INTERN_BUILD_MOBILE)
  set(BLAS "MKL" CACHE STRING "Selected BLAS library")
else()
  set(BLAS "Eigen" CACHE STRING "Selected BLAS library")
  set(AT_MKLDNN_ENABLED 0)
  set(AT_MKL_ENABLED 0)
endif()
set_property(CACHE BLAS PROPERTY STRINGS "ATLAS;BLIS;Eigen;FLAME;Generic;MKL;OpenBLAS;vecLib")
message(STATUS "Trying to find preferred BLAS backend of choice: " ${BLAS})

if(BLAS STREQUAL "Eigen")
  # Eigen is header-only and we do not have any dependent libraries
  set(CAFFE2_USE_EIGEN_FOR_BLAS ON)
elseif(BLAS STREQUAL "ATLAS")
  find_package(Atlas REQUIRED)
  include_directories(SYSTEM ${ATLAS_INCLUDE_DIRS})
  list(APPEND Caffe2_DEPENDENCY_LIBS ${ATLAS_LIBRARIES})
  list(APPEND Caffe2_DEPENDENCY_LIBS cblas)
  set(BLAS_INFO "atlas")
  set(BLAS_FOUND 1)
  set(BLAS_LIBRARIES ${ATLAS_LIBRARIES} cblas)
elseif(BLAS STREQUAL "OpenBLAS")
  find_package(OpenBLAS REQUIRED)
  include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIR})
  list(APPEND Caffe2_DEPENDENCY_LIBS ${OpenBLAS_LIB})
  set(BLAS_INFO "open")
  set(BLAS_FOUND 1)
  set(BLAS_LIBRARIES ${OpenBLAS_LIB})
elseif(BLAS STREQUAL "BLIS")
  find_package(BLIS REQUIRED)
  include_directories(SYSTEM ${BLIS_INCLUDE_DIR})
  list(APPEND Caffe2_DEPENDENCY_LIBS ${BLIS_LIB})
elseif(BLAS STREQUAL "MKL")
  if(BLAS_SET_BY_USER)
    find_package(MKL REQUIRED)
  else()
    find_package(MKL QUIET)
  endif()
  include(${CMAKE_CURRENT_LIST_DIR}/public/mkl.cmake)
  if(MKL_FOUND)
    message(STATUS "MKL libraries: ${MKL_LIBRARIES}")
    message(STATUS "MKL include directory: ${MKL_INCLUDE_DIR}")
    message(STATUS "MKL OpenMP type: ${MKL_OPENMP_TYPE}")
    message(STATUS "MKL OpenMP library: ${MKL_OPENMP_LIBRARY}")
    include_directories(AFTER SYSTEM ${MKL_INCLUDE_DIR})
    list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS caffe2::mkl)
    set(CAFFE2_USE_MKL ON)
    set(BLAS_INFO "mkl")
    set(BLAS_FOUND 1)
    set(BLAS_LIBRARIES ${MKL_LIBRARIES})
  else()
    message(WARNING "MKL could not be found. Defaulting to Eigen")
    set(CAFFE2_USE_EIGEN_FOR_BLAS ON)
    set(CAFFE2_USE_MKL OFF)
  endif()
elseif(BLAS STREQUAL "vecLib")
  find_package(vecLib REQUIRED)
  include_directories(SYSTEM ${vecLib_INCLUDE_DIR})
  list(APPEND Caffe2_DEPENDENCY_LIBS ${vecLib_LINKER_LIBS})
  set(BLAS_INFO "veclib")
  set(BLAS_FOUND 1)
  set(BLAS_LIBRARIES ${vecLib_LINKER_LIBS})
elseif(BLAS STREQUAL "FlexiBLAS")
  find_package(FlexiBLAS REQUIRED)
  include_directories(SYSTEM ${FlexiBLAS_INCLUDE_DIR})
  list(APPEND Caffe2_DEPENDENCY_LIBS ${FlexiBLAS_LIB})
elseif(BLAS STREQUAL "Generic")
  # On Debian family, the CBLAS ABIs have been merged into libblas.so
  if(ENV{GENERIC_BLAS_LIBRARIES} STREQUAL "")
    set(GENERIC_BLAS "blas")
  else()
    set(GENERIC_BLAS $ENV{GENERIC_BLAS_LIBRARIES})
  endif()
  find_library(BLAS_LIBRARIES NAMES ${GENERIC_BLAS})
  message("-- Using BLAS: ${BLAS_LIBRARIES}")
  list(APPEND Caffe2_DEPENDENCY_LIBS ${BLAS_LIBRARIES})
  set(GENERIC_BLAS_FOUND TRUE)
  set(BLAS_INFO "generic")
  set(BLAS_FOUND 1)
else()
  message(FATAL_ERROR "Unrecognized BLAS option: " ${BLAS})
endif()

if(NOT INTERN_BUILD_MOBILE)
  set(AT_MKL_ENABLED 0)
  set(AT_MKL_SEQUENTIAL 0)
  set(AT_MKL_MT 0)
  set(USE_BLAS 1)
  if(NOT (ATLAS_FOUND OR BLIS_FOUND OR GENERIC_BLAS_FOUND OR MKL_FOUND OR OpenBLAS_FOUND OR VECLIB_FOUND OR FlexiBLAS_FOUND))
    message(WARNING "Preferred BLAS (" ${BLAS} ") cannot be found, now searching for a general BLAS library")
    find_package(BLAS)
    if(NOT BLAS_FOUND)
      set(USE_BLAS 0)
    endif()
  endif()

  if(MKL_FOUND)
    if("${MKL_THREADING}" STREQUAL "SEQ")
      set(AT_MKL_SEQUENTIAL 1)
    endif()
    if(MSVC AND MKL_LIBRARIES MATCHES ".*libiomp5md\\.lib.*")
      add_definitions(-D_OPENMP_NOFORCE_MANIFEST)
      set(AT_MKL_MT 1)
    endif()
    set(AT_MKL_ENABLED 1)
  endif()
elseif(INTERN_USE_EIGEN_BLAS)
  # Eigen BLAS for Mobile
  set(USE_BLAS 1)
  include(${CMAKE_CURRENT_LIST_DIR}/External/EigenBLAS.cmake)
  list(APPEND Caffe2_DEPENDENCY_LIBS eigen_blas)
endif()

# ---[ FFTW
set(AT_FFTW_ENABLED 0)
set(USE_FFTW OFF)
if(USE_FFTW OR NOT MKL_FOUND)
  find_library(LIBFFTW3 fftw3)
  if(LIBFFTW3)
    find_path(FFTW3_INCLUDE_DIR NAMES fftw3.h ONLY_CMAKE_FIND_ROOT_PATH)
    if(FFTW3_INCLUDE_DIR)
      SET(AT_FFTW_ENABLED 1)
      SET(USE_FFTW ON)
      include_directories(${FFTW3_INCLUDE_DIR})
    endif()
  endif()
endif()

# --- [ PocketFFT
set(AT_POCKETFFT_ENABLED 0)
if(NOT AT_MKL_ENABLED)
  set(POCKETFFT_INCLUDE_DIR "${Torch_SOURCE_DIR}/third_party/pocketfft/")
  if(NOT EXISTS "${POCKETFFT_INCLUDE_DIR}")
    message(FATAL_ERROR "pocketfft directory not found, expected ${POCKETFFT_INCLUDE_DIR}")
  elif(NOT EXISTS "${POCKETFFT_INCLUDE_DIR}/pocketfft_hdronly.h")
    message(FATAL_ERROR "pocketfft headers not found in ${POCKETFFT_INCLUDE_DIR}")
  endif()

  set(AT_POCKETFFT_ENABLED 1)
  message(STATUS "Using pocketfft in directory: ${POCKETFFT_INCLUDE_DIR}")
endif()

# ---[ Dependencies
# NNPACK and family (QNNPACK, PYTORCH_QNNPACK, and XNNPACK) can download and
# compile their dependencies in isolation as part of their build.  These dependencies
# are then linked statically with PyTorch.  To avoid the possibility of a version
# mismatch between these shared dependencies, explicitly declare our intent to these
# libraries that we are interested in using the exact same source dependencies for all.

if(USE_NNPACK OR USE_QNNPACK OR USE_PYTORCH_QNNPACK OR USE_XNNPACK)
  set(DISABLE_NNPACK_AND_FAMILY OFF)

  # Sanity checks - Can we actually build NNPACK and family given the configuration provided?
  # Disable them and warn the user if not.

  if(IOS)
    list(LENGTH IOS_ARCH IOS_ARCH_COUNT)
    if(IOS_ARCH_COUNT GREATER 1)
      message(WARNING
        "Multi-architecture (${IOS_ARCH}) builds are not supported in {Q/X}NNPACK. "
        "Specify a single architecture in IOS_ARCH and re-configure, or "
        "turn this warning off by USE_{Q/X}NNPACK=OFF.")
      set(DISABLE_NNPACK_AND_FAMILY ON)
    endif()
    if(NOT IOS_ARCH MATCHES "^(i386|x86_64|armv7.*|arm64.*)$")
      message(WARNING
        "Target architecture \"${IOS_ARCH}\" is not supported in {Q/X}NNPACK. "
        "Supported architectures are x86, x86-64, ARM, and ARM64. "
        "Turn this warning off by USE_{Q/X}NNPACK=OFF.")
      set(DISABLE_NNPACK_AND_FAMILY ON)
    endif()
  else()
    if(NOT IOS AND NOT (CMAKE_SYSTEM_NAME MATCHES "^(Android|Linux|Darwin|Windows)$"))
      message(WARNING
        "Target platform \"${CMAKE_SYSTEM_NAME}\" is not supported in {Q/X}NNPACK. "
        "Supported platforms are Android, iOS, Linux, and macOS. "
        "Turn this warning off by USE_{Q/X}NNPACK=OFF.")
      set(DISABLE_NNPACK_AND_FAMILY ON)
    endif()
    if(NOT IOS AND NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "^(i686|AMD64|x86_64|armv[0-9].*|arm64|aarch64)$"))
      message(WARNING
        "Target architecture \"${CMAKE_SYSTEM_PROCESSOR}\" is not supported in {Q/X}NNPACK. "
        "Supported architectures are x86, x86-64, ARM, and ARM64. "
        "Turn this warning off by USE_{Q/X}NNPACK=OFF.")
      set(DISABLE_NNPACK_AND_FAMILY ON)
    endif()
  endif()

  if(DISABLE_NNPACK_AND_FAMILY)
    caffe2_update_option(USE_NNPACK OFF)
    caffe2_update_option(USE_QNNPACK OFF)
    caffe2_update_option(USE_PYTORCH_QNNPACK OFF)
    caffe2_update_option(USE_XNNPACK OFF)
  else()
    # Disable unsupported NNPack combinations with MSVC
    if(MSVC)
      caffe2_update_option(USE_NNPACK OFF)
      caffe2_update_option(USE_QNNPACK OFF)
      caffe2_update_option(USE_PYTORCH_QNNPACK OFF)
    endif()

    set(CAFFE2_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third_party")

    if(NOT DEFINED CPUINFO_SOURCE_DIR)
      set(CPUINFO_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/cpuinfo" CACHE STRING "cpuinfo source directory")
    endif()
    if(NOT DEFINED FP16_SOURCE_DIR)
      set(FP16_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/FP16" CACHE STRING "FP16 source directory")
    endif()
    if(NOT DEFINED FXDIV_SOURCE_DIR)
      set(FXDIV_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/FXdiv" CACHE STRING "FXdiv source directory")
    endif()
    if(NOT DEFINED PSIMD_SOURCE_DIR)
      set(PSIMD_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/psimd" CACHE STRING "PSimd source directory")
    endif()
    if(NOT DEFINED PTHREADPOOL_SOURCE_DIR)
      set(PTHREADPOOL_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/pthreadpool" CACHE STRING "pthreadpool source directory")
    endif()
  endif()
else()
  set(DISABLE_NNPACK_AND_FAMILY ON)
endif()

if(USE_QNNPACK AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" AND CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message(WARNING
    "QNNPACK does not compile for Apple Silicon. "
    "Turn this warning off by explicit USE_QNNPACK=OFF.")
  caffe2_update_option(USE_QNNPACK OFF)
endif()

set(CONFU_DEPENDENCIES_SOURCE_DIR ${PROJECT_BINARY_DIR}/confu-srcs
  CACHE PATH "Confu-style dependencies source directory")
set(CONFU_DEPENDENCIES_BINARY_DIR ${PROJECT_BINARY_DIR}/confu-deps
  CACHE PATH "Confu-style dependencies binary directory")

# ---[ pthreadpool
# Only add a dependency on pthreadpool if we are on a mobile build
# or are building any of the libraries in the {Q/X}NNPACK family.
if(INTERN_BUILD_MOBILE OR NOT DISABLE_NNPACK_AND_FAMILY)
  set(USE_PTHREADPOOL ON CACHE BOOL "" FORCE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_PTHREADPOOL")

  # Always use third_party/pthreadpool.
  set(USE_INTERNAL_PTHREADPOOL_IMPL OFF CACHE BOOL "" FORCE)

  if(NOT TARGET pthreadpool)
    if(USE_SYSTEM_PTHREADPOOL)
      add_library(pthreadpool SHARED IMPORTED)
      find_library(PTHREADPOOL_LIBRARY pthreadpool)
      set_property(TARGET pthreadpool PROPERTY IMPORTED_LOCATION "${PTHREADPOOL_LIBRARY}")
      if(NOT PTHREADPOOL_LIBRARY)
        message(FATAL_ERROR "Cannot find pthreadpool")
      endif()
      message("-- Found pthreadpool: ${PTHREADPOOL_LIBRARY}")
    elseif(NOT USE_INTERNAL_PTHREADPOOL_IMPL)
      if(NOT DEFINED PTHREADPOOL_SOURCE_DIR)
        set(CAFFE2_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third_party")
        set(PTHREADPOOL_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/pthreadpool" CACHE STRING "pthreadpool source directory")
      endif()

      set(PTHREADPOOL_BUILD_TESTS OFF CACHE BOOL "")
      set(PTHREADPOOL_BUILD_BENCHMARKS OFF CACHE BOOL "")
      set(PTHREADPOOL_LIBRARY_TYPE "static" CACHE STRING "")
      set(PTHREADPOOL_ALLOW_DEPRECATED_API ON CACHE BOOL "")
      add_subdirectory(
        "${PTHREADPOOL_SOURCE_DIR}"
        "${CONFU_DEPENDENCIES_BINARY_DIR}/pthreadpool")
      set_property(TARGET pthreadpool PROPERTY POSITION_INDEPENDENT_CODE ON)
    endif()

    if(USE_INTERNAL_PTHREADPOOL_IMPL)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_INTERNAL_PTHREADPOOL_IMPL")
    else()
      list(APPEND Caffe2_DEPENDENCY_LIBS pthreadpool)
    endif()
  endif()
else()
  set(USE_PTHREADPOOL OFF CACHE BOOL "" FORCE)
endif()

# ---[ Caffe2 uses cpuinfo library in the thread pool
if(NOT TARGET cpuinfo AND USE_SYSTEM_CPUINFO)
  add_library(cpuinfo SHARED IMPORTED)
  find_library(CPUINFO_LIBRARY cpuinfo)
  if(NOT CPUINFO_LIBRARY)
    message(FATAL_ERROR "Cannot find cpuinfo")
  endif()
  message("Found cpuinfo: ${CPUINFO_LIBRARY}")
  set_target_properties(cpuinfo PROPERTIES IMPORTED_LOCATION "${CPUINFO_LIBRARY}")
elseif(NOT TARGET cpuinfo)
  if(NOT DEFINED CPUINFO_SOURCE_DIR)
    set(CPUINFO_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party/cpuinfo" CACHE STRING "cpuinfo source directory")
  endif()

  set(CPUINFO_BUILD_TOOLS OFF CACHE BOOL "")
  set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "")
  set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "")
  set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "")
  set(CPUINFO_LIBRARY_TYPE "static" CACHE STRING "")
  set(CPUINFO_LOG_LEVEL "error" CACHE STRING "")
  if(MSVC)
    if(CAFFE2_USE_MSVC_STATIC_RUNTIME)
      set(CPUINFO_RUNTIME_TYPE "static" CACHE STRING "")
    else()
      set(CPUINFO_RUNTIME_TYPE "shared" CACHE STRING "")
    endif()
  endif()
  add_subdirectory(
    "${CPUINFO_SOURCE_DIR}"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/cpuinfo")
  # We build static version of cpuinfo but link
  # them into a shared library for Caffe2, so they need PIC.
  set_property(TARGET cpuinfo PROPERTY POSITION_INDEPENDENT_CODE ON)
  # Need to set this to avoid conflict with XNNPACK's clog external project
  set(CLOG_SOURCE_DIR "${CPUINFO_SOURCE_DIR}/deps/clog")
endif()
list(APPEND Caffe2_DEPENDENCY_LIBS cpuinfo)

# ---[ QNNPACK
if(USE_QNNPACK)
  set(CAFFE2_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third_party")

  if(NOT DEFINED QNNPACK_SOURCE_DIR)
    set(QNNPACK_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/QNNPACK" CACHE STRING "QNNPACK source directory")
  endif()

  if(NOT TARGET qnnpack)
    if(NOT USE_SYSTEM_PTHREADPOOL AND USE_INTERNAL_PTHREADPOOL_IMPL)
      set(QNNPACK_CUSTOM_THREADPOOL ON CACHE BOOL "")
    endif()

    set(QNNPACK_BUILD_TESTS OFF CACHE BOOL "")
    set(QNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "")
    set(QNNPACK_LIBRARY_TYPE "static" CACHE STRING "")
    add_subdirectory(
      "${QNNPACK_SOURCE_DIR}"
      "${CONFU_DEPENDENCIES_BINARY_DIR}/QNNPACK")

    # TODO: See https://github.com/pytorch/pytorch/issues/56285
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      target_compile_options(qnnpack PRIVATE -Wno-deprecated-declarations)
    endif()

    # We build static versions of QNNPACK and pthreadpool but link
    # them into a shared library for Caffe2, so they need PIC.
    set_property(TARGET qnnpack PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET cpuinfo PROPERTY POSITION_INDEPENDENT_CODE ON)

    if(QNNPACK_CUSTOM_THREADPOOL)
      target_compile_definitions(
        qnnpack PRIVATE
        pthreadpool_t=legacy_pthreadpool_t
        pthreadpool_function_1d_t=legacy_pthreadpool_function_1d_t
        pthreadpool_function_1d_tiled_t=legacy_pthreadpool_function_1d_tiled_t
        pthreadpool_function_2d_t=legacy_pthreadpool_function_2d_t
        pthreadpool_function_2d_tiled_t=legacy_pthreadpool_function_2d_tiled_t
        pthreadpool_function_3d_tiled_t=legacy_pthreadpool_function_3d_tiled_t
        pthreadpool_function_4d_tiled_t=legacy_pthreadpool_function_4d_tiled_t
        pthreadpool_create=legacy_pthreadpool_create
        pthreadpool_destroy=legacy_pthreadpool_destroy
        pthreadpool_get_threads_count=legacy_pthreadpool_get_threads_count
        pthreadpool_compute_1d=legacy_pthreadpool_compute_1d
        pthreadpool_parallelize_1d=legacy_pthreadpool_parallelize_1d
        pthreadpool_compute_1d_tiled=legacy_pthreadpool_compute_1d_tiled
        pthreadpool_compute_2d=legacy_pthreadpool_compute_2d
        pthreadpool_compute_2d_tiled=legacy_pthreadpool_compute_2d_tiled
        pthreadpool_compute_3d_tiled=legacy_pthreadpool_compute_3d_tiled
        pthreadpool_compute_4d_tiled=legacy_pthreadpool_compute_4d_tiled)
    endif()
  endif()

  list(APPEND Caffe2_DEPENDENCY_LIBS qnnpack)
endif()

# ---[ Caffe2 Int8 operators (enabled by USE_QNNPACK) depend on gemmlowp and neon2sse headers
if(USE_QNNPACK)
  set(CAFFE2_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third_party")
  include_directories(SYSTEM "${CAFFE2_THIRD_PARTY_ROOT}/gemmlowp")
  include_directories(SYSTEM "${CAFFE2_THIRD_PARTY_ROOT}/neon2sse")
endif()

# ---[ PYTORCH_QNNPACK
if(USE_PYTORCH_QNNPACK)
    if(NOT DEFINED PYTORCH_QNNPACK_SOURCE_DIR)
      set(PYTORCH_QNNPACK_SOURCE_DIR "${PROJECT_SOURCE_DIR}/aten/src/ATen/native/quantized/cpu/qnnpack" CACHE STRING "QNNPACK source directory")
    endif()

    if(NOT TARGET pytorch_qnnpack)
      if(NOT USE_SYSTEM_PTHREADPOOL AND USE_INTERNAL_PTHREADPOOL_IMPL)
        set(PYTORCH_QNNPACK_CUSTOM_THREADPOOL ON CACHE BOOL "")
      endif()

      set(PYTORCH_QNNPACK_BUILD_TESTS OFF CACHE BOOL "")
      set(PYTORCH_QNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "")
      set(PYTORCH_QNNPACK_LIBRARY_TYPE "static" CACHE STRING "")
      add_subdirectory(
        "${PYTORCH_QNNPACK_SOURCE_DIR}"
        "${CONFU_DEPENDENCIES_BINARY_DIR}/pytorch_qnnpack")
      # We build static versions of QNNPACK and pthreadpool but link
      # them into a shared library for Caffe2, so they need PIC.
      set_property(TARGET pytorch_qnnpack PROPERTY POSITION_INDEPENDENT_CODE ON)
      set_property(TARGET cpuinfo PROPERTY POSITION_INDEPENDENT_CODE ON)

      if(PYTORCH_QNNPACK_CUSTOM_THREADPOOL)
        target_compile_definitions(
          pytorch_qnnpack PRIVATE
          pthreadpool_t=legacy_pthreadpool_t
          pthreadpool_function_1d_t=legacy_pthreadpool_function_1d_t
          pthreadpool_function_1d_tiled_t=legacy_pthreadpool_function_1d_tiled_t
          pthreadpool_function_2d_t=legacy_pthreadpool_function_2d_t
          pthreadpool_function_2d_tiled_t=legacy_pthreadpool_function_2d_tiled_t
          pthreadpool_function_3d_tiled_t=legacy_pthreadpool_function_3d_tiled_t
          pthreadpool_function_4d_tiled_t=legacy_pthreadpool_function_4d_tiled_t
          pthreadpool_create=legacy_pthreadpool_create
          pthreadpool_destroy=legacy_pthreadpool_destroy
          pthreadpool_get_threads_count=legacy_pthreadpool_get_threads_count
          pthreadpool_compute_1d=legacy_pthreadpool_compute_1d
          pthreadpool_parallelize_1d=legacy_pthreadpool_parallelize_1d
          pthreadpool_compute_1d_tiled=legacy_pthreadpool_compute_1d_tiled
          pthreadpool_compute_2d=legacy_pthreadpool_compute_2d
          pthreadpool_compute_2d_tiled=legacy_pthreadpool_compute_2d_tiled
          pthreadpool_compute_3d_tiled=legacy_pthreadpool_compute_3d_tiled
          pthreadpool_compute_4d_tiled=legacy_pthreadpool_compute_4d_tiled)
      endif()
    endif()

    list(APPEND Caffe2_DEPENDENCY_LIBS pytorch_qnnpack)
endif()

# ---[ NNPACK
if(USE_NNPACK)
  include(${CMAKE_CURRENT_LIST_DIR}/External/nnpack.cmake)
  if(NNPACK_FOUND)
    if(TARGET nnpack)
      # ---[ NNPACK is being built together with Caffe2: explicitly specify dependency
      list(APPEND Caffe2_DEPENDENCY_LIBS nnpack)
    else()
      include_directories(SYSTEM ${NNPACK_INCLUDE_DIRS})
      list(APPEND Caffe2_DEPENDENCY_LIBS ${NNPACK_LIBRARIES})
    endif()
  else()
    message(WARNING "Not compiling with NNPACK. Suppress this warning with -DUSE_NNPACK=OFF")
    caffe2_update_option(USE_NNPACK OFF)
  endif()
endif()

# ---[ XNNPACK
if(USE_XNNPACK AND NOT USE_SYSTEM_XNNPACK)
  if(NOT DEFINED XNNPACK_SOURCE_DIR)
    set(XNNPACK_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/XNNPACK" CACHE STRING "XNNPACK source directory")
  endif()

  if(NOT DEFINED XNNPACK_INCLUDE_DIR)
    set(XNNPACK_INCLUDE_DIR "${XNNPACK_SOURCE_DIR}/include" CACHE STRING "XNNPACK include directory")
  endif()

  if(NOT TARGET XNNPACK)
    set(XNNPACK_LIBRARY_TYPE "static" CACHE STRING "")
    set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "")
    set(XNNPACK_BUILD_TESTS OFF CACHE BOOL "")

    add_subdirectory(
      "${XNNPACK_SOURCE_DIR}"
      "${CONFU_DEPENDENCIES_BINARY_DIR}/XNNPACK")

    set_property(TARGET XNNPACK PROPERTY POSITION_INDEPENDENT_CODE ON)
    # Workaround for https://github.com/pytorch/pytorch/issues/47292
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_COMPILER_IS_GNUCXX AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.5.0))
      # Compiling qu8-requantization/precise-psimd.c without any optimization flags on gcc-7.4 or older i
      # Fails with internal compiler error
      # Workaround by forcing -O1 for XNNPACK (i.e. build it with RelWithDebInfo)
      set_property(TARGET XNNPACK APPEND_STRING PROPERTY COMPILE_FLAGS "-O1")
    endif()
  endif()

  include_directories(SYSTEM ${XNNPACK_INCLUDE_DIR})
  list(APPEND Caffe2_DEPENDENCY_LIBS XNNPACK)
elseif(NOT TARGET XNNPACK AND USE_SYSTEM_XNNPACK)
  add_library(XNNPACK SHARED IMPORTED)
  find_library(XNNPACK_LIBRARY XNNPACK)
  set_property(TARGET XNNPACK PROPERTY IMPORTED_LOCATION "${XNNPACK_LIBRARY}")
  if(NOT XNNPACK_LIBRARY)
    message(FATAL_ERROR "Cannot find XNNPACK")
  endif()
  message("-- Found XNNPACK: ${XNNPACK_LIBRARY}")
  list(APPEND Caffe2_DEPENDENCY_LIBS XNNPACK)
endif()

# ---[ Vulkan deps
if(USE_VULKAN)
  set(Vulkan_DEFINES)
  set(Vulkan_INCLUDES)
  set(Vulkan_LIBS)
  include(${CMAKE_CURRENT_LIST_DIR}/VulkanDependencies.cmake)
  string(APPEND CMAKE_CXX_FLAGS ${Vulkan_DEFINES})
  include_directories(SYSTEM ${Vulkan_INCLUDES})
  list(APPEND Caffe2_DEPENDENCY_LIBS ${Vulkan_LIBS})
endif()

# ---[ gflags
if(USE_GFLAGS)
  include(${CMAKE_CURRENT_LIST_DIR}/public/gflags.cmake)
  if(NOT TARGET gflags)
    message(WARNING
        "gflags is not found. Caffe2 will build without gflags support but "
        "it is strongly recommended that you install gflags. Suppress this "
        "warning with -DUSE_GFLAGS=OFF")
    caffe2_update_option(USE_GFLAGS OFF)
  endif()
endif()

# ---[ Google-glog
if(USE_GLOG)
  include(${CMAKE_CURRENT_LIST_DIR}/public/glog.cmake)
  if(TARGET glog::glog)
    set(CAFFE2_USE_GOOGLE_GLOG 1)
    include_directories(SYSTEM ${GLOG_INCLUDE_DIR})
    list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS glog::glog)
  else()
    message(WARNING
        "glog is not found. Caffe2 will build without glog support but it is "
        "strongly recommended that you install glog. Suppress this warning "
        "with -DUSE_GLOG=OFF")
    caffe2_update_option(USE_GLOG OFF)
  endif()
endif()


# ---[ Googletest and benchmark
if(BUILD_TEST OR BUILD_MOBILE_BENCHMARK OR BUILD_MOBILE_TEST)
  # Preserve build options.
  set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})

  # We will build gtest as static libs and embed it directly into the binary.
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)

  # For gtest, we will simply embed it into our test binaries, so we won't
  # need to install it.
  set(INSTALL_GTEST OFF CACHE BOOL "Install gtest." FORCE)
  set(BUILD_GMOCK ON CACHE BOOL "Build gmock." FORCE)
  # For Windows, we will check the runtime used is correctly passed in.
  if(NOT CAFFE2_USE_MSVC_STATIC_RUNTIME)
      set(gtest_force_shared_crt ON CACHE BOOL "force shared crt on gtest" FORCE)
  endif()
  # We need to replace googletest cmake scripts too.
  # Otherwise, it will sometimes break the build.
  # To make the git clean after the build, we make a backup first.
  if((MSVC AND MSVC_Z7_OVERRIDE) OR USE_CUDA)
    execute_process(
      COMMAND ${CMAKE_COMMAND}
              "-DFILENAME=${CMAKE_CURRENT_LIST_DIR}/../third_party/googletest/googletest/cmake/internal_utils.cmake"
              "-DBACKUP=${CMAKE_CURRENT_LIST_DIR}/../third_party/googletest/googletest/cmake/internal_utils.cmake.bak"
              "-DREVERT=0"
              "-P"
              "${CMAKE_CURRENT_LIST_DIR}/GoogleTestPatch.cmake"
      RESULT_VARIABLE _exitcode)
    if(NOT _exitcode EQUAL 0)
      message(WARNING "Patching failed for Google Test. The build may fail.")
    endif()
  endif()

  # Add googletest subdirectory but make sure our INCLUDE_DIRECTORIES
  # don't bleed into it. This is because libraries installed into the root conda
  # env (e.g. MKL) add a global /opt/conda/include directory, and if there's
  # gtest installed in conda, the third_party/googletest/**.cc source files
  # would try to include headers from /opt/conda/include/gtest/**.h instead of
  # its own. Once we have proper target-based include directories,
  # this shouldn't be necessary anymore.
  get_property(INC_DIR_temp DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
  set_property(DIRECTORY PROPERTY INCLUDE_DIRECTORIES "")
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/googletest)
  set_property(DIRECTORY PROPERTY INCLUDE_DIRECTORIES ${INC_DIR_temp})

  include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/googletest/googletest/include)
  include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/googletest/googlemock/include)

  # We will not need to test benchmark lib itself.
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing as we don't need it.")
  # We will not need to install benchmark since we link it statically.
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install to avoid overwriting vendor install.")
  if(NOT USE_SYSTEM_BENCHMARK)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/benchmark)
  else()
    add_library(benchmark SHARED IMPORTED)
    find_library(BENCHMARK_LIBRARY benchmark)
    if(NOT BENCHMARK_LIBRARY)
      message(FATAL_ERROR "Cannot find google benchmark library")
    endif()
    message("-- Found benchmark: ${BENCHMARK_LIBRARY}")
    set_property(TARGET benchmark PROPERTY IMPORTED_LOCATION ${BENCHMARK_LIBRARY})
  endif()
  include_directories(${CMAKE_CURRENT_LIST_DIR}/../third_party/benchmark/include)

  # Recover build options.
  set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libs" FORCE)

  # To make the git clean after the build, we revert the changes here.
  if(MSVC AND MSVC_Z7_OVERRIDE)
    execute_process(
      COMMAND ${CMAKE_COMMAND}
              "-DFILENAME=${CMAKE_CURRENT_LIST_DIR}/../third_party/googletest/googletest/cmake/internal_utils.cmake"
              "-DBACKUP=${CMAKE_CURRENT_LIST_DIR}/../third_party/googletest/googletest/cmake/internal_utils.cmake.bak"
              "-DREVERT=1"
              "-P"
              "${CMAKE_CURRENT_LIST_DIR}/GoogleTestPatch.cmake"
      RESULT_VARIABLE _exitcode)
    if(NOT _exitcode EQUAL 0)
      message(WARNING "Reverting changes failed for Google Test. The build may fail.")
    endif()
  endif()

  # Cacheing variables to enable incremental build.
  # Without this is cross compiling we end up having to blow build directory
  # and rebuild from scratch.
  if(CMAKE_CROSSCOMPILING)
    if(COMPILE_HAVE_STD_REGEX)
      set(RUN_HAVE_STD_REGEX 0 CACHE INTERNAL "Cache RUN_HAVE_STD_REGEX output for cross-compile.")
    endif()
  endif()
endif()

# ---[ FBGEMM
if(USE_FBGEMM)
  set(CAFFE2_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third_party")
  if(NOT DEFINED FBGEMM_SOURCE_DIR)
    set(FBGEMM_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/fbgemm" CACHE STRING "FBGEMM source directory")
  endif()
  if(NOT CAFFE2_COMPILER_SUPPORTS_AVX512_EXTENSIONS)
    message(WARNING
      "A compiler with AVX512 support is required for FBGEMM. "
      "Not compiling with FBGEMM. "
      "Turn this warning off by USE_FBGEMM=OFF.")
    set(USE_FBGEMM OFF)
  endif()
  if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(WARNING
      "x64 operating system is required for FBGEMM. "
      "Not compiling with FBGEMM. "
      "Turn this warning off by USE_FBGEMM=OFF.")
    set(USE_FBGEMM OFF)
  endif()
  if(USE_FBGEMM AND NOT TARGET fbgemm)
    set(FBGEMM_BUILD_TESTS OFF CACHE BOOL "")
    set(FBGEMM_BUILD_BENCHMARKS OFF CACHE BOOL "")
    if(MSVC AND BUILD_SHARED_LIBS)
      set(FBGEMM_LIBRARY_TYPE "shared" CACHE STRING "")
    else()
      set(FBGEMM_LIBRARY_TYPE "static" CACHE STRING "")
    endif()
    add_subdirectory("${FBGEMM_SOURCE_DIR}")
    set_property(TARGET fbgemm_generic PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET fbgemm_avx2 PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET fbgemm_avx512 PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET fbgemm PROPERTY POSITION_INDEPENDENT_CODE ON)
    if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 13.0.0)
      # See https://github.com/pytorch/pytorch/issues/74352
      target_compile_options_if_supported(asmjit -Wno-deprecated-copy)
      target_compile_options_if_supported(asmjit -Wno-unused-but-set-variable)
    endif()
  endif()

  if(USE_FBGEMM)
    list(APPEND Caffe2_DEPENDENCY_LIBS fbgemm)
  endif()
endif()

if(USE_FBGEMM)
  caffe2_update_option(USE_FBGEMM ON)
else()
  caffe2_update_option(USE_FBGEMM OFF)
  message(WARNING
    "Turning USE_FAKELOWP off as it depends on USE_FBGEMM.")
  caffe2_update_option(USE_FAKELOWP OFF)
endif()

# ---[ LMDB
if(USE_LMDB)
  find_package(LMDB)
  if(LMDB_FOUND)
    include_directories(SYSTEM ${LMDB_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${LMDB_LIBRARIES})
  else()
    message(WARNING "Not compiling with LMDB. Suppress this warning with -DUSE_LMDB=OFF")
    caffe2_update_option(USE_LMDB OFF)
  endif()
endif()

if(USE_OPENCL)
  message(INFO "USING OPENCL")
  find_package(OpenCL REQUIRED)
  include_directories(SYSTEM ${OpenCL_INCLUDE_DIRS})
  include_directories(${CMAKE_CURRENT_LIST_DIR}/../caffe2/contrib/opencl)
  list(APPEND Caffe2_DEPENDENCY_LIBS ${OpenCL_LIBRARIES})
endif()

# ---[ LevelDB
# ---[ Snappy
if(USE_LEVELDB)
  find_package(LevelDB)
  find_package(Snappy)
  if(LEVELDB_FOUND AND SNAPPY_FOUND)
    include_directories(SYSTEM ${LevelDB_INCLUDE})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${LevelDB_LIBRARIES})
    include_directories(SYSTEM ${Snappy_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${Snappy_LIBRARIES})
  else()
    message(WARNING "Not compiling with LevelDB. Suppress this warning with -DUSE_LEVELDB=OFF")
    caffe2_update_option(USE_LEVELDB OFF)
  endif()
endif()

# ---[ NUMA
if(USE_NUMA)
  if(LINUX)
    find_package(Numa)
    if(NUMA_FOUND)
      include_directories(SYSTEM ${Numa_INCLUDE_DIR})
      list(APPEND Caffe2_DEPENDENCY_LIBS ${Numa_LIBRARIES})
    else()
      message(WARNING "Not compiling with NUMA. Suppress this warning with -DUSE_NUMA=OFF")
      caffe2_update_option(USE_NUMA OFF)
    endif()
  else()
    message(WARNING "NUMA is currently only supported under Linux.")
    caffe2_update_option(USE_NUMA OFF)
  endif()
endif()

# ---[ ZMQ
if(USE_ZMQ)
  find_package(ZMQ)
  if(ZMQ_FOUND)
    include_directories(SYSTEM ${ZMQ_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${ZMQ_LIBRARIES})
  else()
    message(WARNING "Not compiling with ZMQ. Suppress this warning with -DUSE_ZMQ=OFF")
    caffe2_update_option(USE_ZMQ OFF)
  endif()
endif()

# ---[ Redis
if(USE_REDIS)
  find_package(Hiredis)
  if(HIREDIS_FOUND)
    include_directories(SYSTEM ${Hiredis_INCLUDE})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${Hiredis_LIBRARIES})
  else()
    message(WARNING "Not compiling with Redis. Suppress this warning with -DUSE_REDIS=OFF")
    caffe2_update_option(USE_REDIS OFF)
  endif()
endif()


# ---[ OpenCV
if(USE_OPENCV)
  # OpenCV 4
  find_package(OpenCV 4 QUIET COMPONENTS core highgui imgproc imgcodecs optflow videoio video)
  if(NOT OpenCV_FOUND)
    # OpenCV 3
    find_package(OpenCV 3 QUIET COMPONENTS core highgui imgproc imgcodecs videoio video)
    if(NOT OpenCV_FOUND)
      # OpenCV 2
      find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
    endif()
  endif()
  if(OpenCV_FOUND)
    include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${OpenCV_LIBS})
    if(MSVC AND USE_CUDA)
        list(APPEND Caffe2_CUDA_DEPENDENCY_LIBS ${OpenCV_LIBS})
    endif()
    message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
  else()
    message(WARNING "Not compiling with OpenCV. Suppress this warning with -DUSE_OPENCV=OFF")
    caffe2_update_option(USE_OPENCV OFF)
  endif()
endif()

# ---[ FFMPEG
if(USE_FFMPEG)
  find_package(FFmpeg REQUIRED)
  if(FFMPEG_FOUND)
    message("Found FFMPEG/LibAV libraries")
    include_directories(SYSTEM ${FFMPEG_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${FFMPEG_LIBRARIES})
  else()
    message("Not compiling with FFmpeg. Suppress this warning with -DUSE_FFMPEG=OFF")
    caffe2_update_option(USE_FFMPEG OFF)
  endif()
endif()

if(USE_ITT)
  find_package(ITT)
  if(ITT_FOUND)
    include_directories(SYSTEM ${ITT_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${ITT_LIBRARIES})
    list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${ITT_LIBRARIES})
  else()
    message(WARNING "Not compiling with ITT. Suppress this warning with -DUSE_ITT=OFF")
    set(USE_ITT OFF CACHE BOOL "" FORCE)
    caffe2_update_option(USE_ITT OFF)
  endif()
endif()

# ---[ Caffe2 depends on FP16 library for half-precision conversions
if(NOT TARGET fp16 AND NOT USE_SYSTEM_FP16)
  set(CAFFE2_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third_party")
  # PSIMD is required by FP16
  if(NOT DEFINED PSIMD_SOURCE_DIR)
    set(PSIMD_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/psimd" CACHE STRING "PSimd source directory")
  endif()
  if(NOT DEFINED FP16_SOURCE_DIR)
    set(FP16_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/FP16" CACHE STRING "FP16 source directory")
  endif()

  set(FP16_BUILD_TESTS OFF CACHE BOOL "")
  set(FP16_BUILD_BENCHMARKS OFF CACHE BOOL "")
  add_subdirectory(
    "${FP16_SOURCE_DIR}"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/FP16")
elseif(NOT TARGET fp16 AND USE_SYSTEM_FP16)
  add_library(fp16 STATIC "/usr/include/fp16.h")
  set_target_properties(fp16 PROPERTIES LINKER_LANGUAGE C)
endif()
list(APPEND Caffe2_DEPENDENCY_LIBS fp16)

# ---[ EIGEN
# Due to license considerations, we will only use the MPL2 parts of Eigen.
set(EIGEN_MPL2_ONLY 1)
if(USE_SYSTEM_EIGEN_INSTALL)
  find_package(Eigen3)
  if(EIGEN3_FOUND)
    message(STATUS "Found system Eigen at " ${EIGEN3_INCLUDE_DIR})
  else()
    message(STATUS "Did not find system Eigen. Using third party subdirectory.")
    set(EIGEN3_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../third_party/eigen)
    caffe2_update_option(USE_SYSTEM_EIGEN_INSTALL OFF)
  endif()
else()
  message(STATUS "Using third party subdirectory Eigen.")
  set(EIGEN3_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../third_party/eigen)
endif()
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

# ---[ Python + Numpy
if(BUILD_PYTHON)
  # If not given a Python installation, then use the current active Python
  if(NOT PYTHON_EXECUTABLE)
    execute_process(
      COMMAND "which" "python" RESULT_VARIABLE _exitcode OUTPUT_VARIABLE _py_exe)
    if(${_exitcode} EQUAL 0)
      if(NOT MSVC)
        string(STRIP ${_py_exe} PYTHON_EXECUTABLE)
      endif()
      message(STATUS "Setting Python to ${PYTHON_EXECUTABLE}")
    endif()
  endif()

  # Check that Python works
  set(PYTHON_VERSION)
  if(DEFINED PYTHON_EXECUTABLE)
    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" "--version"
        RESULT_VARIABLE _exitcode OUTPUT_VARIABLE PYTHON_VERSION)
    if(NOT _exitcode EQUAL 0)
      message(FATAL_ERROR "The Python executable ${PYTHON_EXECUTABLE} cannot be run. Make sure that it is an absolute path.")
    endif()
    if(PYTHON_VERSION)
      string(REGEX MATCH "([0-9]+)\\.([0-9]+)" PYTHON_VERSION ${PYTHON_VERSION})
    endif()
  endif()

  # Seed PYTHON_INCLUDE_DIR and PYTHON_LIBRARY to be consistent with the
  # executable that we already found (if we didn't actually find an executable
  # then these will just use "python", but at least they'll be consistent with
  # each other).
  if(NOT PYTHON_INCLUDE_DIR)
    # TODO: Verify that sysconfig isn't inaccurate
    pycmd_no_exit(_py_inc _exitcode "import sysconfig; print(sysconfig.get_path('include'))")
    if("${_exitcode}" EQUAL 0 AND IS_DIRECTORY "${_py_inc}")
      set(PYTHON_INCLUDE_DIR "${_py_inc}")
      message(STATUS "Setting Python's include dir to ${_py_inc} from sysconfig")
    else()
      message(WARNING "Could not set Python's include dir to ${_py_inc} from sysconfig")
    endif()
  endif(NOT PYTHON_INCLUDE_DIR)

  if(NOT PYTHON_LIBRARY)
    pycmd_no_exit(_py_lib _exitcode "import sysconfig; print(sysconfig.get_path('stdlib'))")
    if("${_exitcode}" EQUAL 0 AND EXISTS "${_py_lib}" AND EXISTS "${_py_lib}")
      set(PYTHON_LIBRARY "${_py_lib}")
      if(MSVC)
        string(REPLACE "Lib" "libs" _py_static_lib ${_py_lib})
        link_directories(${_py_static_lib})
      endif()
      message(STATUS "Setting Python's library to ${PYTHON_LIBRARY}")
    endif()
  endif(NOT PYTHON_LIBRARY)

  # These should fill in the rest of the variables, like versions, but resepct
  # the variables we set above
  set(Python_ADDITIONAL_VERSIONS ${PYTHON_VERSION} 3.8 3.7)
  find_package(PythonInterp 3.0)
  find_package(PythonLibs 3.0)

  if(${PYTHONLIBS_VERSION_STRING} VERSION_LESS 3)
    message(FATAL_ERROR
      "Found Python libraries version ${PYTHONLIBS_VERSION_STRING}. Python 2 has reached end-of-life and is no longer supported by PyTorch.")
  endif()
  if(${PYTHONLIBS_VERSION_STRING} VERSION_LESS 3.7)
    message(FATAL_ERROR
      "Found Python libraries version ${PYTHONLIBS_VERSION_STRING}. Python 3.6 is no longer supported by PyTorch.")
  endif()

  # When building pytorch, we pass this in directly from setup.py, and
  # don't want to overwrite it because we trust python more than cmake
  if(NUMPY_INCLUDE_DIR)
    set(NUMPY_FOUND ON)
  elseif(USE_NUMPY)
    find_package(NumPy)
    if(NOT NUMPY_FOUND)
      message(WARNING "NumPy could not be found. Not building with NumPy. Suppress this warning with -DUSE_NUMPY=OFF")
    endif()
  endif()

  if(PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND)
    add_library(python::python INTERFACE IMPORTED)
    target_include_directories(python::python SYSTEM INTERFACE ${PYTHON_INCLUDE_DIRS})
    if(WIN32)
      target_link_libraries(python::python INTERFACE ${PYTHON_LIBRARIES})
    endif()

    caffe2_update_option(USE_NUMPY OFF)
    if(NUMPY_FOUND)
      caffe2_update_option(USE_NUMPY ON)
      add_library(numpy::numpy INTERFACE IMPORTED)
      target_include_directories(numpy::numpy SYSTEM INTERFACE ${NUMPY_INCLUDE_DIR})
    endif()
    # Observers are required in the python build
    caffe2_update_option(USE_OBSERVERS ON)
  else()
    message(WARNING "Python dependencies not met. Not compiling with python. Suppress this warning with -DBUILD_PYTHON=OFF")
    caffe2_update_option(BUILD_PYTHON OFF)
  endif()
endif()

# ---[ pybind11
if(USE_SYSTEM_PYBIND11)
  find_package(pybind11 CONFIG)
  if(NOT pybind11_FOUND)
    find_package(pybind11)
  endif()
  if(NOT pybind11_FOUND)
    message(FATAL "Cannot find system pybind11")
  endif()
else()
    message(STATUS "Using third_party/pybind11.")
    set(pybind11_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../third_party/pybind11/include)
    install(DIRECTORY ${pybind11_INCLUDE_DIRS}
            DESTINATION ${CMAKE_INSTALL_PREFIX}
            FILES_MATCHING PATTERN "*.h")
endif()
message(STATUS "pybind11 include dirs: " "${pybind11_INCLUDE_DIRS}")
add_library(pybind::pybind11 INTERFACE IMPORTED)
target_include_directories(pybind::pybind11 SYSTEM INTERFACE ${pybind11_INCLUDE_DIRS})
target_link_libraries(pybind::pybind11 INTERFACE python::python)

# ---[ MPI
if(USE_MPI)
  find_package(MPI)
  if(MPI_CXX_FOUND)
    message(STATUS "MPI support found")
    message(STATUS "MPI compile flags: " ${MPI_CXX_COMPILE_FLAGS})
    message(STATUS "MPI include path: " ${MPI_CXX_INCLUDE_PATH})
    message(STATUS "MPI LINK flags path: " ${MPI_CXX_LINK_FLAGS})
    message(STATUS "MPI libraries: " ${MPI_CXX_LIBRARIES})
    include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${MPI_CXX_LIBRARIES})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}")
    find_program(OMPI_INFO
      NAMES ompi_info
      HINTS ${MPI_CXX_LIBRARIES}/../bin)
    if(OMPI_INFO)
      execute_process(COMMAND ${OMPI_INFO}
                      OUTPUT_VARIABLE _output)
      if(_output MATCHES "smcuda")
        message(STATUS "Found OpenMPI with CUDA support built.")
      else()
        message(WARNING "OpenMPI found, but it is not built with CUDA support.")
        set(CAFFE2_FORCE_FALLBACK_CUDA_MPI 1)
      endif()
    endif()
  else()
    message(WARNING "Not compiling with MPI. Suppress this warning with -DUSE_MPI=OFF")
    caffe2_update_option(USE_MPI OFF)
  endif()
endif()

# ---[ OpenMP
if(USE_OPENMP)
  # OpenMP support?
  set(WITH_OPENMP ON CACHE BOOL "OpenMP support if available?")

  # macOS + GCC
  if(APPLE AND CMAKE_COMPILER_IS_GNUCC)
    exec_program(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
    string(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
    message(STATUS "macOS Darwin version: ${DARWIN_VERSION}")
    if(DARWIN_VERSION GREATER 9)
      set(APPLE_OPENMP_SUCKS 1)
    endif(DARWIN_VERSION GREATER 9)
    execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion
      OUTPUT_VARIABLE GCC_VERSION)
    if(APPLE_OPENMP_SUCKS AND GCC_VERSION VERSION_LESS 4.6.2)
      message(WARNING "Disabling OpenMP (unstable with this version of GCC). "
        "Install GCC >= 4.6.2 or change your OS to enable OpenMP.")
      add_compile_options(-Wno-unknown-pragmas)
      set(WITH_OPENMP OFF CACHE BOOL "OpenMP support if available?" FORCE)
    endif()
  endif()

  if("${CMAKE_CXX_SIMULATE_ID}" STREQUAL "MSVC"
    AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    message(STATUS "Setting OpenMP flags for clang-cl")
    set(OpenMP_CXX_FLAGS "-Xclang -fopenmp")
    set(OpenMP_C_FLAGS "-Xclang -fopenmp")
    set(CHECKED_OPENMP ON CACHE BOOL "already checked for OpenMP")
    set(OPENMP_FOUND ON CACHE BOOL "OpenMP Support found")
    if(NOT MKL_FOUND)
      execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE clang_version_output)
      string(REGEX REPLACE ".*InstalledDir: ([^\n]+).*" "\\1" CLANG_BINDIR ${clang_version_output})

      get_filename_component(CLANG_ROOT ${CLANG_BINDIR} DIRECTORY)
      set(CLANG_OPENMP_LIBRARY "${CLANG_ROOT}/lib/libiomp5md.lib")

      if(NOT TARGET caffe2::openmp)
        add_library(caffe2::openmp INTERFACE IMPORTED)
      endif()

      set_property(
        TARGET caffe2::openmp PROPERTY INTERFACE_LINK_LIBRARIES
        ${CLANG_OPENMP_LIBRARY})

      list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS caffe2::openmp)
    endif()
  endif()

  if(WITH_OPENMP AND NOT CHECKED_OPENMP)
    find_package(OpenMP QUIET)
    set(CHECKED_OPENMP ON CACHE BOOL "already checked for OpenMP")

    # OPENMP_FOUND is not cached in FindOpenMP.cmake (all other variables are cached)
    # see https://github.com/Kitware/CMake/blob/master/Modules/FindOpenMP.cmake
    set(OPENMP_FOUND ${OPENMP_FOUND} CACHE BOOL "OpenMP Support found")
  endif()

  if(OPENMP_FOUND)
    message(STATUS "Adding OpenMP CXX_FLAGS: " ${OpenMP_CXX_FLAGS})
    if("${OpenMP_CXX_LIBRARIES}" STREQUAL "")
        message(STATUS "No OpenMP library needs to be linked against")
    else()
        message(STATUS "Will link against OpenMP libraries: ${OpenMP_CXX_LIBRARIES}")
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  else()
    message(WARNING "Not compiling with OpenMP. Suppress this warning with -DUSE_OPENMP=OFF")
    caffe2_update_option(USE_OPENMP OFF)
  endif()
endif()


# ---[ Android specific ones
if(ANDROID)
  list(APPEND Caffe2_DEPENDENCY_LIBS log)
endif()

# ---[ Kernel asserts
# Kernel asserts are enabled by default for CUDA and disabled for ROCm.
# For ROCm, it can be enabled by setting ROCM_FORCE_ENABLE_GPU_ASSERTS
if(USE_ROCM AND ROCM_FORCE_ENABLE_GPU_ASSERTS)
  message(STATUS "Forcefully enabling kernel asserts on ROCM")
elseif(USE_ROCM AND NOT ROCM_FORCE_ENABLE_GPU_ASSERTS)
  message(STATUS "Disabling kernel asserts for ROCm")
  caffe2_update_option(TORCH_DISABLE_GPU_ASSERTS ON)
endif()

# ---[ LLVM
if(USE_LLVM)
  message(STATUS "Looking for LLVM in ${USE_LLVM}")
  find_package(LLVM PATHS ${USE_LLVM} NO_DEFAULT_PATH)

  if(LLVM_FOUND)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

    include_directories(${LLVM_INCLUDE_DIRS})
    add_definitions(-DTORCH_ENABLE_LLVM)
  endif(LLVM_FOUND)
endif(USE_LLVM)

# ---[ cuDNN
if(USE_CUDNN)
  set(CUDNN_FRONTEND_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../third_party/cudnn_frontend/include)
  include_directories(${CUDNN_FRONTEND_INCLUDE_DIR})
endif()

# ---[ HIP
if(USE_ROCM)
  # This prevents linking in the libtinfo from /opt/conda/lib which conflicts with ROCm libtinfo.
  # Currently only active for Ubuntu 20.04 and greater versions.
  if(UNIX AND EXISTS "/etc/os-release")
    file(STRINGS /etc/os-release OS_RELEASE)
    string(REGEX REPLACE "NAME=\"([A-Za-z]+).*" "\\1" OS_DISTRO ${OS_RELEASE})
    string(REGEX REPLACE ".*VERSION_ID=\"([0-9\.]+).*" "\\1" OS_VERSION ${OS_RELEASE})
    if(OS_DISTRO STREQUAL "Ubuntu" AND OS_VERSION VERSION_GREATER_EQUAL "20.04")
      find_library(LIBTINFO_LOC tinfo NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH)
      if(LIBTINFO_LOC)
        get_filename_component(LIBTINFO_LOC_PARENT ${LIBTINFO_LOC} DIRECTORY)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link,${LIBTINFO_LOC_PARENT}")
      endif()
    endif()
  endif()

  include(${CMAKE_CURRENT_LIST_DIR}/public/LoadHIP.cmake)
  if(PYTORCH_FOUND_HIP)
    message(INFO "Compiling with HIP for AMD.")
    caffe2_update_option(USE_ROCM ON)

    if(USE_NCCL AND NOT USE_SYSTEM_NCCL)
      message(INFO "Forcing USE_SYSTEM_NCCL to ON since it's required by using RCCL")
      caffe2_update_option(USE_SYSTEM_NCCL ON)
    endif()

    list(APPEND HIP_CXX_FLAGS -fPIC)
    list(APPEND HIP_CXX_FLAGS -D__HIP_PLATFORM_HCC__=1)
    list(APPEND HIP_CXX_FLAGS -DCUDA_HAS_FP16=1)
    list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF_OPERATORS__=1)
    list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF_CONVERSIONS__=1)
    list(APPEND HIP_CXX_FLAGS -DTORCH_HIP_VERSION=${TORCH_HIP_VERSION})
    list(APPEND HIP_CXX_FLAGS -Wno-macro-redefined)
    list(APPEND HIP_CXX_FLAGS -Wno-inconsistent-missing-override)
    list(APPEND HIP_CXX_FLAGS -Wno-exceptions)
    list(APPEND HIP_CXX_FLAGS -Wno-shift-count-negative)
    list(APPEND HIP_CXX_FLAGS -Wno-shift-count-overflow)
    list(APPEND HIP_CXX_FLAGS -Wno-unused-command-line-argument)
    list(APPEND HIP_CXX_FLAGS -Wno-duplicate-decl-specifier)
    list(APPEND HIP_CXX_FLAGS -Wno-implicit-int-float-conversion)
    list(APPEND HIP_CXX_FLAGS -DCAFFE2_USE_MIOPEN)
    list(APPEND HIP_CXX_FLAGS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
    list(APPEND HIP_CXX_FLAGS -std=c++14)
    add_definitions(-DROCM_VERSION=${ROCM_VERSION_DEV_INT})
    add_definitions(-DTORCH_HIP_VERSION=${TORCH_HIP_VERSION})
    message("TORCH_HIP_VERSION=${TORCH_HIP_VERSION} is added as a compiler defines")

    if(CMAKE_BUILD_TYPE MATCHES Debug)
       list(APPEND HIP_CXX_FLAGS -g2)
       list(APPEND HIP_CXX_FLAGS -O0)
       list(APPEND HIP_HIPCC_FLAGS -fdebug-info-for-profiling)
    endif(CMAKE_BUILD_TYPE MATCHES Debug)

    set(HIP_CLANG_FLAGS ${HIP_CXX_FLAGS})
    # Ask hcc to generate device code during compilation so we can use
    # host linker to link.
    list(APPEND HIP_CLANG_FLAGS -fno-gpu-rdc)
    foreach(pytorch_rocm_arch ${PYTORCH_ROCM_ARCH})
      list(APPEND HIP_CLANG_FLAGS --amdgpu-target=${pytorch_rocm_arch})
    endforeach()

    set(Caffe2_HIP_INCLUDE
       $<INSTALL_INTERFACE:include> ${Caffe2_HIP_INCLUDE})
    # This is needed for library added by hip_add_library (same for hip_add_executable)
    hip_include_directories(${Caffe2_HIP_INCLUDE})

    set(Caffe2_PUBLIC_HIP_DEPENDENCY_LIBS
      ${PYTORCH_HIP_HCC_LIBRARIES} ${PYTORCH_MIOPEN_LIBRARIES} ${hipcub_LIBRARIES} ${ROCM_HIPRTC_LIB} ${ROCM_ROCTX_LIB})

    # Note [rocblas & rocfft cmake bug]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # TODO: There is a bug in rocblas's & rocfft's cmake files that exports the wrong targets name in ${rocblas_LIBRARIES}
    # If you get this wrong, you'll get a complaint like 'ld: cannot find -lrocblas-targets'
    if(ROCM_VERSION_DEV VERSION_GREATER_EQUAL "4.1.0")
      list(APPEND Caffe2_PUBLIC_HIP_DEPENDENCY_LIBS
        roc::rocblas hip::hipfft hip::hiprand roc::hipsparse)
    else()
      list(APPEND Caffe2_PUBLIC_HIP_DEPENDENCY_LIBS
        roc::rocblas roc::rocfft hip::hiprand roc::hipsparse)
    endif()
  else()
    caffe2_update_option(USE_ROCM OFF)
  endif()
endif()

# ---[ ROCm
if(USE_ROCM AND ROCM_VERSION_DEV VERSION_LESS "5.2.0")
  # We check again for USE_ROCM because it might have been set to OFF
  # in the if above
  include_directories(SYSTEM ${HIP_PATH}/include)
  include_directories(SYSTEM ${ROCBLAS_PATH}/include)
  if(ROCM_VERSION_DEV VERSION_GREATER_EQUAL "4.1.0")
    include_directories(SYSTEM ${HIPFFT_PATH}/include)
  else()
    include_directories(SYSTEM ${ROCFFT_PATH}/include)
  endif()
  include_directories(SYSTEM ${HIPSPARSE_PATH}/include)
  include_directories(SYSTEM ${HIPRAND_PATH}/include)
  include_directories(SYSTEM ${ROCRAND_PATH}/include)
  include_directories(SYSTEM ${THRUST_PATH})
endif()

# ---[ NCCL
if(USE_NCCL)
  if(NOT (USE_CUDA OR USE_ROCM))
    message(WARNING
        "Not using CUDA/ROCM, so disabling USE_NCCL. Suppress this warning with "
        "-DUSE_NCCL=OFF.")
    caffe2_update_option(USE_NCCL OFF)
  elseif(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(WARNING "NCCL is currently only supported under Linux.")
    caffe2_update_option(USE_NCCL OFF)
  elseif(USE_CUDA)
    include(${CMAKE_CURRENT_LIST_DIR}/External/nccl.cmake)
    list(APPEND Caffe2_CUDA_DEPENDENCY_LIBS __caffe2_nccl)
  elseif(USE_ROCM)
    include(${CMAKE_CURRENT_LIST_DIR}/External/rccl.cmake)
    list(APPEND Caffe2_CUDA_DEPENDENCY_LIBS __caffe2_nccl)
  endif()
endif()

# ---[ UCC
if(USE_UCC)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(WARNING "UCC is currently only supported under Linux.")
    caffe2_update_option(USE_UCC OFF)
  else()
    include(${CMAKE_CURRENT_LIST_DIR}/External/ucc.cmake)
  endif()
endif()

# ---[ CUB
if(USE_CUDA)
  find_package(CUB)
  if(CUB_FOUND)
    include_directories(SYSTEM ${CUB_INCLUDE_DIRS})
  else()
    include_directories(SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/cub)
  endif()
endif()

if(USE_DISTRIBUTED AND USE_TENSORPIPE)
  if(MSVC)
    message(WARNING "Tensorpipe cannot be used on Windows.")
  else()
    if(USE_CUDA)
      set(TP_USE_CUDA ON CACHE BOOL "" FORCE)
      set(TP_ENABLE_CUDA_IPC ON CACHE BOOL "" FORCE)
    endif()
    set(TP_BUILD_LIBUV ON CACHE BOOL "" FORCE)
    set(TP_STATIC_OR_SHARED STATIC CACHE STRING "" FORCE)

    add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/tensorpipe)


    list(APPEND Caffe2_DEPENDENCY_LIBS tensorpipe)
    if(USE_CUDA)
      list(APPEND Caffe2_CUDA_DEPENDENCY_LIBS tensorpipe_cuda)
    elseif(USE_ROCM)
      message(WARNING "TensorPipe doesn't yet support ROCm")
      # Not yet...
      # list(APPEND Caffe2_HIP_DEPENDENCY_LIBS tensorpipe_hip)
    endif()
  endif()
endif()

if(USE_GLOO)
  if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(WARNING "Gloo can only be used on 64-bit systems.")
    caffe2_update_option(USE_GLOO OFF)
  else()
    # Don't install gloo
    set(GLOO_INSTALL OFF CACHE BOOL "" FORCE)
    set(GLOO_STATIC_OR_SHARED STATIC CACHE STRING "" FORCE)

    # Temporarily override variables to avoid building Gloo tests/benchmarks
    set(__BUILD_TEST ${BUILD_TEST})
    set(__BUILD_BENCHMARK ${BUILD_BENCHMARK})
    set(BUILD_TEST OFF)
    set(BUILD_BENCHMARK OFF)
    if(USE_ROCM)
      set(ENV{GLOO_ROCM_ARCH} "${PYTORCH_ROCM_ARCH}")
    endif()
    if(NOT USE_SYSTEM_GLOO)
      if(USE_DISTRIBUED AND USE_TENSORPIPE)
        get_target_property(_include_dirs uv_a INCLUDE_DIRECTORIES)
        set_target_properties(uv_a PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_include_dirs}")
      endif()
      if(USE_NCCL AND NOT USE_SYSTEM_NCCL)
        # Tell Gloo build system to use bundled NCCL, see
        # https://github.com/facebookincubator/gloo/blob/950c0e23819779a9e0c70b861db4c52b31d1d1b2/cmake/Dependencies.cmake#L123
        set(NCCL_EXTERNAL ON)
      endif()
      set(GLOO_USE_CUDA_TOOLKIT ON CACHE BOOL "" FORCE)
      add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/gloo)
    else()
      add_library(gloo SHARED IMPORTED)
      find_library(GLOO_LIBRARY gloo)
      if(NOT GLOO_LIBRARY)
        message(FATAL_ERROR "Cannot find gloo")
      endif()
      message("Found gloo: ${GLOO_LIBRARY}")
      set_target_properties(gloo PROPERTIES IMPORTED_LOCATION ${GLOO_LIBRARY})
    endif()
    # Here is a little bit hacky. We have to put PROJECT_BINARY_DIR in front
    # of PROJECT_SOURCE_DIR with/without conda system. The reason is that
    # gloo generates a new config.h in the binary diretory.
    include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/gloo)
    include_directories(BEFORE SYSTEM ${PROJECT_BINARY_DIR}/third_party/gloo)
    set(BUILD_TEST ${__BUILD_TEST})
    set(BUILD_BENCHMARK ${__BUILD_BENCHMARK})

    # Add explicit dependency since NCCL is built from third_party.
    # Without dependency, make -jN with N>1 can fail if the NCCL build
    # hasn't finished when CUDA targets are linked.
    if(NOT USE_SYSTEM_NCCL AND USE_NCCL AND NOT USE_ROCM)
      add_dependencies(gloo_cuda nccl_external)
    endif()
    # Pick the right dependency depending on USE_CUDA
    list(APPEND Caffe2_DEPENDENCY_LIBS gloo)
    if(USE_CUDA)
      list(APPEND Caffe2_CUDA_DEPENDENCY_LIBS gloo_cuda)
    elseif(USE_ROCM)
      list(APPEND Caffe2_HIP_DEPENDENCY_LIBS gloo_hip)
    endif()
    add_compile_options(-DCAFFE2_USE_GLOO)
  endif()
endif()

# ---[ profiling
if(USE_PROF)
  find_package(htrace)
  if(htrace_FOUND)
    set(USE_PROF_HTRACE ON)
  else()
    message(WARNING "htrace not found. Caffe2 will build without htrace prof")
  endif()
endif()

if(USE_SNPE AND ANDROID)
  if(SNPE_LOCATION AND SNPE_HEADERS)
    message(STATUS "Using SNPE location specified by -DSNPE_LOCATION: " ${SNPE_LOCATION})
    message(STATUS "Using SNPE headers specified by -DSNPE_HEADERS: " ${SNPE_HEADERS})
    include_directories(SYSTEM ${SNPE_HEADERS})
    add_library(snpe SHARED IMPORTED)
    set_property(TARGET snpe PROPERTY IMPORTED_LOCATION ${SNPE_LOCATION})
    list(APPEND Caffe2_DEPENDENCY_LIBS snpe)
  else()
    caffe2_update_option(USE_SNPE OFF)
  endif()
endif()

if(USE_METAL)
  if(NOT IOS)
    message(WARNING "Metal is only used in ios builds.")
    caffe2_update_option(USE_METAL OFF)
  endif()
endif()

if(USE_NNAPI AND NOT ANDROID)
  message(WARNING "NNApi is only used in android builds.")
  caffe2_update_option(USE_NNAPI OFF)
endif()

if(NOT INTERN_BUILD_MOBILE AND BUILD_CAFFE2_OPS)
  if(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
    list(APPEND Caffe2_DEPENDENCY_LIBS aten_op_header_gen)
    if(USE_CUDA)
      list(APPEND Caffe2_CUDA_DEPENDENCY_LIBS aten_op_header_gen)
    endif()
    include_directories(${PROJECT_BINARY_DIR}/caffe2/contrib/aten)
  endif()
endif()

if(USE_ZSTD)
  list(APPEND Caffe2_DEPENDENCY_LIBS libzstd_static)
  include_directories(SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/zstd/lib)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/zstd/build/cmake)
  set_property(TARGET libzstd_static PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

# ---[ Onnx
if(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO AND NOT INTERN_DISABLE_ONNX)
  if(EXISTS "${CAFFE2_CUSTOM_PROTOC_EXECUTABLE}")
    set(ONNX_CUSTOM_PROTOC_EXECUTABLE ${CAFFE2_CUSTOM_PROTOC_EXECUTABLE})
  endif()
  set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS OFF)
  set(ONNX_USE_MSVC_STATIC_RUNTIME ${CAFFE2_USE_MSVC_STATIC_RUNTIME})
  set(ONNX_USE_LITE_PROTO ${CAFFE2_USE_LITE_PROTO})
  # If linking local protobuf, make sure ONNX has the same protobuf
  # patches as Caffe2 and Caffe proto. This forces some functions to
  # not be inline and instead route back to the statically-linked protobuf.
  if(CAFFE2_LINK_LOCAL_PROTOBUF)
    set(ONNX_PROTO_POST_BUILD_SCRIPT ${PROJECT_SOURCE_DIR}/cmake/ProtoBufPatch.cmake)
  endif()
  if(ONNX_ML)
    add_definitions(-DONNX_ML=1)
  endif()
  add_definitions(-DONNXIFI_ENABLE_EXT=1)
  # Add op schemas in "ai.onnx.pytorch" domain
  add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../caffe2/onnx/torch_ops")
  if(NOT USE_SYSTEM_ONNX)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/onnx EXCLUDE_FROM_ALL)
  endif()
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/foxi EXCLUDE_FROM_ALL)

  add_definitions(-DONNX_NAMESPACE=${ONNX_NAMESPACE})
  if(NOT USE_SYSTEM_ONNX)
    include_directories(${ONNX_INCLUDE_DIRS})
    # In mobile build we care about code size, and so we need drop
    # everything (e.g. checker) in onnx but the pb definition.
    if(ANDROID OR IOS)
      caffe2_interface_library(onnx_proto onnx_library)
    else()
      caffe2_interface_library(onnx onnx_library)
    endif()
    list(APPEND Caffe2_DEPENDENCY_WHOLE_LINK_LIBS onnx_library)
    # TODO: Delete this line once https://github.com/pytorch/pytorch/pull/55889 lands
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      target_compile_options(onnx PRIVATE -Wno-deprecated-declarations)
    endif()
  else()
    add_library(onnx SHARED IMPORTED)
    find_library(ONNX_LIBRARY onnx)
    if(NOT ONNX_LIBRARY)
      message(FATAL_ERROR "Cannot find onnx")
    endif()
    set_property(TARGET onnx PROPERTY IMPORTED_LOCATION ${ONNX_LIBRARY})
    add_library(onnx_proto SHARED IMPORTED)
    find_library(ONNX_PROTO_LIBRARY onnx_proto)
    if(NOT ONNX_PROTO_LIBRARY)
      message(FATAL_ERROR "Cannot find onnx")
    endif()
    set_property(TARGET onnx_proto PROPERTY IMPORTED_LOCATION ${ONNX_PROTO_LIBRARY})
    message("-- Found onnx: ${ONNX_LIBRARY} ${ONNX_PROTO_LIBRARY}")
    list(APPEND Caffe2_DEPENDENCY_LIBS onnx_proto onnx)
  endif()
  include_directories(${FOXI_INCLUDE_DIRS})
  list(APPEND Caffe2_DEPENDENCY_LIBS foxi_loader)
  # Recover the build shared libs option.
  set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS})
endif()

# --[ TensorRT integration with onnx-trt
function(add_onnx_tensorrt_subdir)
  # We pass the paths we found to onnx tensorrt.
  set(CUDNN_INCLUDE_DIR "${CUDNN_INCLUDE_PATH}")
  set(CUDNN_LIBRARY "${CUDNN_LIBRARY_PATH}")
  set(CMAKE_VERSION_ORIG "{CMAKE_VERSION}")
  if(FIND_CUDA_MODULE_DEPRECATED)
    # TODO: this WAR is for https://github.com/pytorch/pytorch/issues/18524
    set(CMAKE_VERSION "3.9.0")
  endif()
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/onnx-tensorrt EXCLUDE_FROM_ALL)
  set(CMAKE_VERSION "{CMAKE_VERSION_ORIG}")
endfunction()
if(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  if(USE_TENSORRT)
    add_onnx_tensorrt_subdir()
    include_directories("${CMAKE_CURRENT_LIST_DIR}/../third_party/onnx-tensorrt")
    caffe2_interface_library(nvonnxparser_static onnx_trt_library)
    list(APPEND Caffe2_DEPENDENCY_WHOLE_LINK_LIBS onnx_trt_library)
    set(CAFFE2_USE_TRT 1)
  endif()
endif()

# --[ ATen checks
set(USE_LAPACK 0)

# we need to build all targets to be linked with PIC
if(USE_KINETO AND INTERN_BUILD_MOBILE AND USE_LITE_INTERPRETER_PROFILER)
  set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
endif()

if(NOT INTERN_BUILD_MOBILE)
  set(TORCH_CUDA_ARCH_LIST $ENV{TORCH_CUDA_ARCH_LIST})
  string(APPEND CMAKE_CUDA_FLAGS " $ENV{TORCH_NVCC_FLAGS}")
  set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

  # Top-level build config
  ############################################
  # Flags
  # When using MSVC
  # Detect CUDA architecture and get best NVCC flags
  # finding cuda must be first because other things depend on the result
  #
  # NB: We MUST NOT run this find_package if NOT USE_CUDA is set, because upstream
  # FindCUDA has a bug where it will still attempt to make use of NOTFOUND
  # compiler variables to run various probe tests.  We could try to fix
  # this, but since FindCUDA upstream is subsumed by first-class support
  # for CUDA language, it seemed not worth fixing.

  if(MSVC)
    # we want to respect the standard, and we are bored of those **** .
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE=1)
    string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=/wd4819,/wd4503,/wd4190,/wd4244,/wd4251,/wd4275,/wd4522")
  endif()

  if(NOT MSVC)
    set(CMAKE_C_STANDARD 11 CACHE STRING "The C standard whose features are requested to build this target.")
  endif()

  string(APPEND CMAKE_CUDA_FLAGS " -Wno-deprecated-gpu-targets --expt-extended-lambda")

  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ standard whose features are requested to build this target.")
  endif()

  # use cub in a safe manner, see:
  # https://github.com/pytorch/pytorch/pull/55292
  if(NOT ${CUDA_VERSION} LESS 11.5)
    string(APPEND CMAKE_CUDA_FLAGS " -DCUB_WRAPPED_NAMESPACE=at_cuda_detail")
  endif()

  if(CUDA_HAS_FP16 OR NOT ${CUDA_VERSION} LESS 7.5)
    message(STATUS "Found CUDA with FP16 support, compiling with torch.cuda.HalfTensor")
    string(APPEND CMAKE_CUDA_FLAGS " -DCUDA_HAS_FP16=1"
                                   " -D__CUDA_NO_HALF_OPERATORS__"
                                   " -D__CUDA_NO_HALF_CONVERSIONS__"
                                   " -D__CUDA_NO_HALF2_OPERATORS__"
                                   " -D__CUDA_NO_BFLOAT16_CONVERSIONS__")
  else()
    message(STATUS "Could not find CUDA with FP16 support, compiling without torch.CudaHalfTensor")
  endif()

  string(APPEND CMAKE_C_FLAGS_RELEASE " -DNDEBUG")
  string(APPEND CMAKE_CXX_FLAGS_RELEASE " -DNDEBUG")
  if(NOT GENERATOR_IS_MULTI_CONFIG)
    if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
      message(STATUS "Adding -DNDEBUG to compile flags")
      string(APPEND CMAKE_C_FLAGS " -DNDEBUG")
      string(APPEND CMAKE_CXX_FLAGS " -DNDEBUG")
    else()
      message(STATUS "Removing -DNDEBUG from compile flags")
      string(REGEX REPLACE "[-/]DNDEBUG" "" CMAKE_C_FLAGS "" ${CMAKE_C_FLAGS})
      string(REGEX REPLACE "[-/]DNDEBUG" "" CMAKE_CXX_FLAGS "" ${CMAKE_CXX_FLAGS})
    endif()
  endif()
  string(REGEX REPLACE "[-/]DNDEBUG" "" CMAKE_C_FLAGS_DEBUG "" ${CMAKE_C_FLAGS_DEBUG})
  string(REGEX REPLACE "[-/]DNDEBUG" "" CMAKE_CXX_FLAGS_DEBUG "" ${CMAKE_CXX_FLAGS_DEBUG})

  set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)

  if(USE_MAGMA)
    find_package(MAGMA)
  endif()
  if((USE_CUDA OR USE_ROCM) AND MAGMA_FOUND)
    set(USE_MAGMA 1)
    message(STATUS "Compiling with MAGMA support")
    message(STATUS "MAGMA INCLUDE DIRECTORIES: ${MAGMA_INCLUDE_DIR}")
    message(STATUS "MAGMA LIBRARIES: ${MAGMA_LIBRARIES}")
    message(STATUS "MAGMA V2 check: ${MAGMA_V2}")
  elseif(USE_MAGMA)
    message(WARNING
      "Not compiling with MAGMA. Suppress this warning with "
      "-DUSE_MAGMA=OFF.")
    caffe2_update_option(USE_MAGMA OFF)
  else()
    message(STATUS "MAGMA not found. Compiling without MAGMA support")
    caffe2_update_option(USE_MAGMA OFF)
  endif()

  # ARM specific flags
  find_package(ARM)
  if(ASIMD_FOUND)
    message(STATUS "asimd/Neon found with compiler flag : -D__NEON__")
    add_compile_options(-D__NEON__)
  elseif(NEON_FOUND)
    message(STATUS "Neon found with compiler flag : -mfpu=neon -D__NEON__")
    add_compile_options(-mfpu=neon -D__NEON__)
  endif()
  if(CORTEXA8_FOUND)
    message(STATUS "Cortex-A8 Found with compiler flag : -mcpu=cortex-a8")
    add_compile_options(-mcpu=cortex-a8 -fprefetch-loop-arrays)
  endif()
  if(CORTEXA9_FOUND)
    message(STATUS "Cortex-A9 Found with compiler flag : -mcpu=cortex-a9")
    add_compile_options(-mcpu=cortex-a9)
  endif()

  if(WIN32 AND NOT CYGWIN)
    set(BLAS_INSTALL_LIBRARIES "OFF"
      CACHE BOOL "Copy the required BLAS DLLs into the TH install dirs")
  endif()

  find_package(LAPACK)
  if(LAPACK_FOUND)
    set(USE_LAPACK 1)
    list(APPEND Caffe2_PRIVATE_DEPENDENCY_LIBS ${LAPACK_LIBRARIES})
  endif()

  if(NOT USE_CUDA)
    message("disabling CUDA because NOT USE_CUDA is set")
    set(AT_CUDA_ENABLED 0)
  else()
    set(AT_CUDA_ENABLED 1)
  endif()

  if(NOT USE_CUDNN)
    message(STATUS "USE_CUDNN is set to 0. Compiling without cuDNN support")
    set(AT_CUDNN_ENABLED 0)
  elseif(NOT CUDNN_FOUND)
    message(WARNING "CuDNN not found. Compiling without CuDNN support")
    set(AT_CUDNN_ENABLED 0)
  else()
    include_directories(SYSTEM ${CUDNN_INCLUDE_PATH})
    set(AT_CUDNN_ENABLED 1)
  endif()

  if(NOT USE_ROCM)
    message("disabling ROCM because NOT USE_ROCM is set")
    message(STATUS "MIOpen not found. Compiling without MIOpen support")
    set(AT_ROCM_ENABLED 0)
  else()
    include_directories(BEFORE ${MIOPEN_INCLUDE_DIRS})
    set(AT_ROCM_ENABLED 1)
  endif()

  set(AT_MKLDNN_ENABLED 0)
  if(USE_MKLDNN)
    if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
      message(WARNING
        "x64 operating system is required for MKLDNN. "
        "Not compiling with MKLDNN. "
        "Turn this warning off by USE_MKLDNN=OFF.")
      set(USE_MKLDNN OFF)
    endif()
  endif()
  if(USE_MKLDNN)
    include(${CMAKE_CURRENT_LIST_DIR}/public/mkldnn.cmake)
    if(MKLDNN_FOUND)
      set(AT_MKLDNN_ENABLED 1)
      include_directories(AFTER SYSTEM ${MKLDNN_INCLUDE_DIR})
      if(BUILD_CAFFE2_OPS)
        list(APPEND Caffe2_DEPENDENCY_LIBS caffe2::mkldnn)
      endif(BUILD_CAFFE2_OPS)
    else()
      message(WARNING "MKLDNN could not be found.")
      caffe2_update_option(USE_MKLDNN OFF)
    endif()
  else()
    message("disabling MKLDNN because USE_MKLDNN is not set")
  endif()

  if(UNIX AND NOT APPLE)
     include(CheckLibraryExists)
     # https://github.com/libgit2/libgit2/issues/2128#issuecomment-35649830
     CHECK_LIBRARY_EXISTS(rt clock_gettime "time.h" NEED_LIBRT)
     if(NEED_LIBRT)
       list(APPEND Caffe2_DEPENDENCY_LIBS rt)
       set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} rt)
     endif(NEED_LIBRT)
  endif(UNIX AND NOT APPLE)

  if(UNIX)
    set(CMAKE_EXTRA_INCLUDE_FILES "sys/mman.h")
    CHECK_FUNCTION_EXISTS(mmap HAVE_MMAP)
    if(HAVE_MMAP)
      add_definitions(-DHAVE_MMAP=1)
    endif(HAVE_MMAP)
    # done for lseek: https://www.gnu.org/software/libc/manual/html_node/File-Position-Primitive.html
    add_definitions(-D_FILE_OFFSET_BITS=64)
    CHECK_FUNCTION_EXISTS(shm_open HAVE_SHM_OPEN)
    if(HAVE_SHM_OPEN)
      add_definitions(-DHAVE_SHM_OPEN=1)
    endif(HAVE_SHM_OPEN)
    CHECK_FUNCTION_EXISTS(shm_unlink HAVE_SHM_UNLINK)
    if(HAVE_SHM_UNLINK)
      add_definitions(-DHAVE_SHM_UNLINK=1)
    endif(HAVE_SHM_UNLINK)
    CHECK_FUNCTION_EXISTS(malloc_usable_size HAVE_MALLOC_USABLE_SIZE)
    if(HAVE_MALLOC_USABLE_SIZE)
      add_definitions(-DHAVE_MALLOC_USABLE_SIZE=1)
    endif(HAVE_MALLOC_USABLE_SIZE)
  endif(UNIX)

  add_definitions(-DUSE_EXTERNAL_MZCRC)
  add_definitions(-DMINIZ_DISABLE_ZIP_READER_CRC32_CHECKS)

  # Is __thread supported?
  if(NOT MSVC)
    CHECK_C_SOURCE_COMPILES("static __thread int x = 1; int main() { return x; }" C_HAS_THREAD)
  else(NOT MSVC)
    CHECK_C_SOURCE_COMPILES("static __declspec( thread ) int x = 1; int main() { return x; }" C_HAS_THREAD)
  endif(NOT MSVC)
  if(NOT C_HAS_THREAD)
    message(STATUS "Warning: __thread is not supported, generating thread-unsafe code")
  else(NOT C_HAS_THREAD)
    add_compile_options(-DTH_HAVE_THREAD)
  endif(NOT C_HAS_THREAD)
endif()

#
# End ATen checks
#
set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/fmt)

# Disable compiler feature checks for `fmt`.
#
# CMake compiles a little program to check compiler features. Some of our build
# configurations (notably the mobile build analyzer) will populate
# CMAKE_CXX_FLAGS in ways that break feature checks. Since we already know
# `fmt` is compatible with a superset of the compilers that PyTorch is, it
# shouldn't be too bad to just disable the checks.
set_target_properties(fmt-header-only PROPERTIES INTERFACE_COMPILE_FEATURES "")

list(APPEND Caffe2_DEPENDENCY_LIBS fmt::fmt-header-only)
set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libs" FORCE)

# ---[ Kineto
# edge profiler depends on KinetoProfiler but it only does cpu
# profiling. Thus we dont need USE_CUDA/USE_ROCM
if(USE_KINETO AND INTERN_BUILD_MOBILE AND NOT (BUILD_LITE_INTERPRETER AND USE_LITE_INTERPRETER_PROFILER))
  message(STATUS "Not using libkineto in a mobile build.")
  set(USE_KINETO OFF)
endif()

if(USE_KINETO AND INTERN_BUILD_MOBILE AND USE_LITE_INTERPRETER_PROFILER AND (USE_CUDA OR USE_ROCM))
  message(FATAL_ERROR "Mobile build with profiler does not support CUDA or ROCM")
endif()

if(USE_KINETO)
  if((NOT USE_CUDA) OR MSVC)
    set(LIBKINETO_NOCUPTI ON CACHE STRING "" FORCE)
  else()
    set(LIBKINETO_NOCUPTI OFF CACHE STRING "")
    message(STATUS "Using Kineto with CUPTI support")
  endif()

  if(NOT USE_ROCM)
    set(LIBKINETO_NOROCTRACER ON CACHE STRING "" FORCE)
  else()
    set(LIBKINETO_NOROCTRACER OFF CACHE STRING "")
    message(STATUS "Using Kineto with Roctracer support")
  endif()

  set(CAFFE2_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third_party" CACHE STRING "")
  set(KINETO_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/kineto/libkineto" CACHE STRING "")
  set(KINETO_BUILD_TESTS OFF CACHE BOOL "")
  set(KINETO_LIBRARY_TYPE "static" CACHE STRING "")
  set(LIBKINETO_DYNAMIC_CUPTI "${USE_CUPTI_SO}" CACHE STRING "" FORCE)

  if(NOT TARGET kineto)
    add_subdirectory("${KINETO_SOURCE_DIR}")
    set_property(TARGET kineto PROPERTY POSITION_INDEPENDENT_CODE ON)
  endif()
  list(APPEND Caffe2_DEPENDENCY_LIBS kineto)
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_KINETO")
  if(LIBKINETO_NOCUPTI)
    string(APPEND CMAKE_CXX_FLAGS " -DLIBKINETO_NOCUPTI")
  endif()

  if(LIBKINETO_NOCUPTI AND LIBKINETO_NOROCTRACER)
    message(STATUS "Configured Kineto (CPU):")
  else()
    message(STATUS "Configured Kineto:")
  endif()

  message(STATUS "  KINETO_SOURCE_DIR = ${KINETO_SOURCE_DIR}")
  message(STATUS "  KINETO_BUILD_TESTS = ${KINETO_BUILD_TESTS}")
  message(STATUS "  KINETO_LIBRARY_TYPE = ${KINETO_LIBRARY_TYPE}")
  if(USE_CUDA OR NOT LIBKINETO_NOCUPTI)
    message(STATUS "  LIBKINETO_NOCUPTI = ${LIBKINETO_NOCUPTI}")
    message(STATUS "  LIBKINETO_DYNAMIC_CUPTI = ${LIBKINETO_DYNAMIC_CUPTI}")
    message(STATUS "  CUDAToolkit_CUPTI_INCLUDE_DIR = ${CUDAToolkit_CUPTI_INCLUDE_DIR}")
    message(STATUS "  CUDA_cupti_LIBRARY = ${CUDA_cupti_LIBRARY}")
  endif()
  if(USE_ROCM OR NOT LIBKINETO_NOROCTRACER)
    message(STATUS "  LIBKINETO_NOROCTRACER = ${LIBKINETO_NOROCTRACER}")
  endif()
endif()

# Include google/FlatBuffers
include(${CMAKE_CURRENT_LIST_DIR}/FlatBuffers.cmake)
