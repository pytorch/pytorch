cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
#cmake_policy(SET CMP0022 NEW)
#cmake_policy(SET CMP0023 NEW)

# Use compiler ID "AppleClang" instead of "Clang" for XCode.
# Not setting this sometimes makes XCode C compiler gets detected as "Clang",
# even when the C++ one is detected as "AppleClang".
cmake_policy(SET CMP0010 NEW)
cmake_policy(SET CMP0025 NEW)

# Suppress warning flags in default MSVC configuration.  It's not
# mandatory that we do this (and we don't if cmake is old), but it's
# nice when it's possible, and it's possible on our Windows configs.
if(NOT CMAKE_VERSION VERSION_LESS 3.15.0)
  cmake_policy(SET CMP0092 NEW)
endif()

# ---[ Project and semantic versioning.
project(Caffe2 CXX C)

if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(LINUX TRUE)
else()
  set(LINUX FALSE)
endif()

set(CMAKE_INSTALL_MESSAGE NEVER)

set(CMAKE_CXX_STANDARD 11)
if (NOT MSVC)
  set(CMAKE_C_STANDARD 11)
endif()
if (DEFINED GLIBCXX_USE_CXX11_ABI)
  if (${GLIBCXX_USE_CXX11_ABI} EQUAL 1)
    set(CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
  endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# One variable that determines whether the current cmake process is being run
# with the main Caffe2 library. This is useful for building modules - if
# modules are built with the main Caffe2 library then one does not need to do
# find caffe2 in the cmake script. One can usually guard it in some way like
#    if (NOT CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
#      find_package(Caffe2 REQUIRED)
#    endif()
set(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO ON)

if(NOT DEFINED BLAS_SET_BY_USER)
  if(DEFINED BLAS)
    set(BLAS_SET_BY_USER TRUE)
  else()
    message(STATUS "Not forcing any particular BLAS to be found")
    set(BLAS_SET_BY_USER FALSE)
  endif()
  set(BLAS_SET_BY_USER ${BLAS_SET_BY_USER} CACHE STRING "Marks whether BLAS was manually set by user or auto-detected")
endif()

# Apple specific
if(APPLE)
  # These lines are an attempt to make find_package(cuda) pick up
  # libcuda.dylib, and not cuda.framework.  It doesn't work all
  # the time, but it seems to help for some users.
  # TODO: replace this with a more robust fix
  set(CMAKE_FIND_FRAMEWORK LAST)
  set(CMAKE_FIND_APPBUNDLE LAST)

  # Get clang version on macOS
  EXECUTE_PROCESS( COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE clang_full_version_string )
  string(REGEX REPLACE "Apple LLVM version ([0-9]+\\.[0-9]+).*" "\\1" CLANG_VERSION_STRING ${clang_full_version_string})
  MESSAGE( STATUS "CLANG_VERSION_STRING:         " ${CLANG_VERSION_STRING} )


  # RPATH stuff
  set(CMAKE_MACOSX_RPATH ON)
endif()

if (${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "(x86_64|i[3-6]+86)")
  set(CPU_INTEL ON)
else ()
  set(CPU_INTEL OFF)
endif ()

# For non-supported platforms, turn USE_DISTRIBUTED off by default.
# It is not tested and likely won't work without additional changes.
if(NOT LINUX)
  set(USE_DISTRIBUTED OFF CACHE STRING "Use distributed")
  # On macOS, if USE_DISTRIBUTED is enabled (specified by the user),
  # then make Gloo build with the libuv transport.
  if(APPLE AND USE_DISTRIBUTED)
    set(USE_LIBUV ON CACHE STRING "")
  endif()
endif()

# ---[ Options.
# Note to developers: if you add an option below, make sure you also add it to
# cmake/Summary.cmake so that the summary prints out the option values.
include(CMakeDependentOption)
option(ATEN_NO_TEST "Do not build ATen test binaries" OFF)
option(BUILD_BINARY "Build C++ binaries" OFF)
option(BUILD_DOCS "Build Caffe2 documentation" OFF)
option(BUILD_CUSTOM_PROTOBUF "Build and use Caffe2's own protobuf under third_party" ON)
option(BUILD_PYTHON "Build Python binaries" ON)
option(BUILD_CAFFE2_OPS "Build Caffe2 operators" ON)
option(BUILD_SHARED_LIBS "Build libcaffe2.so" ON)
option(BUILD_CAFFE2_MOBILE "Build libcaffe2 for mobile (deprecating)" ON)
option(BUILD_NAMEDTENSOR "Experimental: compile with namedtensor support" OFF)
option(USE_STATIC_DISPATCH "Use static dispatch for ATen operators" OFF)
cmake_dependent_option(
    CAFFE2_LINK_LOCAL_PROTOBUF "If set, build protobuf inside libcaffe2.so." ON
    "BUILD_SHARED_LIBS AND BUILD_CUSTOM_PROTOBUF" OFF)
cmake_dependent_option(
    CAFFE2_USE_MSVC_STATIC_RUNTIME "Using MSVC static runtime libraries" ON
    "NOT BUILD_SHARED_LIBS" OFF)
option(BUILD_TEST "Build C++ test binaries (need gtest and gbenchmark)" OFF)
cmake_dependent_option(
    INSTALL_TEST "Install test binaries if BUILD_TEST is on" ON
    "BUILD_TEST" OFF)
option(COLORIZE_OUTPUT "Colorize output during compilation" ON)
option(USE_ASAN "Use Address Sanitizer" OFF)
option(USE_CUDA "Use CUDA" ON)
option(USE_ROCM "Use ROCm" ON)
option(CAFFE2_STATIC_LINK_CUDA "Statically link CUDA libraries" OFF)
cmake_dependent_option(
    USE_CUDNN "Use cuDNN" ON
    "USE_CUDA" OFF)
cmake_dependent_option(
    USE_STATIC_CUDNN "Use cuDNN static libraries" OFF
    "USE_CUDNN" OFF)
option(USE_FBGEMM "Use FBGEMM (quantized 8-bit server operators)" ON)
option(USE_FFMPEG "Use ffmpeg" OFF)
option(USE_GFLAGS "Use GFLAGS" OFF)
option(USE_GLOG "Use GLOG" OFF)
option(USE_LEVELDB "Use LEVELDB" OFF)
option(USE_LITE_PROTO "Use lite protobuf instead of full." OFF)
option(USE_LMDB "Use LMDB" OFF)
option(USE_METAL "Use Metal for iOS build" ON)
option(USE_NATIVE_ARCH "Use -march=native" OFF)
cmake_dependent_option(
    USE_NCCL "Use NCCL" ON
    "USE_CUDA;UNIX;NOT APPLE" OFF)
cmake_dependent_option(
    USE_STATIC_NCCL "Use static NCCL" OFF
    "USE_NCCL" OFF)
cmake_dependent_option(
    USE_SYSTEM_NCCL "Use system-wide NCCL" OFF
    "USE_NCCL" OFF)
option(USE_NNAPI "Use NNAPI" OFF)
option(USE_NNPACK "Use NNPACK" ON)
cmake_dependent_option(
    USE_NUMA "Use NUMA. Only available on Linux." ON
    "LINUX" OFF)
cmake_dependent_option(
    USE_NVRTC "Use NVRTC. Only available if USE_CUDA is on." OFF
    "USE_CUDA" OFF)
option(USE_NUMPY "Use NumPy" ON)
option(USE_OBSERVERS "Use observers module." OFF)
option(USE_OPENCL "Use OpenCL" OFF)
option(USE_OPENCV "Use OpenCV" OFF)
option(USE_OPENMP "Use OpenMP for parallel code" ON)
option(USE_PROF "Use profiling" OFF)
option(USE_QNNPACK "Use QNNPACK (quantized 8-bit operators)" ON)
option(USE_PYTORCH_QNNPACK "Use ATen/QNNPACK (quantized 8-bit operators)" ON)
option(USE_REDIS "Use Redis" OFF)
option(USE_ROCKSDB "Use RocksDB" OFF)
option(USE_SNPE "Use Qualcomm's SNPE library" OFF)
option(USE_SYSTEM_EIGEN_INSTALL
    "Use system Eigen instead of the one under third_party" OFF)
option(USE_TENSORRT "Using Nvidia TensorRT library" OFF)
option(USE_ZMQ "Use ZMQ" OFF)
option(USE_ZSTD "Use ZSTD" OFF)
cmake_dependent_option(
  USE_MKLDNN "Use MKLDNN. Only available on x86 and x86_64." ON
  "CPU_INTEL" OFF)
set(MKLDNN_ENABLE_CONCURRENT_EXEC ${USE_MKLDNN})
cmake_dependent_option(
    USE_MKLDNN_CBLAS "Use CBLAS in MKLDNN" OFF
    "USE_MKLDNN" OFF)
option(USE_DISTRIBUTED "Use distributed" ON)
cmake_dependent_option(
    USE_MPI "Use MPI for Caffe2. Only available if USE_DISTRIBUTED is on." ON
    "USE_DISTRIBUTED" OFF)
cmake_dependent_option(
    USE_GLOO "Use Gloo. Only available if USE_DISTRIBUTED is on." ON
    "USE_DISTRIBUTED" OFF)
option(USE_TBB "Use TBB" OFF)

# Used when building Caffe2 through setup.py
option(BUILDING_WITH_TORCH_LIBS "Tell cmake if Caffe2 is being built alongside torch libs" ON)

# /Z7 override option
# When generating debug symbols, CMake default to use the flag /Zi.
# However, it is not compatible with sccache. So we rewrite it off.
# But some users don't use sccache; this override is for them.
cmake_dependent_option(
  MSVC_Z7_OVERRIDE "Work around sccache bug by replacing /Zi and /ZI with /Z7 when using MSVC (if you are not using sccache, you can turn this OFF)" ON
  "MSVC" OFF)

set(ONNX_NAMESPACE "onnx_torch" CACHE STRING "A namespace for ONNX; needed to build with other frameworks that share ONNX.")

# This is a fix for a rare build issue on Ubuntu:
# symbol lookup error: miniconda3/envs/pytorch-py3.7/lib/libmkl_intel_lp64.so: undefined symbol: mkl_blas_dsyrk
# https://software.intel.com/en-us/articles/symbol-lookup-error-when-linking-intel-mkl-with-gcc-on-ubuntu
if(LINUX)
  set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")
endif()

if (MSVC)
  foreach(flag_var
      CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
      CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO
      CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    # Replace /Zi and /ZI with /Z7
    if(MSVC_Z7_OVERRIDE)
      if(${flag_var} MATCHES "/Z[iI]")
        string(REGEX REPLACE "/Z[iI]" "/Z7" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/Z[iI]")
    endif(MSVC_Z7_OVERRIDE)
    # Turn off warnings on Windows.  In an ideal world we'd be warning
    # clean on Windows too, but this is too much work for our
    # non-Windows developers.
    #
    # NB: Technically, this is not necessary if CMP0092 was applied
    # properly, but only cmake >= 3.15 has this policy, so we nail
    # it one more time just be safe.
    #
    # NB2: This is NOT enough to prevent warnings from nvcc on MSVC.  At the
    # moment only CMP0092 is enough to prevent those warnings too.
    string(REPLACE "/W3" "" ${flag_var} "${${flag_var}}")
    # Suppress EHs is overridden by EHa warning
    string(REPLACE "/EHsc" "" ${flag_var} "${${flag_var}}")

    # Turn off warnings (Windows build is currently is extremely warning
    # unclean and the warnings aren't telling us anything useful.)
    #
    # Turn on EHa; I'm not altogether clear why we use the asynchronous
    # exception handling model, but someone added it at some point, so
    # keep using it.
    string(APPEND ${flag_var} " /w /EHa")

    if (${CAFFE2_USE_MSVC_STATIC_RUNTIME})
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    else()
      if(${flag_var} MATCHES "/MT")
        string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
      endif()
    endif()

    # /bigobj increases number of sections in .obj file, which is needed to link
    # against libaries in Python 2.7 under Windows
    set(${flag_var} "${${flag_var}} /MP /bigobj")
  endforeach(flag_var)

  foreach(flag_var
      CMAKE_SHARED_LINKER_FLAGS_DEBUG CMAKE_STATIC_LINKER_FLAGS_DEBUG
      CMAKE_EXE_LINKER_FLAGS_DEBUG CMAKE_MODULE_LINKER_FLAGS_DEBUG)
    # Switch off incremental linking in debug builds
    if(${flag_var} MATCHES "/INCREMENTAL" AND NOT ${flag_var} MATCHES "/INCREMENTAL:NO")
      string(REGEX REPLACE "/INCREMENTAL" "/INCREMENTAL:NO" ${flag_var} "${${flag_var}}")
    endif()
  endforeach(flag_var)

  foreach(flag_var
      CMAKE_SHARED_LINKER_FLAGS CMAKE_STATIC_LINKER_FLAGS
      CMAKE_EXE_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS)
    string(APPEND ${flag_var} " /ignore:4049 /ignore:4217")
  endforeach(flag_var)

  # Try harder
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler /w -w")
endif(MSVC)

# Set INTERN_BUILD_MOBILE for all mobile builds. Components that are not
# applicable to mobile are disabled by this variable.
if (ANDROID OR IOS)
  set(INTERN_BUILD_MOBILE ON)
endif()

# Setting `PYTORCH_BUILD_MOBILE` environment variable can force it to do mobile
# build with host toolchain.
if (DEFINED ENV{PYTORCH_BUILD_MOBILE})
  set(INTERN_BUILD_MOBILE ON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DC10_MOBILE")
endif()

# INTERN_BUILD_ATEN_OPS is used to control whether to build ATen/TH operators.
# It's disabled for caffe2 mobile library.
if (INTERN_BUILD_MOBILE AND BUILD_CAFFE2_MOBILE)
  set(INTERN_BUILD_ATEN_OPS OFF)
else()
  set(INTERN_BUILD_ATEN_OPS ON)
endif()

# BUILD_CAFFE2_MOBILE is the master switch to choose between libcaffe2 v.s. libtorch mobile build.
# When it's enabled it builds original libcaffe2 mobile library without ATen/TH ops nor TorchScript support;
# When it's disabled it builds libtorch mobile library, which contains ATen/TH ops and native support for
# TorchScript model, but doesn't contain not-yet-unified caffe2 ops;
if (INTERN_BUILD_MOBILE AND NOT BUILD_CAFFE2_MOBILE)
  if (NOT BUILD_SHARED_LIBS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_EXPORT")
  endif()
  set(BUILD_PYTHON OFF)
  set(BUILD_CAFFE2_OPS OFF)
  set(USE_DISTRIBUTED OFF)
  set(FEATURE_TORCH_MOBILE ON)
  set(NO_API ON)
  set(USE_FBGEMM OFF)
  set(USE_PYTORCH_QNNPACK ON)
  set(USE_QNNPACK OFF)
  set(USE_STATIC_DISPATCH ON)
  set(INTERN_DISABLE_ONNX ON)
  set(INTERN_DISABLE_AUTOGRAD ON)
  set(INTERN_USE_EIGEN_BLAS ON)
endif()

# ---[ Utils
# TODO: merge the following 3 files into cmake/public/utils.cmake.
include(cmake/Utils.cmake)
include(cmake/public/utils.cmake)

# ---[ Version numbers for generated libraries
set(TORCH_DEFAULT_VERSION "1.1.0")
set(TORCH_BUILD_VERSION "${TORCH_DEFAULT_VERSION}" CACHE STRING "Torch build version")
if (DEFINED ENV{PYTORCH_BUILD_VERSION})
  set(TORCH_BUILD_VERSION "$ENV{PYTORCH_BUILD_VERSION}"
    CACHE STRING "Torch build version" FORCE)
endif()
if (NOT TORCH_BUILD_VERSION)
  # An empty string was specified so force version to the default
  set(TORCH_BUILD_VERSION "${TORCH_DEFAULT_VERSION}"
    CACHE STRING "Torch build version" FORCE)
endif()
caffe2_parse_version_str(TORCH ${TORCH_BUILD_VERSION})
caffe2_parse_version_str(CAFFE2 ${TORCH_BUILD_VERSION})

# ---[ CMake scripts + modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# ---[ CMake build directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

enable_testing()

# ---[ Build variables set within the cmake tree
include(cmake/BuildVariables.cmake)
set(CAFFE2_WHITELIST "" CACHE STRING "A whitelist file of files that one should build.")

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type not set - defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build from: Debug Release RelWithDebInfo MinSizeRel Coverage." FORCE)
endif()

# ---[ Misc checks to cope with various compiler modes
include(cmake/MiscCheck.cmake)

# External projects
include(ExternalProject)

# ---[ Dependencies
# ---[ FBGEMM doesn't work on x86 32bit and CMAKE_SYSTEM_PROCESSOR thinks its 64bit
if(USE_FBGEMM AND ((CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" AND CMAKE_SIZEOF_VOID_P EQUAL 4) OR CMAKE_SYSTEM_PROCESSOR STREQUAL "x86"))
  set(USE_FBGEMM OFF)
endif()

include(cmake/Dependencies.cmake)

if(USE_FBGEMM)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_FBGEMM")
endif()

if(BUILD_NAMEDTENSOR)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBUILD_NAMEDTENSOR")
endif()

if(USE_QNNPACK)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_QNNPACK")
endif()

if(USE_PYTORCH_QNNPACK)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_PYTORCH_QNNPACK")
endif()

# ---[ Whitelist file if whitelist is specified
include(cmake/Whitelist.cmake)

# ---[ Set link flag, handle additional deps for gcc 4.8 and above
if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.8.0 AND NOT ANDROID)
  message(STATUS "GCC ${CMAKE_CXX_COMPILER_VERSION}: Adding gcc and gcc_s libs to link line")
  list(APPEND Caffe2_DEPENDENCY_LIBS gcc_s gcc)
endif()

# ---[ Build flags
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fPIC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing")
  # Eigen fails to build with some versions, so convert this to a warning
  # Details at http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1459
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-field-initializers")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-type-limits")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-array-bounds")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-overflow")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-aliasing")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations")
  if (CMAKE_COMPILER_IS_GNUCXX AND NOT (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0.0))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-stringop-overflow")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=pedantic")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=redundant-decls")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=old-style-cast")
  # These flags are not available in GCC-4.8.5. Set only when using clang.
  # Compared against https://gcc.gnu.org/onlinedocs/gcc-4.8.5/gcc/Option-Summary.html
  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-invalid-partial-specialization")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-typedef-redefinition")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-warning-option")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-private-field")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-inconsistent-missing-override")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-aligned-allocation-unavailable")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++14-extensions")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-constexpr-not-const")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-braces")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Qunused-arguments")
    if (${COLORIZE_OUTPUT})
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics")
    endif()
  endif()
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)
    if (${COLORIZE_OUTPUT})
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
    endif()
  endif()
  if ((APPLE AND (NOT ("${CLANG_VERSION_STRING}" VERSION_LESS "9.0")))
    OR (CMAKE_COMPILER_IS_GNUCXX
    AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0 AND NOT APPLE)))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -faligned-new")
  endif()
  if (WERROR)
    check_cxx_compiler_flag("-Werror" COMPILER_SUPPORT_WERROR)
    if (NOT COMPILER_SUPPORT_WERROR)
      set(WERROR FALSE)
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    endif()
  endif(WERROR)
  if (NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
  endif()
  set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -O0")
  set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_STATIC_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -O0")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-math-errno")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-trapping-math")
endif()

if (USE_ASAN)
    set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
    set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_STATIC_LINKER_FLAGS_DEBUG} -fsanitize=address")
endif()

if (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-private-field")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-braces")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++14-extensions")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-constexpr-not-const")
endif()

if (EMSCRIPTEN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-implicit-function-declaration -DEMSCRIPTEN -s DISABLE_EXCEPTION_CATCHING=0")
endif()

if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0.0)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-stringop-overflow")
endif()

if(ANDROID)
  if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s")
  else()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s")
  endif()
endif()

if(NOT APPLE AND UNIX)
  list(APPEND Caffe2_DEPENDENCY_LIBS dl)
endif()

# Prefix path to Caffe2 headers.
# If a directory containing installed Caffe2 headers was inadvertently
# added to the list of include directories, prefixing
# PROJECT_SOURCE_DIR means this source tree always takes precedence.
include_directories(BEFORE ${PROJECT_SOURCE_DIR})

# Prefix path to generated Caffe2 headers.
# These need to take precedence over their empty counterparts located
# in PROJECT_SOURCE_DIR.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

include_directories(BEFORE ${PROJECT_SOURCE_DIR}/aten/src/)
include_directories(BEFORE ${PROJECT_BINARY_DIR}/aten/src/)

# ---[ Main build
add_subdirectory(c10)
add_subdirectory(caffe2)

# --[ Documentation
if(BUILD_DOCS)
  # check if Doxygen is installed
  find_package(Doxygen)
  if (DOXYGEN_FOUND)
    message("Generating documentation")

    set(DOXYGEN_C_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/caffe2/.Doxyfile-c)
    set(DOXYGEN_C_OUT ${CMAKE_CURRENT_SOURCE_DIR}/docs/caffe2/Doxyfile-c)
    set(DOXYGEN_P_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/caffe2/.Doxyfile-python)
    set(DOXYGEN_P_OUT ${CMAKE_CURRENT_SOURCE_DIR}/docs/caffe2/Doxyfile-python)

    if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/docs)
      file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/docs)
    endif()

    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs)
    configure_file(${DOXYGEN_C_IN} ${DOXYGEN_C_OUT} @ONLY)
    configure_file(${DOXYGEN_P_IN} ${DOXYGEN_P_OUT} @ONLY)

    add_custom_target(doc_doxygen_c ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_C_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating C++ API documentation with Doxygen"
        VERBATIM)

    add_custom_target(doc_doxygen_python ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_P_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating Python API documentation with Doxygen"
        VERBATIM)
  else()
    message(FATAL_ERROR "Doxygen needs to be installed to generate the documentation")
  endif()
endif()

# ---[ CMake related files
# Uninistall option.
if(NOT TARGET caffe2_uninstall)
  configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in
      ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
      IMMEDIATE @ONLY)

  add_custom_target(caffe2_uninstall
      COMMAND ${CMAKE_COMMAND} -P
      ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

# ---[ Make configuration files for cmake to allow dependent libraries
# easier access to Caffe2.

if ((NOT USE_GLOG) OR (NOT USE_GFLAGS) OR BUILD_CUSTOM_PROTOBUF)
  message(WARNING
      "Generated cmake files are only fully tested if one builds "
      "with system glog, gflags, and protobuf. Other settings may "
      "generate files that are not well tested.")
endif()

if (USE_CUDA OR USE_ROCM)
  # TODO: check if we should include other cuda dependency libraries
  # to the interface as well.

endif()

# Note(jiayq): when building static libraries, all PRIVATE dependencies
# will also become interface libraries, and as a result if there are any
# dependency libraries that are not exported, the following install export
# script will fail. As a result, we will only provide the targets cmake
# files for shared lib installation. For more info, read:
# https://cmake.org/pipermail/cmake/2016-May/063400.html
if (BUILD_SHARED_LIBS)
  configure_file(
      ${PROJECT_SOURCE_DIR}/cmake/Caffe2ConfigVersion.cmake.in
      ${PROJECT_BINARY_DIR}/Caffe2ConfigVersion.cmake
      @ONLY)
  configure_file(
      ${PROJECT_SOURCE_DIR}/cmake/Caffe2Config.cmake.in
      ${PROJECT_BINARY_DIR}/Caffe2Config.cmake
      @ONLY)
  install(FILES
      ${PROJECT_BINARY_DIR}/Caffe2ConfigVersion.cmake
      ${PROJECT_BINARY_DIR}/Caffe2Config.cmake
      DESTINATION share/cmake/Caffe2
      COMPONENT dev)
  install(FILES
      ${PROJECT_SOURCE_DIR}/cmake/public/cuda.cmake
      ${PROJECT_SOURCE_DIR}/cmake/public/glog.cmake
      ${PROJECT_SOURCE_DIR}/cmake/public/gflags.cmake
      ${PROJECT_SOURCE_DIR}/cmake/public/mkl.cmake
      ${PROJECT_SOURCE_DIR}/cmake/public/mkldnn.cmake
      ${PROJECT_SOURCE_DIR}/cmake/public/protobuf.cmake
      ${PROJECT_SOURCE_DIR}/cmake/public/threads.cmake
      ${PROJECT_SOURCE_DIR}/cmake/public/utils.cmake
      DESTINATION share/cmake/Caffe2/public
      COMPONENT dev)
  install(DIRECTORY
      ${PROJECT_SOURCE_DIR}/cmake/Modules_CUDA_fix
      DESTINATION share/cmake/Caffe2/
      COMPONENT dev)

  install(EXPORT Caffe2Targets DESTINATION share/cmake/Caffe2
      FILE Caffe2Targets.cmake
      COMPONENT dev)
else()
  message(WARNING
      "Generated cmake files are only available when building "
      "shared libs.")
endif()

# ---[ Modules
add_subdirectory(modules)

# ---[ Binaries
# Binaries will be built after the Caffe2 main libraries and the modules
# are built. For the binaries, they will be linked to the Caffe2 main
# libraries, as well as all the modules that are built with Caffe2 (the ones
# built in the previous Modules section above).
if (BUILD_BINARY)
  add_subdirectory(binaries)
endif()

include(cmake/Summary.cmake)
caffe2_print_configuration_summary()
