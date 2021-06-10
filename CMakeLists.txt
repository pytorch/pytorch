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

if(NOT CMAKE_VERSION VERSION_LESS 3.10)
  set(FIND_CUDA_MODULE_DEPRECATED ON)
endif()

# ---[ Project and semantic versioning.
project(Torch CXX C)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(LINUX TRUE)
else()
  set(LINUX FALSE)
endif()

set(CMAKE_INSTALL_MESSAGE NEVER)

# check and set CMAKE_CXX_STANDARD
string(FIND "${CMAKE_CXX_FLAGS}" "-std=c++" env_cxx_standard)
if(env_cxx_standard GREATER -1)
  message(
      WARNING "C++ standard version definition detected in environment variable."
      "PyTorch requires -std=c++14. Please remove -std=c++ settings in your environment.")
endif()
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 11)

if(DEFINED GLIBCXX_USE_CXX11_ABI)
  if(${GLIBCXX_USE_CXX11_ABI} EQUAL 1)
    set(CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
  endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# One variable that determines whether the current cmake process is being run
# with the main Caffe2 library. This is useful for building modules - if
# modules are built with the main Caffe2 library then one does not need to do
# find caffe2 in the cmake script. One can usually guard it in some way like
#    if(NOT CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
#      find_package(Caffe2 REQUIRED)
#    endif()
set(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO ON)

# Googletest's cmake files are going to set it on once they are processed. Let's
# set it at the very beginning so that the entire build is deterministic.
set(THREADS_PREFER_PTHREAD_FLAG ON)

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
  execute_process( COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE clang_full_version_string )
  string(REGEX REPLACE "Apple LLVM version ([0-9]+\\.[0-9]+).*" "\\1" CLANG_VERSION_STRING ${clang_full_version_string})
  message( STATUS "CLANG_VERSION_STRING:         " ${CLANG_VERSION_STRING} )


  # RPATH stuff
  set(CMAKE_MACOSX_RPATH ON)
  if(NOT IOS)
    # Determine if we can link against ML Compute
    set(MLCOMPUTE_FOUND OFF)
    execute_process(
      COMMAND bash -c "xcrun --sdk macosx --show-sdk-path"
      OUTPUT_VARIABLE _macosx_sdk_path
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(_SDK_SEARCH_PATH "${_macosx_sdk_path}/System/Library/Frameworks/")
    set(_FRAMEWORK_SEARCH_PATH "/System/Library/Frameworks/")

    find_library(_MLCompute_fwrk_path_ NAMES MLCompute PATHS ${_FRAMEWORK_SEARCH_PATH} NO_DEFAULT_PATH)
    find_library(_MLCompute_sdk_path_ NAMES MLCompute PATHS ${_SDK_SEARCH_PATH} NO_DEFAULT_PATH)

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mlc)
      set(_MLC_FOLDER_EXISTS YES)
    else()
      set(_MLC_FOLDER_EXISTS NO)
    endif()

    if(_MLCompute_fwrk_path_ AND _MLCompute_sdk_path_ AND _MLC_FOLDER_EXISTS)
      set(MLCOMPUTE_FOUND ON)
      message(STATUS "ML Compute framework found")
    else()
      message(STATUS "ML Compute framework not found")
    endif()
  endif()
endif()

set(CPU_AARCH64 OFF)
set(CPU_INTEL OFF)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(AMD64|x86_64)")
  set(CPU_INTEL ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
  set(CPU_AARCH64 ON)
endif()


# For non-supported platforms, turn USE_DISTRIBUTED off by default.
# It is not tested and likely won't work without additional changes.
if(NOT LINUX AND NOT WIN32)
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
option(BUILD_CAFFE2 "Master flag to build Caffe2" ON)
option(BUILD_LITE_INTERPRETER "Master flag to build Lite Interpreter" OFF)
cmake_dependent_option(
    BUILD_CAFFE2_OPS "Build Caffe2 operators" ON
    "BUILD_CAFFE2" OFF)
cmake_dependent_option(
    BUILD_CAFFE2_MOBILE "Build libcaffe2 for mobile (deprecating)" OFF
    "BUILD_CAFFE2" OFF)
option(BUILD_SHARED_LIBS "Build libcaffe2.so" ON)
cmake_dependent_option(
    CAFFE2_LINK_LOCAL_PROTOBUF "If set, build protobuf inside libcaffe2.so." ON
    "BUILD_SHARED_LIBS AND BUILD_CUSTOM_PROTOBUF" OFF)
cmake_dependent_option(
    CAFFE2_USE_MSVC_STATIC_RUNTIME "Using MSVC static runtime libraries" ON
    "NOT BUILD_SHARED_LIBS" OFF)
option(BUILD_TEST "Build C++ test binaries (need gtest and gbenchmark)" OFF)
option(BUILD_STATIC_RUNTIME_BENCHMARK "Build C++ binaries for static runtime benchmarks (need gbenchmark)" OFF)
option(BUILD_TENSOREXPR_BENCHMARK "Build C++ binaries for tensorexpr benchmarks (need gbenchmark)" OFF)
option(BUILD_MOBILE_BENCHMARK "Build C++ test binaries for mobile (ARM) targets(need gtest and gbenchmark)" OFF)
option(BUILD_MOBILE_TEST "Build C++ test binaries for mobile (ARM) targets(need gtest and gbenchmark)" OFF)
option(BUILD_JNI "Build JNI bindings" OFF)
option(BUILD_MOBILE_AUTOGRAD "Build autograd function in mobile build (in development)" OFF)
cmake_dependent_option(
    INSTALL_TEST "Install test binaries if BUILD_TEST is on" ON
    "BUILD_TEST" OFF)
option(USE_CPP_CODE_COVERAGE "Compile C/C++ with code coverage flags" OFF)
option(COLORIZE_OUTPUT "Colorize output during compilation" ON)
option(USE_ASAN "Use Address Sanitizer" OFF)
option(USE_TSAN "Use Thread Sanitizer" OFF)
option(USE_CUDA "Use CUDA" ON)
# BUILD_SPLIT_CUDA must also be exported as an environment variable before building, with
# `export BUILD_SPLIT_CUDA=1` because cpp_extension.py can only work properly if this variable
# also exists in the environment.
# This option is incompatible with CUDA_SEPARABLE_COMPILATION.
cmake_dependent_option(
    BUILD_SPLIT_CUDA "Split torch_cuda library into torch_cuda_cu and torch_cuda_cpp" OFF
    "USE_CUDA AND NOT CUDA_SEPARABLE_COMPILATION" OFF)
option(USE_FAST_NVCC "Use parallel NVCC build" OFF)
option(USE_ROCM "Use ROCm" ON)
option(CAFFE2_STATIC_LINK_CUDA "Statically link CUDA libraries" OFF)
cmake_dependent_option(
    USE_CUDNN "Use cuDNN" ON
    "USE_CUDA" OFF)
cmake_dependent_option(
    USE_STATIC_CUDNN "Use cuDNN static libraries" OFF
    "USE_CUDNN" OFF)
cmake_dependent_option(
  USE_WHOLE_CUDNN "Use whole-library linking for cuDNN" OFF
    "USE_STATIC_CUDNN" OFF)
option(USE_FBGEMM "Use FBGEMM (quantized 8-bit server operators)" ON)
option(USE_KINETO "Use Kineto profiling library" ON)
option(USE_CUPTI_SO "Use CUPTI as a shared library" OFF)
option(USE_FAKELOWP "Use FakeLowp operators" OFF)
option(USE_FFMPEG "Use ffmpeg" OFF)
option(USE_GFLAGS "Use GFLAGS" OFF)
option(USE_GLOG "Use GLOG" OFF)
option(USE_LEVELDB "Use LEVELDB" OFF)
option(USE_LITE_PROTO "Use lite protobuf instead of full." OFF)
option(USE_LMDB "Use LMDB" OFF)
option(USE_MAGMA "Use MAGMA" ON)
option(USE_METAL "Use Metal for Caffe2 iOS build" ON)
option(USE_PYTORCH_METAL "Use Metal for PyTorch iOS build" OFF)
option(USE_NATIVE_ARCH "Use -march=native" OFF)
cmake_dependent_option(
    USE_MLCOMPUTE "Use ML Compute for macOS build" ON
    "MLCOMPUTE_FOUND" OFF)
cmake_dependent_option(
    USE_NCCL "Use NCCL" ON
    "USE_CUDA OR USE_ROCM;UNIX;NOT APPLE" OFF)
cmake_dependent_option(USE_RCCL "Use RCCL" ON
    USE_NCCL OFF)
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
cmake_dependent_option(
    USE_VALGRIND "Use Valgrind. Only available on Linux." ON
    "LINUX" OFF)

if(NOT DEFINED USE_VULKAN)
  cmake_dependent_option(
      USE_VULKAN "Use Vulkan GPU backend" ON
      "ANDROID" OFF)
endif()

option(USE_SOURCE_DEBUG_ON_MOBILE "Enable " ON)
option(USE_VULKAN_FP16_INFERENCE "Vulkan - Use fp16 inference" OFF)
option(USE_VULKAN_RELAXED_PRECISION "Vulkan - Use relaxed precision math in the kernels (mediump)" OFF)
option(USE_VULKAN_SHADERC_RUNTIME "Vulkan - Use runtime shader compilation as opposed to build-time (needs libshaderc)" OFF)
option(USE_XNNPACK "Use XNNPACK" ON)
option(USE_ZMQ "Use ZMQ" OFF)
option(USE_ZSTD "Use ZSTD" OFF)
# Ensure that an MKLDNN build is the default for x86 CPUs
# but optional for AArch64 (dependent on -DUSE_MKLDNN).
cmake_dependent_option(
  USE_MKLDNN "Use MKLDNN. Only available on x86, x86_64, and AArch64." "${CPU_INTEL}"
  "CPU_INTEL OR CPU_AARCH64" OFF)
cmake_dependent_option(
  USE_MKLDNN_ACL "Use Compute Library for the Arm architecture." OFF
  "USE_MKLDNN AND CPU_AARCH64" OFF)
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
cmake_dependent_option(
    USE_GLOO_WITH_OPENSSL "Use Gloo with OpenSSL. Only available if USE_GLOO is on." OFF
    "USE_GLOO AND LINUX AND NOT INTERN_BUILD_MOBILE" OFF)
cmake_dependent_option(
    USE_TENSORPIPE "Use TensorPipe. Only available if USE_DISTRIBUTED is on." ON
    "USE_DISTRIBUTED" OFF)
option(USE_TBB "Use TBB" OFF)
option(ONNX_ML "Enable traditional ONNX ML API." ON)
option(HAVE_SOVERSION "Whether to add SOVERSION to the shared objects" OFF)
option(BUILD_LIBTORCH_CPU_WITH_DEBUG "Enable RelWithDebInfo for libtorch_cpu target only" OFF)
cmake_dependent_option(
    USE_DEPLOY "Build embedded torch::deploy interpreter.  See torch/csrc/deploy/README.md for more info." OFF
    "BUILD_PYTHON" OFF)
cmake_dependent_option(USE_CCACHE "Attempt using CCache to wrap the compilation" ON "UNIX" OFF)
option(WERROR "Build with -Werror supported by the compiler" OFF)

if(USE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  else()
    message(STATUS "Could not find ccache. Consider installing ccache to speed up compilation.")
  endif()
endif()

# Since TensorPipe does not support Windows, set it to OFF when WIN32 detected
# On Windows platform, if user does not install libuv in build conda env and
# does not set libuv_ROOT environment variable. Set USE_DISTRIBUTED to OFF.
if(WIN32)
  set(USE_TENSORPIPE OFF)
  message(WARNING "TensorPipe cannot be used on Windows. Set it to OFF")

  if(USE_DISTRIBUTED AND NOT DEFINED ENV{libuv_ROOT})
    find_library(
      libuv_tmp_LIBRARY
      NAMES uv libuv
      HINTS $ENV{CONDA_PREFIX}\\Library $ENV{PREFIX}\\Library
      PATH_SUFFIXES lib
      NO_DEFAULT_PATH)
    if(NOT libuv_tmp_LIBRARY)
      set(USE_DISTRIBUTED OFF)
      set(USE_GLOO OFF)
      message(
        WARNING "Libuv is not installed in current conda env. Set USE_DISTRIBUTED to OFF. "
        "Please run command 'conda install -c conda-forge libuv=1.39' to install libuv.")
    else()
      set(ENV{libuv_ROOT} ${libuv_tmp_LIBRARY}/../../)
    endif()
  endif()
endif()

if(USE_GLOO_WITH_OPENSSL)
  set(USE_TCP_OPENSSL_LOAD ON CACHE STRING "")
endif()

# Linux distributions do not want too many embedded sources, in that sense we
# need to be able to build pytorch with an (almost) empty third_party
# directory.
# USE_SYSTEM_LIBS is a shortcut variable to toggle all the # USE_SYSTEM_*
# variables on. Individual USE_SYSTEM_* variables can be toggled with
# USE_SYSTEM_LIBS being "OFF".
option(USE_SYSTEM_LIBS "Use all available system-provided libraries." OFF)
option(USE_SYSTEM_CPUINFO "Use system-provided cpuinfo." OFF)
option(USE_SYSTEM_SLEEF "Use system-provided sleef." OFF)
option(USE_SYSTEM_GLOO "Use system-provided gloo." OFF)
option(USE_SYSTEM_FP16 "Use system-provided fp16." OFF)
option(USE_SYSTEM_PTHREADPOOL "Use system-provided pthreadpool." OFF)
option(USE_SYSTEM_PSIMD "Use system-provided psimd." OFF)
option(USE_SYSTEM_FXDIV "Use system-provided fxdiv." OFF)
option(USE_SYSTEM_BENCHMARK "Use system-provided google benchmark." OFF)
option(USE_SYSTEM_ONNX "Use system-provided onnx." OFF)
option(USE_SYSTEM_XNNPACK "Use system-provided xnnpack." OFF)
option(USE_GOLD_LINKER "Use ld.gold to link" OFF)
if(USE_SYSTEM_LIBS)
  set(USE_SYSTEM_CPUINFO ON)
  set(USE_SYSTEM_SLEEF ON)
  set(USE_SYSTEM_GLOO ON)
  set(BUILD_CUSTOM_PROTOBUF OFF)
  set(USE_SYSTEM_EIGEN_INSTALL ON)
  set(USE_SYSTEM_FP16 ON)
  set(USE_SYSTEM_PTHREADPOOL ON)
  set(USE_SYSTEM_PSIMD ON)
  set(USE_SYSTEM_FXDIV ON)
  set(USE_SYSTEM_BENCHMARK ON)
  set(USE_SYSTEM_ONNX ON)
  set(USE_SYSTEM_XNNPACK ON)
endif()

# Used when building Caffe2 through setup.py
option(BUILDING_WITH_TORCH_LIBS "Tell cmake if Caffe2 is being built alongside torch libs" ON)

# /Z7 override option
# When generating debug symbols, CMake default to use the flag /Zi.
# However, it is not compatible with sccache. So we rewrite it off.
# But some users don't use sccache; this override is for them.
cmake_dependent_option(
  MSVC_Z7_OVERRIDE "Work around sccache bug by replacing /Zi and /ZI with /Z7 when using MSVC (if you are not using sccache, you can turn this OFF)" ON
  "MSVC" OFF)

if(NOT USE_SYSTEM_ONNX)
  set(ONNX_NAMESPACE "onnx_torch" CACHE STRING "A namespace for ONNX; needed to build with other frameworks that share ONNX.")
else()
  set(ONNX_NAMESPACE "onnx" CACHE STRING "A namespace for ONNX; needed to build with other frameworks that share ONNX.")
endif()
set(SELECTED_OP_LIST "" CACHE STRING
    "Path to the yaml file that contains the list of operators to include for custom build. Include all operators by default.")
set(OP_DEPENDENCY "" CACHE STRING
    "Path to the yaml file that contains the op dependency graph for custom build.")
set(STATIC_DISPATCH_BACKEND "" CACHE STRING
    "Name of the backend for which static dispatch code is generated, e.g.: CPU.")

# This is a fix for a rare build issue on Ubuntu:
# symbol lookup error: miniconda3/envs/pytorch-py3.7/lib/libmkl_intel_lp64.so: undefined symbol: mkl_blas_dsyrk
# https://software.intel.com/en-us/articles/symbol-lookup-error-when-linking-intel-mkl-with-gcc-on-ubuntu
if(LINUX)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")
endif()

if(MSVC)
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

    # Turn off warnings (Windows build is currently is extremely warning
    # unclean and the warnings aren't telling us anything useful.)
    string(APPEND ${flag_var} " /w")

    if(${CAFFE2_USE_MSVC_STATIC_RUNTIME})
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    else()
      if(${flag_var} MATCHES "/MT")
        string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
      endif()
    endif()

    # /bigobj increases number of sections in .obj file, which is needed to link
    # against libraries in Python 2.7 under Windows
    # For Visual Studio generators, if /MP is not added, then we may need
    # to add /MP to the flags.
    # For other generators like ninja, we don't need to add /MP because it is
    # already handled by the generator itself.
    if(CMAKE_GENERATOR MATCHES "Visual Studio" AND NOT ${flag_var} MATCHES "/MP")
      set(${flag_var} "${${flag_var}} /MP /bigobj")
    else()
      set(${flag_var} "${${flag_var}} /bigobj")
    endif()
  endforeach(flag_var)

  foreach(flag_var
      CMAKE_C_FLAGS CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_MINSIZEREL
      CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL)
    if(${flag_var} MATCHES "/Z[iI7]")
      string(REGEX REPLACE "/Z[iI7]" "" ${flag_var} "${${flag_var}}")
    endif()
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
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "/w" "-w")
endif(MSVC)

list(APPEND CUDA_NVCC_FLAGS "-Xfatbin" "-compress-all")
list(APPEND CUDA_NVCC_FLAGS_DEBUG "-Xfatbin" "-compress-all")
list(APPEND CUDA_NVCC_FLAGS_RELWITHDEBINFO "-Xfatbin" "-compress-all")

if(NOT MSVC)
  list(APPEND CUDA_NVCC_FLAGS_DEBUG "-g" "-lineinfo" "--source-in-ptx")
  list(APPEND CUDA_NVCC_FLAGS_RELWITHDEBINFO "-g" "-lineinfo" "--source-in-ptx")
endif(NOT MSVC)

# Set INTERN_BUILD_MOBILE for all mobile builds. Components that are not
# applicable to mobile are disabled by this variable.
# Setting `BUILD_PYTORCH_MOBILE_WITH_HOST_TOOLCHAIN` environment variable can
# force it to do mobile build with host toolchain - which is useful for testing
# purpose.
if(ANDROID OR IOS OR DEFINED ENV{BUILD_PYTORCH_MOBILE_WITH_HOST_TOOLCHAIN})
  set(INTERN_BUILD_MOBILE ON)

  if(DEFINED ENV{BUILD_PYTORCH_MOBILE_WITH_HOST_TOOLCHAIN})
    # C10_MOBILE is derived from Android/iOS toolchain macros in
    # c10/macros/Macros.h, so it needs to be explicitly set here.
    string(APPEND CMAKE_CXX_FLAGS " -DC10_MOBILE")
  endif()
endif()

# INTERN_BUILD_ATEN_OPS is used to control whether to build ATen/TH operators.
# It's disabled for caffe2 mobile library.
if(INTERN_BUILD_MOBILE AND BUILD_CAFFE2_MOBILE)
  set(INTERN_BUILD_ATEN_OPS OFF)
else()
  set(INTERN_BUILD_ATEN_OPS ON)
endif()

# BUILD_CAFFE2_MOBILE is the master switch to choose between libcaffe2 v.s. libtorch mobile build.
# When it's enabled it builds original libcaffe2 mobile library without ATen/TH ops nor TorchScript support;
# When it's disabled it builds libtorch mobile library, which contains ATen/TH ops and native support for
# TorchScript model, but doesn't contain not-yet-unified caffe2 ops;
if(INTERN_BUILD_MOBILE AND NOT BUILD_CAFFE2_MOBILE)
  if(NOT BUILD_SHARED_LIBS AND NOT "${SELECTED_OP_LIST}" STREQUAL "")
    string(APPEND CMAKE_CXX_FLAGS " -DNO_EXPORT")
  endif()
  if(BUILD_MOBILE_AUTOGRAD)
    set(INTERN_DISABLE_AUTOGRAD OFF)
  else()
    set(INTERN_DISABLE_AUTOGRAD ON)
  endif()
  set(BUILD_PYTHON OFF)
  set(BUILD_CAFFE2_OPS OFF)
  set(USE_DISTRIBUTED OFF)
  set(NO_API ON)
  set(USE_FBGEMM OFF)
  set(USE_QNNPACK OFF)
  set(INTERN_DISABLE_ONNX ON)
  set(INTERN_USE_EIGEN_BLAS ON)
  # Disable developing mobile interpreter for actual mobile build.
  # Enable it elsewhere to capture build error.
  set(INTERN_DISABLE_MOBILE_INTERP ON)
endif()

# ---[ Utils
include(cmake/public/utils.cmake)

# ---[ Version numbers for generated libraries
file(READ version.txt TORCH_DEFAULT_VERSION)
# Strip trailing newline
string(REGEX REPLACE "\n$" "" TORCH_DEFAULT_VERSION "${TORCH_DEFAULT_VERSION}")
if("${TORCH_DEFAULT_VERSION} " STREQUAL " ")
  message(WARNING "Could not get version from base 'version.txt'")
  # If we can't get the version from the version file we should probably
  # set it to something non-sensical like 0.0.0
  set(TORCH_DEFAULT_VERSION, "0.0.0")
endif()
set(TORCH_BUILD_VERSION "${TORCH_DEFAULT_VERSION}" CACHE STRING "Torch build version")
if(DEFINED ENV{PYTORCH_BUILD_VERSION})
  set(TORCH_BUILD_VERSION "$ENV{PYTORCH_BUILD_VERSION}"
    CACHE STRING "Torch build version" FORCE)
endif()
if(NOT TORCH_BUILD_VERSION)
  # An empty string was specified so force version to the default
  set(TORCH_BUILD_VERSION "${TORCH_DEFAULT_VERSION}"
    CACHE STRING "Torch build version" FORCE)
endif()
caffe2_parse_version_str(TORCH ${TORCH_BUILD_VERSION})
caffe2_parse_version_str(CAFFE2 ${TORCH_BUILD_VERSION})
set(TORCH_SOVERSION "${TORCH_VERSION_MAJOR}.${TORCH_VERSION_MINOR}")

# ---[ CMake scripts + modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# ---[ CMake build directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

enable_testing()

# ---[ Build variables set within the cmake tree
include(cmake/BuildVariables.cmake)
set(CAFFE2_ALLOWLIST "" CACHE STRING "A allowlist file of files that one should build.")

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type not set - defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build from: Debug Release RelWithDebInfo MinSizeRel Coverage." FORCE)
endif()

# The below means we are cross compiling for arm64 or x86_64 on MacOSX
if(NOT IOS AND CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_OSX_ARCHITECTURES MATCHES "^(x86_64|arm64)$")
  set(CROSS_COMPILING_MACOSX TRUE)
  # We need to compile a universal protoc to not fail protobuf build
  # We set CMAKE_TRY_COMPILE_TARGET_TYPE to STATIC_LIBRARY (vs executable) to succeed the cmake compiler check for cross-compiling
  set(protoc_build_command "./scripts/build_host_protoc.sh --other-flags -DCMAKE_OSX_ARCHITECTURES=\"x86_64;arm64\" -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY -DCMAKE_C_COMPILER_WORKS=1 -DCMAKE_CXX_COMPILER_WORKS=1")
  # We write to a temp scriptfile because CMake COMMAND dislikes double quotes in commands
  file(WRITE ${PROJECT_SOURCE_DIR}/tmp_protoc_script.sh "#!/bin/bash\n${protoc_build_command}")
  file(COPY ${PROJECT_SOURCE_DIR}/tmp_protoc_script.sh DESTINATION ${PROJECT_SOURCE_DIR}/scripts/ FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ)
  execute_process(COMMAND ./scripts/tmp_protoc_script.sh
                  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                  RESULT_VARIABLE BUILD_HOST_PROTOC_RESULT)
  file(REMOVE ${PROJECT_SOURCE_DIR}/tmp_protoc_script.sh ${PROJECT_SOURCE_DIR}/scripts/tmp_protoc_script.sh)
  if(NOT BUILD_HOST_PROTOC_RESULT EQUAL "0")
    message(FATAL_ERROR "Could not compile universal protoc.")
  endif()
  set(PROTOBUF_PROTOC_EXECUTABLE "${PROJECT_SOURCE_DIR}/build_host_protoc/bin/protoc")
  set(CAFFE2_CUSTOM_PROTOC_EXECUTABLE "${PROJECT_SOURCE_DIR}/build_host_protoc/bin/protoc")
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
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_FBGEMM")
endif()

if(USE_QNNPACK)
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_QNNPACK")
endif()

if(USE_PYTORCH_QNNPACK)
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_PYTORCH_QNNPACK")
endif()

if(USE_XNNPACK)
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_XNNPACK")
endif()

if(USE_VULKAN)
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_VULKAN")
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_VULKAN_API")

  if(USE_VULKAN_FP16_INFERENCE)
    string(APPEND CMAKE_CXX_FLAGS " -DUSE_VULKAN_FP16_INFERENCE")
  endif()

  if(USE_VULKAN_RELAXED_PRECISION)
    string(APPEND CMAKE_CXX_FLAGS " -DUSE_VULKAN_RELAXED_PRECISION")
  endif()

  if(USE_VULKAN_SHADERC_RUNTIME)
    string(APPEND CMAKE_CXX_FLAGS " -DUSE_VULKAN_SHADERC_RUNTIME")
  endif()
endif()

if(BUILD_LITE_INTERPRETER)
  string(APPEND CMAKE_CXX_FLAGS " -DBUILD_LITE_INTERPRETER")
endif()

if(USE_PYTORCH_METAL)
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_PYTORCH_METAL")
endif()

if(USE_SOURCE_DEBUG_ON_MOBILE)
  string(APPEND CMAKE_CXX_FLAGS " -DSYMBOLICATE_MOBILE_DEBUG_HANDLE")
endif()

# ---[ Allowlist file if allowlist is specified
include(cmake/Allowlist.cmake)

# ---[ Set link flag, handle additional deps for gcc 4.8 and above
if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.8.0 AND NOT ANDROID)
  message(STATUS "GCC ${CMAKE_CXX_COMPILER_VERSION}: Adding gcc and gcc_s libs to link line")
  list(APPEND Caffe2_DEPENDENCY_LIBS gcc_s gcc)
endif()

# ---[ Build flags
if(NOT MSVC)
  string(APPEND CMAKE_CXX_FLAGS " -O2 -fPIC")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-narrowing")
  # Eigen fails to build with some versions, so convert this to a warning
  # Details at http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1459
  string(APPEND CMAKE_CXX_FLAGS " -Wall")
  string(APPEND CMAKE_CXX_FLAGS " -Wextra")
  string(APPEND CMAKE_CXX_FLAGS " -Werror=return-type")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-missing-field-initializers")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-type-limits")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-array-bounds")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-unknown-pragmas")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-sign-compare")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-parameter")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-variable")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-function")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-result")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-local-typedefs")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-strict-overflow")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-strict-aliasing")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-error=deprecated-declarations")
  if(CMAKE_COMPILER_IS_GNUCXX AND NOT (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0.0))
    string(APPEND CMAKE_CXX_FLAGS " -Wno-stringop-overflow")
  endif()
  if(CMAKE_COMPILER_IS_GNUCXX)
    # Suppress "The ABI for passing parameters with 64-byte alignment has changed in GCC 4.6"
    string(APPEND CMAKE_CXX_FLAGS " -Wno-psabi")
  endif()

  # Use ld.gold if available, fall back to ld.bfd (the default ld) if not
  if(USE_GOLD_LINKER)
    if(USE_DISTRIBUTED AND USE_MPI)
      # Same issue as here with default MPI on Ubuntu
      # https://bugs.launchpad.net/ubuntu/+source/deal.ii/+bug/1841577
      message(WARNING "Refusing to use gold when USE_MPI=1")
    else()
      execute_process(
        COMMAND
        "${CMAKE_C_COMPILER}" -fuse-ld=gold -Wl,--version
         ERROR_QUIET
         OUTPUT_VARIABLE LD_VERSION)
      if(NOT "${LD_VERSION}" MATCHES "GNU gold")
        message(WARNING "USE_GOLD_LINKER was set but ld.gold isn't available, turning it off")
        set(USE_GOLD_LINKER OFF)
      else()
        message(STATUS "ld.gold is available, using it to link")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=gold")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fuse-ld=gold")
      endif()
    endif()
  endif()

  string(APPEND CMAKE_CXX_FLAGS " -Wno-error=pedantic")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-error=redundant-decls")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-error=old-style-cast")
  # These flags are not available in GCC-4.8.5. Set only when using clang.
  # Compared against https://gcc.gnu.org/onlinedocs/gcc-4.8.5/gcc/Option-Summary.html
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-invalid-partial-specialization")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-typedef-redefinition")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-unknown-warning-option")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-private-field")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-inconsistent-missing-override")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-aligned-allocation-unavailable")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-c++14-extensions")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-constexpr-not-const")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-missing-braces")
    string(APPEND CMAKE_CXX_FLAGS " -Qunused-arguments")
    if(${COLORIZE_OUTPUT})
      string(APPEND CMAKE_CXX_FLAGS " -fcolor-diagnostics")
    endif()
  endif()
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)
    if(${COLORIZE_OUTPUT})
      string(APPEND CMAKE_CXX_FLAGS " -fdiagnostics-color=always")
    endif()
  endif()
  if((APPLE AND (NOT ("${CLANG_VERSION_STRING}" VERSION_LESS "9.0")))
    OR(CMAKE_COMPILER_IS_GNUCXX
    AND(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0 AND NOT APPLE)))
    string(APPEND CMAKE_CXX_FLAGS " -faligned-new")
  endif()
  if(WERROR)
    check_cxx_compiler_flag("-Werror" COMPILER_SUPPORT_WERROR)
    if(NOT COMPILER_SUPPORT_WERROR)
      set(WERROR FALSE)
    else()
      string(APPEND CMAKE_CXX_FLAGS " -Werror")
    endif()
  endif(WERROR)
  if(NOT APPLE)
    string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-but-set-variable")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-maybe-uninitialized")
  endif()
  string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fno-omit-frame-pointer -O0")
  string(APPEND CMAKE_LINKER_FLAGS_DEBUG " -fno-omit-frame-pointer -O0")
  string(APPEND CMAKE_CXX_FLAGS " -fno-math-errno")
  string(APPEND CMAKE_CXX_FLAGS " -fno-trapping-math")
  check_cxx_compiler_flag("-Werror=format" HAS_WERROR_FORMAT)
  if(HAS_WERROR_FORMAT)
    string(APPEND CMAKE_CXX_FLAGS " -Werror=format")
  endif()
  check_cxx_compiler_flag("-Werror=cast-function-type" HAS_WERROR_CAST_FUNCTION_TYPE)
  if(HAS_WERROR_CAST_FUNCTION_TYPE)
    string(APPEND CMAKE_CXX_FLAGS " -Werror=cast-function-type")
  endif()
endif()

if(USE_ASAN)
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fsanitize=address")
    string(APPEND CMAKE_LINKER_FLAGS_DEBUG " -fsanitize=address")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  include(CheckCSourceCompiles)
  check_c_source_compiles("#include <arm_neon.h>
int main() {
  float a[] = {1.0, 1.0};
  float32x4x2_t v;
  v.val[0] = vcombine_f32 (vcreate_f32 (0UL), vcreate_f32 (0UL));
  v.val[1] = vcombine_f32 (vcreate_f32 (0UL), vcreate_f32 (0UL));
  vst1q_f32_x2(a, v);
  return 0;
}" HAS_VST1)

  if(NOT HAS_VST1)
    string(APPEND CMAKE_CXX_FLAGS " -DMISSING_ARM_VST1")
  endif()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  include(CheckCSourceCompiles)
  check_c_source_compiles("#include <arm_neon.h>
int main() {
  float a[] = {1.0, 1.0};
  vld1q_f32_x2(a);
  return 0;
}" HAS_VLD1)

  if(NOT HAS_VLD1)
    string(APPEND CMAKE_CXX_FLAGS " -DMISSING_ARM_VLD1")
  endif()
endif()


# Add code coverage flags to supported compilers
if(USE_CPP_CODE_COVERAGE)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    string(APPEND CMAKE_C_FLAGS  " --coverage -fprofile-abs-path")
    string(APPEND CMAKE_CXX_FLAGS  " --coverage -fprofile-abs-path")
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    string(APPEND CMAKE_C_FLAGS  " -fprofile-instr-generate -fcoverage-mapping")
    string(APPEND CMAKE_CXX_FLAGS " -fprofile-instr-generate -fcoverage-mapping")
  else()
    message(ERROR "Code coverage for compiler ${CMAKE_CXX_COMPILER_ID} is unsupported")
  endif()

endif()

if(APPLE)
    if(USE_MLCOMPUTE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MLCOMPUTE -fobjc-arc -framework MLCompute -framework Metal")
    endif()
    string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-private-field")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-missing-braces")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-c++14-extensions")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-constexpr-not-const")
endif()

if(EMSCRIPTEN)
  string(APPEND CMAKE_CXX_FLAGS " -Wno-implicit-function-declaration -DEMSCRIPTEN -s DISABLE_EXCEPTION_CATCHING=0")
endif()

if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0.0)
  string(APPEND CMAKE_CXX_FLAGS " -Wno-stringop-overflow")
endif()

if(ANDROID AND (NOT ANDROID_DEBUG_SYMBOLS))
  if(CMAKE_COMPILER_IS_GNUCXX)
    string(APPEND CMAKE_CXX_FLAGS " -s")
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    string(APPEND CMAKE_CXX_FLAGS " -g0")
  else()
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -s")
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
  if(DOXYGEN_FOUND)
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

if((NOT USE_GLOG) OR (NOT USE_GFLAGS) OR BUILD_CUSTOM_PROTOBUF)
  message(WARNING
      "Generated cmake files are only fully tested if one builds "
      "with system glog, gflags, and protobuf. Other settings may "
      "generate files that are not well tested.")
endif()

if(USE_CUDA OR USE_ROCM)
  # TODO: check if we should include other cuda dependency libraries
  # to the interface as well.

endif()

# Note(jiayq): when building static libraries, all PRIVATE dependencies
# will also become interface libraries, and as a result if there are any
# dependency libraries that are not exported, the following install export
# script will fail. As a result, we will only provide the targets cmake
# files for shared lib installation. For more info, read:
# https://cmake.org/pipermail/cmake/2016-May/063400.html
if(BUILD_SHARED_LIBS)
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
# If master flag for buildling Caffe2 is disabled, we also disable the
# build for Caffe2 related operator modules.
if(BUILD_CAFFE2)
  add_subdirectory(modules)
endif()

# ---[ Binaries
# Binaries will be built after the Caffe2 main libraries and the modules
# are built. For the binaries, they will be linked to the Caffe2 main
# libraries, as well as all the modules that are built with Caffe2 (the ones
# built in the previous Modules section above).
if(BUILD_BINARY)
  add_subdirectory(binaries)
endif()

# ---[ JNI
if(BUILD_JNI)
  set(BUILD_LIBTORCH_WITH_JNI 1)
  set(FBJNI_SKIP_TESTS 1)
  add_subdirectory(android/pytorch_android)
endif()

include(cmake/Summary.cmake)
caffe2_print_configuration_summary()

# ---[ Torch Deploy
if(USE_DEPLOY)
  add_subdirectory(torch/csrc/deploy)
endif()
