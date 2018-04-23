# ---[ Macro to update cached options.
macro (caffe2_update_option variable value)
  get_property(__help_string CACHE ${variable} PROPERTY HELPSTRING)
  set(${variable} ${value} CACHE BOOL ${__help_string} FORCE)
endmacro()

# ---[ Custom Protobuf
include("cmake/ProtoBuf.cmake")

# ---[ Threads
include(cmake/public/threads.cmake)
if (TARGET Threads::Threads)
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS Threads::Threads)
else()
  message(FATAL_ERROR
      "Cannot find threading library. Caffe2 requires Threads to compile.")
endif()

# ---[ protobuf
if(USE_LITE_PROTO)
  set(CAFFE2_USE_LITE_PROTO 1)
endif()

# ---[ git: used to generate git build string.
find_package(Git)
if(GIT_FOUND)
  execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
                  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
                  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                  OUTPUT_VARIABLE CAFFE2_GIT_VERSION
                  RESULT_VARIABLE __git_result)
  if(NOT ${__git_result} EQUAL 0)
    set(CAFFE2_GIT_VERSION "unknown")
  endif()
else()
  message(
      WARNING
      "Cannot find git, so Caffe2 won't have any git build info available")
endif()


# ---[ BLAS
set(BLAS "Eigen" CACHE STRING "Selected BLAS library")
set_property(CACHE BLAS PROPERTY STRINGS "Eigen;ATLAS;OpenBLAS;MKL;vecLib")
message(STATUS "The BLAS backend of choice:" ${BLAS})

if(BLAS STREQUAL "Eigen")
  # Eigen is header-only and we do not have any dependent libraries
  set(CAFFE2_USE_EIGEN_FOR_BLAS 1)
elseif(BLAS STREQUAL "ATLAS")
  find_package(Atlas REQUIRED)
  include_directories(${ATLAS_INCLUDE_DIRS})
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS ${ATLAS_LIBRARIES})
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS cblas)
elseif(BLAS STREQUAL "OpenBLAS")
  find_package(OpenBLAS REQUIRED)
  include_directories(${OpenBLAS_INCLUDE_DIR})
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS ${OpenBLAS_LIB})
elseif(BLAS STREQUAL "MKL")
  find_package(MKL REQUIRED)
  include_directories(${MKL_INCLUDE_DIR})
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS ${MKL_LIBRARIES})
elseif(BLAS STREQUAL "vecLib")
  find_package(vecLib REQUIRED)
  include_directories(${vecLib_INCLUDE_DIR})
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS ${vecLib_LINKER_LIBS})
else()
  message(FATAL_ERROR "Unrecognized blas option:" ${BLAS})
endif()

# Directory where NNPACK and cpuinfo will download and build all dependencies
set(CONFU_DEPENDENCIES_SOURCE_DIR ${PROJECT_BINARY_DIR}/confu-srcs
  CACHE PATH "Confu-style dependencies source directory")
set(CONFU_DEPENDENCIES_BINARY_DIR ${PROJECT_BINARY_DIR}/confu-deps
  CACHE PATH "Confu-style dependencies binary directory")

# ---[ NNPACK
if(USE_NNPACK)
  include("cmake/External/nnpack.cmake")
  if(NNPACK_FOUND)
    if(TARGET nnpack)
      # ---[ NNPACK is being built together with Caffe2: explicitly specify dependency
      list(APPEND Caffe2_DEPENDENCY_LIBS nnpack)
    else()
      include_directories(${NNPACK_INCLUDE_DIRS})
      list(APPEND Caffe2_DEPENDENCY_LIBS ${NNPACK_LIBRARIES})
    endif()
  else()
    message(WARNING "Not compiling with NNPACK. Suppress this warning with -DUSE_NNPACK=OFF")
    caffe2_update_option(USE_NNPACK OFF)
  endif()
endif()

# ---[ Caffe2 uses cpuinfo library in the thread pool
if (NOT TARGET cpuinfo)
  if (NOT DEFINED CPUINFO_SOURCE_DIR)
    set(CPUINFO_SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/cpuinfo" CACHE STRING "cpuinfo source directory")
  endif()

  set(CPUINFO_BUILD_TOOLS OFF CACHE BOOL "")
  set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "")
  set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "")
  set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "")
  set(CPUINFO_LIBRARY_TYPE "static" CACHE STRING "")
  if(MSVC)
    if (CAFFE2_USE_MSVC_STATIC_RUNTIME)
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
endif()
list(APPEND Caffe2_DEPENDENCY_LIBS cpuinfo)

# ---[ gflags
if(USE_GFLAGS)
  include(cmake/public/gflags.cmake)
  if (TARGET gflags)
    set(CAFFE2_USE_GFLAGS 1)
    list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS gflags)
  else()
    message(WARNING
        "gflags is not found. Caffe2 will build without gflags support but "
        "it is strongly recommended that you install gflags. Suppress this "
        "warning with -DUSE_GFLAGS=OFF")
    caffe2_update_option(USE_GFLAGS OFF)
  endif()
endif()

# ---[ Google-glog
if(USE_GLOG)
  include(cmake/public/glog.cmake)
  if (TARGET glog::glog)
    set(CAFFE2_USE_GOOGLE_GLOG 1)
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
if(BUILD_TEST)
  set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  # We will build gtest as static libs and embed it directly into the binary.
  set(BUILD_SHARED_LIBS OFF)
  # For gtest, we will simply embed it into our test binaries, so we won't
  # need to install it.
  set(BUILD_GTEST ON)
  set(INSTALL_GTEST OFF)
  # We currently don't need gmock right now.
  set(BUILD_GMOCK OFF)
  # For Windows, we will check the runtime used is correctly passed in.
  if (NOT CAFFE2_USE_MSVC_STATIC_RUNTIME)
    set(gtest_force_shared_crt ON)
  endif()
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/googletest)
  include_directories(${PROJECT_SOURCE_DIR}/third_party/googletest/googletest/include)

  # We will not need to test benchmark lib itself.
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing as we don't need it.")
  # We will not need to install benchmark since we link it statically.
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install to avoid overwriting vendor install.")
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/benchmark)
  include_directories(${PROJECT_SOURCE_DIR}/third_party/benchmark/include)

  # Recover the build shared libs option.
  set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS})
endif()

# ---[ LMDB
if(USE_LMDB)
  find_package(LMDB)
  if(LMDB_FOUND)
    include_directories(${LMDB_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${LMDB_LIBRARIES})
  else()
    message(WARNING "Not compiling with LMDB. Suppress this warning with -DUSE_LMDB=OFF")
    caffe2_update_option(USE_LMDB OFF)
  endif()
endif()

if (USE_OPENCL)
  message(INFO "USING OPENCL")
  find_package(OpenCL REQUIRED)
  include_directories(${OpenCV_INCLUDE_DIRS})
  include_directories(${PROJECT_SOURCE_DIR}/caffe2/contrib/opencl)
  list(APPEND Caffe2_DEPENDENCY_LIBS ${OpenCL_LIBRARIES})
endif()

# ---[ LevelDB
# ---[ Snappy
if(USE_LEVELDB)
  find_package(LevelDB)
  find_package(Snappy)
  if(LEVELDB_FOUND AND SNAPPY_FOUND)
    include_directories(${LevelDB_INCLUDE})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${LevelDB_LIBRARIES})
    include_directories(${Snappy_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${Snappy_LIBRARIES})
  else()
    message(WARNING "Not compiling with LevelDB. Suppress this warning with -DUSE_LEVELDB=OFF")
    caffe2_update_option(USE_LEVELDB OFF)
  endif()
endif()

# ---[ NUMA
if(USE_NUMA)
  if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    message(WARNING "NUMA is currently only supported under Linux.")
    caffe2_update_option(USE_NUMA OFF)
  else()
    find_package(Numa)
    if(NUMA_FOUND)
      include_directories(${Numa_INCLUDE_DIR})
      list(APPEND Caffe2_DEPENDENCY_LIBS ${Numa_LIBRARIES})
    else()
      message(WARNING "Not compiling with NUMA. Suppress this warning with -DUSE_NUMA=OFF")
      caffe2_update_option(USE_NUMA OFF)
    endif()
  endif()
endif()

# ---[ ZMQ
if(USE_ZMQ)
  find_package(ZMQ)
  if(ZMQ_FOUND)
    include_directories(${ZMQ_INCLUDE_DIR})
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
    include_directories(${Hiredis_INCLUDE})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${Hiredis_LIBRARIES})
  else()
    message(WARNING "Not compiling with Redis. Suppress this warning with -DUSE_REDIS=OFF")
    caffe2_update_option(USE_REDIS OFF)
  endif()
endif()


# ---[ OpenCV
if(USE_OPENCV)
  # OpenCV 3
  find_package(OpenCV 3 QUIET COMPONENTS core highgui imgproc imgcodecs videoio video)
  if(NOT OpenCV_FOUND)
    # OpenCV 2
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
  endif()
  if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${OpenCV_LIBS})
    message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
  else()
    message(WARNING "Not compiling with OpenCV. Suppress this warning with -DUSE_OPENCV=OFF")
    caffe2_update_option(USE_OPENCV OFF)
  endif()
endif()

# ---[ FFMPEG
if(USE_FFMPEG)
  find_package(FFmpeg REQUIRED)
  if (FFMPEG_FOUND)
    message("Found FFMPEG/LibAV libraries")
    include_directories(${FFMPEG_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${FFMPEG_LIBRARIES})
  else ()
    message("Not compiling with FFmpeg. Suppress this warning with -DUSE_FFMPEG=OFF")
    caffe2_update_option(USE_FFMPEG OFF)
  endif ()
endif()

# ---[ EIGEN
# Due to license considerations, we will only use the MPL2 parts of Eigen.
set(EIGEN_MPL2_ONLY 1)
find_package(Eigen3)
if(EIGEN3_FOUND)
  message(STATUS "Found system Eigen at " ${EIGEN3_INCLUDE_DIR})
else()
  message(STATUS "Did not find system Eigen. Using third party subdirectory.")
  set(EIGEN3_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/third_party/eigen)
endif()
include_directories(${EIGEN3_INCLUDE_DIR})

# ---[ Python + Numpy
if(BUILD_PYTHON)
  set(Python_ADDITIONAL_VERSIONS 2.8 2.7 2.6)
  find_package(PythonInterp 2.7)
  find_package(PythonLibs 2.7)
  find_package(NumPy REQUIRED)
  if(PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND AND NUMPY_FOUND)
    include_directories(${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIR})
    # Observers are required in the python build
    caffe2_update_option(USE_OBSERVERS ON)
  else()
    message(WARNING "Python dependencies not met. Not compiling with python. Suppress this warning with -DBUILD_PYTHON=OFF")
    caffe2_update_option(BUILD_PYTHON OFF)
  endif()
endif()

# ---[ pybind11
find_package(pybind11)
if(pybind11_FOUND)
  include_directories(${pybind11_INCLUDE_DIRS})
else()
  include_directories(${PROJECT_SOURCE_DIR}/third_party/pybind11/include)
endif()

# ---[ MPI
if(USE_MPI)
  find_package(MPI)
  if(MPI_CXX_FOUND)
    message(STATUS "MPI support found")
    message(STATUS "MPI compile flags: " ${MPI_CXX_COMPILE_FLAGS})
    message(STATUS "MPI include path: " ${MPI_CXX_INCLUDE_PATH})
    message(STATUS "MPI LINK flags path: " ${MPI_CXX_LINK_FLAGS})
    message(STATUS "MPI libraries: " ${MPI_CXX_LIBRARIES})
    include_directories(${MPI_CXX_INCLUDE_PATH})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${MPI_CXX_LIBRARIES})
    set(CMAKE_EXE_LINKER_FLAGS ${MPI_CXX_LINK_FLAGS})
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
  find_package(OpenMP)
  if(OPENMP_FOUND)
    message(STATUS "Adding " ${OpenMP_CXX_FLAGS})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  else()
    message(WARNING "Not compiling with OpenMP. Suppress this warning with -DUSE_OPENMP=OFF")
    caffe2_update_option(USE_OPENMP OFF)
  endif()
endif()


# ---[ Android specific ones
if(ANDROID)
  list(APPEND Caffe2_DEPENDENCY_LIBS log)
endif()

# ---[ CUDA
if(USE_CUDA)
  include(cmake/public/cuda.cmake)
  if(CAFFE2_FOUND_CUDA)
    # A helper variable recording the list of Caffe2 dependent librareis
    # caffe2::cudart is dealt with separately, due to CUDA_ADD_LIBRARY
    # design reason (it adds CUDA_LIBRARIES itself).
    set(Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS
        caffe2::cuda caffe2::curand caffe2::cublas caffe2::cudnn caffe2::nvrtc)
    if(USE_TENSORRT) 
      list(APPEND Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS caffe2::tensorrt) 
    endif()
  else()
    message(WARNING
        "Not compiling with CUDA. Suppress this warning with "
        "-DUSE_CUDA=OFF.")
    caffe2_update_option(USE_CUDA OFF)
  endif()
endif()

# ---[ NCCL
if(USE_NCCL)
  if(NOT USE_CUDA)
    message(WARNING "If not using cuda, one should not use NCCL either.")
    caffe2_update_option(USE_NCCL OFF)
  elseif(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    message(WARNING "NCCL is currently only supported under Linux.")
    caffe2_update_option(USE_NCCL OFF)
  else()
    include("cmake/External/nccl.cmake")
    list(APPEND Caffe2_CUDA_DEPENDENCY_LIBS __caffe2_nccl)
  endif()
endif()

# ---[ CUB
if(USE_CUDA)
  find_package(CUB)
  if(CUB_FOUND)
    include_directories(${CUB_INCLUDE_DIRS})
  else()
    include_directories(${PROJECT_SOURCE_DIR}/third_party/cub)
  endif()
endif()

if(USE_GLOO)
  if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    message(WARNING "Gloo can only be used on Linux.")
    caffe2_update_option(USE_GLOO OFF)
  elseif(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(WARNING "Gloo can only be used on 64-bit systems.")
    caffe2_update_option(USE_GLOO OFF)
  else()
    set(Gloo_USE_CUDA ${USE_CUDA})
    find_package(Gloo)
    if(Gloo_FOUND)
      include_directories(${Gloo_INCLUDE_DIRS})
      list(APPEND Caffe2_DEPENDENCY_LIBS gloo)
    else()
      set(GLOO_INSTALL OFF CACHE BOOL "" FORCE)
      set(GLOO_STATIC_OR_SHARED STATIC CACHE STRING "" FORCE)

      # Temporarily override variables to avoid building Gloo tests/benchmarks
      set(__BUILD_TEST ${BUILD_TEST})
      set(__BUILD_BENCHMARK ${BUILD_BENCHMARK})
      set(BUILD_TEST OFF)
      set(BUILD_BENCHMARK OFF)
      add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/gloo)
      # Here is a little bit hacky. We have to put PROJECT_BINARY_DIR in front
      # of PROJECT_SOURCE_DIR with/without conda system. The reason is that
      # gloo generates a new config.h in the binary diretory.
      include_directories(BEFORE SYSTEM ${PROJECT_SOURCE_DIR}/third_party/gloo)
      include_directories(BEFORE SYSTEM ${PROJECT_BINARY_DIR}/third_party/gloo)
      set(BUILD_TEST ${__BUILD_TEST})
      set(BUILD_BENCHMARK ${__BUILD_BENCHMARK})

      # Add explicit dependency if NCCL is built from third_party.
      # Without dependency, make -jN with N>1 can fail if the NCCL build
      # hasn't finished when CUDA targets are linked.
      if(NCCL_EXTERNAL)
        add_dependencies(gloo_cuda nccl_external)
      endif()
    endif()
    # Pick the right dependency depending on USE_CUDA
    list(APPEND Caffe2_DEPENDENCY_LIBS gloo)
    if(USE_CUDA)
      list(APPEND Caffe2_CUDA_DEPENDENCY_LIBS gloo_cuda)
    endif()
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

if (USE_MOBILE_OPENGL)
  if (ANDROID)
    list(APPEND Caffe2_DEPENDENCY_LIBS EGL GLESv2)
  elseif (IOS)
    message(STATUS "TODO item for adding ios opengl dependency")
  else()
    message(WARNING "mobile opengl is only used in android or ios builds.")
    caffe2_update_option(USE_MOBILE_OPENGL OFF)
  endif()
endif()

# ---[ ARM Compute Library: check compatibility.
if (USE_ACL)
  if (NOT ANDROID)
    message(WARNING "ARM Compute Library is only supported for Android builds.")
    caffe2_update_option(USE_ACL OFF)
  else()
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^armv")
      # 32-bit ARM (armv7, armv7-a, armv7l, etc)
      set(ACL_ARCH "armv7a")
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
      # 64-bit ARM
      set(ACL_ARCH "arm64-v8a")
    else()
      message(WARNING "ARM Compute Library is only supported for ARM/ARM64 builds.")
      caffe2_update_option(USE_ACL OFF)
    endif()
  endif()
endif()

# ---[ ARM Compute Library: build the target.
if (USE_ACL)
  list(APPEND ARM_COMPUTE_INCLUDE_DIRS "third_party/ComputeLibrary/")
  list(APPEND ARM_COMPUTE_INCLUDE_DIRS "third_party/ComputeLibrary/include")
  include_directories(${ARM_COMPUTE_INCLUDE_DIRS})
  string (REPLACE ";" " -I" ANDROID_STL_INCLUDE_FLAGS "-I${ANDROID_STL_INCLUDE_DIRS}")
  set (ARM_COMPUTE_SRC_DIR "${PROJECT_SOURCE_DIR}/third_party/ComputeLibrary/")
  set (ARM_COMPUTE_LIB "${CMAKE_CURRENT_BINARY_DIR}/libarm_compute.a")
  set (ARM_COMPUTE_CORE_LIB "${CMAKE_CURRENT_BINARY_DIR}/libarm_compute_core.a")
  set (ARM_COMPUTE_LIBS ${ARM_COMPUTE_LIB} ${ARM_COMPUTE_CORE_LIB})

  add_custom_command(
      OUTPUT ${ARM_COMPUTE_LIBS}
      COMMAND
        /bin/sh -c "export PATH=\"$PATH:$(dirname ${CMAKE_CXX_COMPILER})\" && \
        scons -C \"${ARM_COMPUTE_SRC_DIR}\" -Q \
          examples=no validation_tests=no benchmark_tests=no standalone=yes \
          embed_kernels=yes opencl=no gles_compute=yes \
          os=android arch=${ACL_ARCH} \
          extra_cxx_flags=\"${ANDROID_CXX_FLAGS} ${ANDROID_STL_INCLUDE_FLAGS}\"" &&
        /bin/sh -c "cp ${ARM_COMPUTE_SRC_DIR}/build/libarm_compute-static.a ${CMAKE_CURRENT_BINARY_DIR}/libarm_compute.a" &&
        /bin/sh -c "cp ${ARM_COMPUTE_SRC_DIR}/build/libarm_compute_core-static.a ${CMAKE_CURRENT_BINARY_DIR}/libarm_compute_core.a" &&
        /bin/sh -c "rm -r ${ARM_COMPUTE_SRC_DIR}/build"
      COMMENT "Building ARM compute library" VERBATIM)
  add_custom_target(arm_compute_build ALL DEPENDS ${ARM_COMPUTE_LIBS})

  add_library(arm_compute_core STATIC IMPORTED)
  add_dependencies(arm_compute_core arm_compute_build)
  set_property(TARGET arm_compute_core PROPERTY IMPORTED_LOCATION ${ARM_COMPUTE_CORE_LIB})

  add_library(arm_compute STATIC IMPORTED)
  add_dependencies(arm_compute arm_compute_build)
  set_property(TARGET arm_compute PROPERTY IMPORTED_LOCATION ${ARM_COMPUTE_LIB})

  list(APPEND Caffe2_DEPENDENCY_LIBS arm_compute arm_compute_core)
endif()

if (USE_SNPE AND ANDROID)
  if (SNPE_LOCATION AND SNPE_HEADERS)
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

if (USE_METAL)
  if (NOT IOS)
    message(WARNING "Metal is only used in ios builds.")
    caffe2_update_option(USE_METAL OFF)
  endif()
endif()

if (USE_NNAPI AND NOT ANDROID)
  message(WARNING "NNApi is only used in android builds.")
  caffe2_update_option(USE_NNAPI OFF)
endif()

if (USE_ATEN)
  list(APPEND Caffe2_DEPENDENCY_LIBS aten_op_header_gen ATen)
  include_directories(${PROJECT_BINARY_DIR}/caffe2/contrib/aten/aten/src/ATen)
  include_directories(${PROJECT_SOURCE_DIR}/aten/src)
  include_directories(${PROJECT_BINARY_DIR}/caffe2/contrib/aten)
endif()

if (USE_ZSTD)
  list(APPEND Caffe2_DEPENDENCY_LIBS libzstd_static)
  include_directories(${PROJECT_SOURCE_DIR}/third_party/zstd/lib)
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/zstd/build/cmake)
  set_property(TARGET libzstd_static PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

# ---[ Onnx
SET(ONNX_NAMESPACE "onnx_c2")
if(EXISTS "${CAFFE2_CUSTOM_PROTOC_EXECUTABLE}")
  set(ONNX_CUSTOM_PROTOC_EXECUTABLE ${CAFFE2_CUSTOM_PROTOC_EXECUTABLE})
endif()
set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
# We will build onnx as static libs and embed it directly into the binary.
set(BUILD_SHARED_LIBS OFF)
set(ONNX_USE_MSVC_STATIC_RUNTIME ${CAFFE2_USE_MSVC_STATIC_RUNTIME})
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/onnx)
include_directories(${ONNX_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DONNX_NAMESPACE=${ONNX_NAMESPACE}")
# In mobile build we care about code size, and so we need drop
# everything (e.g. checker, optimizer) in onnx but the pb definition.
if (ANDROID OR IOS)
  caffe2_interface_library(onnx_proto onnx_library)
else()
  caffe2_interface_library(onnx onnx_library)
endif()
list(APPEND Caffe2_DEPENDENCY_WHOLE_LINK_LIBS onnx_library)
# Recover the build shared libs option.
set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS})

# --[ TensorRT integration with onnx-trt
if (USE_TENSORRT) 
  set(CMAKE_CUDA_COMPILER ${CUDA_NVCC_EXECUTABLE})
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/onnx-trt)
  include_directories("${PROJECT_SOURCE_DIR}/third_party/onnx-trt")
  caffe2_interface_library(onnx2trt_importer_static onnx_trt_library)
  list(APPEND Caffe2_DEPENDENCY_WHOLE_LINK_LIBS onnx_trt_library)
  set(CAFFE2_USE_TRT 1)
endif()
