# This list is required for static linking and exported to Caffe2Config.cmake
set(Caffe2_DEPENDENCY_LIBS "")
set(Caffe2_CUDA_DEPENDENCY_LIBS "")
set(Caffe2_PYTHON_DEPENDENCY_LIBS "")
set(Caffe2_EXTERNAL_DEPENDENCIES "")

# ---[ Custom Protobuf
include("cmake/ProtoBuf.cmake")

# ---[ Threads
if(USE_THREADS)
  find_package(Threads REQUIRED)
  list(APPEND Caffe2_DEPENDENCY_LIBS ${CMAKE_THREAD_LIBS_INIT})
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
  caffe2_include_directories(${ATLAS_INCLUDE_DIRS})
  list(APPEND Caffe2_DEPENDENCY_LIBS ${ATLAS_LIBRARIES})
  list(APPEND Caffe2_DEPENDENCY_LIBS cblas)
elseif(BLAS STREQUAL "OpenBLAS")
  find_package(OpenBLAS REQUIRED)
  caffe2_include_directories(${OpenBLAS_INCLUDE_DIR})
  list(APPEND Caffe2_DEPENDENCY_LIBS ${OpenBLAS_LIB})
elseif(BLAS STREQUAL "MKL")
  find_package(MKL REQUIRED)
  caffe2_include_directories(${MKL_INCLUDE_DIR})
  list(APPEND Caffe2_DEPENDENCY_LIBS ${MKL_LIBRARIES})
  set(CAFFE2_USE_MKL 1)
elseif(BLAS STREQUAL "vecLib")
  find_package(vecLib REQUIRED)
  caffe2_include_directories(${vecLib_INCLUDE_DIR})
  list(APPEND Caffe2_DEPENDENCY_LIBS ${vecLib_LINKER_LIBS})
else()
  message(FATAL_ERROR "Unrecognized blas option:" ${BLAS})
endif()

# ---[ NNPACK
if(USE_NNPACK)
  include("cmake/External/nnpack.cmake")
  if(NNPACK_FOUND)
    caffe2_include_directories(${NNPACK_INCLUDE_DIRS})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${NNPACK_LIBRARIES})
    if(TARGET nnpack)
      # ---[ NNPACK is being built together with Caffe2: explicitly specify dependency
      list(APPEND Caffe2_EXTERNAL_DEPENDENCIES nnpack)
    endif()
  else()
    message(WARNING "Not compiling with NNPACK. Suppress this warning with -DUSE_NNPACK=OFF")
    set(USE_NNPACK OFF)
  endif()
endif()

# ---[ gflags
if(USE_GFLAGS)
  include(cmake/public/gflags.cmake)
  if (TARGET gflags)
    set(CAFFE2_USE_GFLAGS 1)
    list(APPEND Caffe2_DEPENDENCY_LIBS gflags)
  else()
    message(WARNING
        "gflags is not found. Caffe2 will build without gflags support but "
        "it is strongly recommended that you install gflags. Suppress this "
        "warning with -DUSE_GFLAGS=OFF")
    set(USE_GFLAGS OFF)
  endif()
endif()

# ---[ Google-glog
if(USE_GLOG)
  include(cmake/public/glog.cmake)
  if (TARGET glog::glog)
    set(CAFFE2_USE_GOOGLE_GLOG 1)
    list(APPEND Caffe2_DEPENDENCY_LIBS glog::glog)
  else()
    message(WARNING
        "glog is not found. Caffe2 will build without glog support but it is "
        "strongly recommended that you install glog. Suppress this warning "
        "with -DUSE_GLOG=OFF")
    set(USE_GLOG OFF)
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
  caffe2_include_directories(${PROJECT_SOURCE_DIR}/third_party/googletest/googletest/include)

  # We will not need to test benchmark lib itself.
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing as we don't need it.")
  # We will not need to install benchmark since we link it statically.
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install to avoid overwriting vendor install.")
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/benchmark)
  caffe2_include_directories(${PROJECT_SOURCE_DIR}/third_party/benchmark/include)

  # Recover the build shared libs option.
  set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS})
endif()

# ---[ LMDB
if(USE_LMDB)
  find_package(LMDB)
  if(LMDB_FOUND)
    caffe2_include_directories(${LMDB_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${LMDB_LIBRARIES})
  else()
    message(WARNING "Not compiling with LMDB. Suppress this warning with -DUSE_LMDB=OFF")
    set(USE_LMDB OFF)
  endif()
endif()

# ---[ LevelDB
# ---[ Snappy
if(USE_LEVELDB)
  find_package(LevelDB)
  find_package(Snappy)
  if(LEVELDB_FOUND AND SNAPPY_FOUND)
    caffe2_include_directories(${LevelDB_INCLUDE})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${LevelDB_LIBRARIES})
    caffe2_include_directories(${Snappy_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${Snappy_LIBRARIES})
  else()
    message(WARNING "Not compiling with LevelDB. Suppress this warning with -DUSE_LEVELDB=OFF")
    set(USE_LEVELDB OFF)
  endif()
endif()

# ---[ Rocksdb
if(USE_ROCKSDB)
  find_package(RocksDB)
  if(ROCKSDB_FOUND)
    caffe2_include_directories(${RocksDB_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${RocksDB_LIBRARIES})
  else()
    message(WARNING "Not compiling with RocksDB. Suppress this warning with -DUSE_ROCKSDB=OFF")
    set(USE_ROCKSDB OFF)
  endif()
endif()

# ---[ ZMQ
if(USE_ZMQ)
  find_package(ZMQ)
  if(ZMQ_FOUND)
    caffe2_include_directories(${ZMQ_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${ZMQ_LIBRARIES})
  else()
    message(WARNING "Not compiling with ZMQ. Suppress this warning with -DUSE_ZMQ=OFF")
    set(USE_ZMQ OFF)
  endif()
endif()

# ---[ Redis
if(USE_REDIS)
  find_package(Hiredis)
  if(HIREDIS_FOUND)
    caffe2_include_directories(${Hiredis_INCLUDE})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${Hiredis_LIBRARIES})
  else()
    message(WARNING "Not compiling with Redis. Suppress this warning with -DUSE_REDIS=OFF")
    set(USE_REDIS OFF)
  endif()
endif()


# ---[ OpenCV
if(USE_OPENCV)
  # OpenCV 3
  find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
  if(NOT OpenCV_FOUND)
    # OpenCV 2
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
  endif()
  if(OpenCV_FOUND)
    caffe2_include_directories(${OpenCV_INCLUDE_DIRS})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${OpenCV_LIBS})
    message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
  else()
    message(WARNING "Not compiling with OpenCV. Suppress this warning with -DUSE_OPENCV=OFF")
    set(USE_OPENCV OFF)
  endif()
endif()

# ---[ FFMPEG
if(USE_FFMPEG)
  find_package(FFmpeg REQUIRED)
  if (FFMPEG_FOUND)
    message("Found FFMPEG/LibAV libraries")
    caffe2_include_directories(${FFMPEG_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${FFMPEG_LIBRARIES})
  else ()
    message("Not compiling with FFmpeg. Suppress this warning with -DUSE_FFMPEG=OFF")
    set(USE_FFMPEG OFF)
  endif ()
endif()

# ---[ EIGEN
# Due to license considerations, we will only use the MPL2 parts of Eigen.
set(EIGEN_MPL2_ONLY 1)
find_package(Eigen3)
if(EIGEN3_FOUND)
  message(STATUS "Found system Eigen at " ${EIGEN3_INCLUDE_DIR})
  caffe2_include_directories(${EIGEN3_INCLUDE_DIR})
else()
  message(STATUS "Did not find system Eigen. Using third party subdirectory.")
  caffe2_include_directories(${PROJECT_SOURCE_DIR}/third_party/eigen)
endif()

# ---[ Python + Numpy
if(BUILD_PYTHON)
  set(Python_ADDITIONAL_VERSIONS 2.8 2.7 2.6)
  find_package(PythonInterp 2.7)
  find_package(PythonLibs 2.7)
  find_package(NumPy REQUIRED)
  # Observers are required in the python build
  set(USE_OBSERVERS ON)
  if(PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND AND NUMPY_FOUND)
    caffe2_include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
  else()
    message(WARNING "Python dependencies not met. Not compiling with python. Suppress this warning with -DBUILD_PYTHON=OFF")
    set(BUILD_PYTHON OFF)
  endif()
endif()

# ---[ pybind11
find_package(pybind11)
if(pybind11_FOUND)
  caffe2_include_directories(${pybind11_INCLUDE_DIRS})
else()
  caffe2_include_directories(${PROJECT_SOURCE_DIR}/third_party/pybind11/include)
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
    caffe2_include_directories(${MPI_CXX_INCLUDE_PATH})
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
    set(USE_MPI OFF)
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
    set(USE_OPENMP OFF)
  endif()
endif()


# ---[ Android specific ones
if(ANDROID)
  list(APPEND Caffe2_DEPENDENCY_LIBS log)
endif()

# ---[ CUDA
if(USE_CUDA)
  include(cmake/Cuda.cmake)
  if(HAVE_CUDA)
    # CUDA 9.x requires GCC version <= 6
    if ((CUDA_VERSION VERSION_EQUAL   9.0) OR
        (CUDA_VERSION VERSION_GREATER 9.0  AND CUDA_VERSION VERSION_LESS 10.0))
      if (CMAKE_C_COMPILER_ID STREQUAL "GNU" AND
          NOT CMAKE_C_COMPILER_VERSION VERSION_LESS 7.0 AND
          CUDA_HOST_COMPILER STREQUAL CMAKE_C_COMPILER)
        message(FATAL_ERROR
          "CUDA ${CUDA_VERSION} is not compatible with GCC version >= 7. "
          "Use the following option to use another version (for example): \n"
          "  -DCUDA_HOST_COMPILER=/usr/bin/gcc-6\n")
      endif()
    # CUDA 8.0 requires GCC version <= 5
    elseif (CUDA_VERSION VERSION_EQUAL 8.0)
      if (CMAKE_C_COMPILER_ID STREQUAL "GNU" AND
          NOT CMAKE_C_COMPILER_VERSION VERSION_LESS 6.0 AND
          CUDA_HOST_COMPILER STREQUAL CMAKE_C_COMPILER)
        message(FATAL_ERROR
          "CUDA 8.0 is not compatible with GCC version >= 6. "
          "Use the following option to use another version (for example): \n"
          "  -DCUDA_HOST_COMPILER=/usr/bin/gcc-5\n")
      endif()
    endif()
  endif()
  # ---[ CUDNN
  if(HAVE_CUDA)
    find_package(CuDNN REQUIRED)
    if(CUDNN_FOUND)
      caffe2_include_directories(${CUDNN_INCLUDE_DIRS})
      list(APPEND Caffe2_CUDA_DEPENDENCY_LIBS ${CUDNN_LIBRARIES})
    endif()
  else()
    message(WARNING "Not compiling with CUDA. Suppress this warning with -DUSE_CUDA=OFF")
    set(USE_CUDA OFF)
  endif()
endif()

# ---[ NCCL
if(USE_NCCL)
  if(NOT USE_CUDA)
    message(WARNING "If not using cuda, one should not use NCCL either.")
    set(USE_NCCL OFF)
  elseif(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    message(WARNING "NCCL is currently only supported under Linux.")
    set(USE_NCCL OFF)
  else()
    include("cmake/External/nccl.cmake")
    caffe2_include_directories(${NCCL_INCLUDE_DIRS})
    message(STATUS "NCCL: ${NCCL_LIBRARIES}")
    list(APPEND Caffe2_CUDA_DEPENDENCY_LIBS ${NCCL_LIBRARIES})
  endif()
endif()

# ---[ CUB
if(USE_CUDA)
  find_package(CUB)
  if(CUB_FOUND)
    caffe2_include_directories(${CUB_INCLUDE_DIRS})
  else()
    caffe2_include_directories(${PROJECT_SOURCE_DIR}/third_party/cub)
  endif()
endif()

if(USE_GLOO)
  if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    message(WARNING "Gloo can only be used on Linux.")
    set(USE_GLOO OFF)
  elseif(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(WARNING "Gloo can only be used on 64-bit systems.")
    set(USE_GLOO OFF)
  else()
    set(Gloo_USE_CUDA ${USE_CUDA})
    find_package(Gloo)
    if(Gloo_FOUND)
      caffe2_include_directories(${Gloo_INCLUDE_DIRS})
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
    set(USE_MOBILE_OPENGL OFF)
  endif()
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
    set(USE_SNPE OFF)
  endif()
endif()

if (USE_METAL)
  if (NOT IOS)
    message(WARNING "Metal is only used in ios builds.")
    set(USE_METAL OFF)
  endif()
endif()

if (USE_ATEN)
  list(APPEND Caffe2_EXTERNAL_DEPENDENCIES aten_build)
  list(APPEND Caffe2_DEPENDENCY_LIBS ATen)
  caffe2_include_directories(${PROJECT_BINARY_DIR}/caffe2/contrib/aten/aten/src/ATen)
  caffe2_include_directories(${PROJECT_SOURCE_DIR}/third_party/aten/src)
  caffe2_include_directories(${PROJECT_BINARY_DIR}/caffe2/contrib/aten)
endif()

if (USE_ZSTD)
  list(APPEND Caffe2_DEPENDENCY_LIBS libzstd_static)
  caffe2_include_directories(${PROJECT_SOURCE_DIR}/third_party/zstd/lib)
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/zstd/build/cmake)
endif()
