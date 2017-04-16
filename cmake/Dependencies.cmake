# This list is required for static linking and exported to Caffe2Config.cmake
set(Caffe2_DEPENDENCY_LIBS "")
set(Caffe2_PYTHON_DEPENDENCY_LIBS "")

# ---[ Custom Protobuf
include("cmake/ProtoBuf.cmake")

# ---[ Threads
if (USE_THREADS)
  find_package(Threads REQUIRED)
  list(APPEND Caffe2_DEPENDENCY_LIBS ${CMAKE_THREAD_LIBS_INIT})
endif()

# ---[ protobuf
if (USE_LITE_PROTO)
  add_definitions(-DCAFFE2_USE_LITE_PROTO)
endif()

# ---[ BLAS
set(BLAS "Eigen" CACHE STRING "Selected BLAS library")
set_property(CACHE BLAS PROPERTY STRINGS "Eigen;ATLAS;OpenBLAS;MKL;vecLib")
message(STATUS "The BLAS backend of choice:" ${BLAS})

if(BLAS STREQUAL "Eigen")
  # Eigen is header-only and we do not have any dependent libraries
  add_definitions(-DCAFFE2_USE_EIGEN_FOR_BLAS)
elseif(BLAS STREQUAL "ATLAS")
  find_package(Atlas REQUIRED)
  include_directories(SYSTEM ${ATLAS_INCLUDE_DIRS})
  list(APPEND Caffe2_DEPENDENCY_LIBS ${ATLAS_LIBRARIES})
  list(APPEND Caffe2_DEPENDENCY_LIBS cblas)
elseif(BLAS STREQUAL "OpenBLAS")
  find_package(OpenBLAS REQUIRED)
  include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIR})
  list(APPEND Caffe2_DEPENDENCY_LIBS ${OpenBLAS_LIB})
  list(APPEND Caffe2_DEPENDENCY_LIBS cblas)
elseif(BLAS STREQUAL "MKL")
  find_package(MKL REQUIRED)
  include_directories(SYSTEM ${MKL_INCLUDE_DIR})
  list(APPEND Caffe2_DEPENDENCY_LIBS ${MKL_LIBRARIES})
  add_definitions(-DCAFFE2_USE_MKL)
elseif(BLAS STREQUAL "vecLib")
  find_package(vecLib REQUIRED)
  include_directories(SYSTEM ${vecLib_INCLUDE_DIR})
  list(APPEND Caffe2_DEPENDENCY_LIBS ${vecLib_LINKER_LIBS})
else()
  message(FATAL_ERROR "Unrecognized blas option:" ${BLAS})
endif()

# ---[ NNPACK
if (USE_NNPACK)
  include("cmake/External/nnpack.cmake")
  if (NNPACK_FOUND)
    include_directories(SYSTEM ${NNPACK_INCLUDE_DIRS})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${NNPACK_LIBRARIES})
  else()
    message(WARNING "Not compiling with NNPACK. Suppress this warning with -DUSE_NNPACK=OFF")
    set(USE_NNPACK OFF)
  endif()
endif()

# ---[ Google-glog
if (USE_GLOG)
  include("cmake/External/glog.cmake")
  if (GLOG_FOUND)
    add_definitions(-DCAFFE2_USE_GOOGLE_GLOG)
    include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${GLOG_LIBRARIES})
  else()
    message(WARNING "Not compiling with glog. Suppress this warning with -DUSE_GLOG=OFF")
    set(USE_GLOG OFF)
  endif()
endif()

# ---[ Google-gflags
if (USE_GFLAGS)
  include("cmake/External/gflags.cmake")
  if (GFLAGS_FOUND)
    add_definitions(-DCAFFE2_USE_GFLAGS)
    include_directories(SYSTEM ${GFLAGS_INCLUDE_DIRS})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${GFLAGS_LIBRARIES})
  else()
    message(WARNING "Not compiling with gflags. Suppress this warning with -DUSE_GFLAGS=OFF")
    set(USE_GFLAGS OFF)
  endif()
endif()

# ---[ Googletest and benchmark
if (BUILD_TEST)
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/googletest)
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/googletest/googletest/include)
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/benchmark)
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/benchmark/include)
endif()

# ---[ LMDB
if(USE_LMDB)
  find_package(LMDB)
  if (LMDB_FOUND)
    include_directories(SYSTEM ${LMDB_INCLUDE_DIR})
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
  if (LEVELDB_FOUND AND SNAPPY_FOUND)
    include_directories(SYSTEM ${LevelDB_INCLUDE})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${LevelDB_LIBRARIES})
    include_directories(SYSTEM ${Snappy_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${Snappy_LIBRARIES})
  else()
    message(WARNING "Not compiling with LevelDB. Suppress this warning with -DUSE_LEVELDB=OFF")
    set(USE_LEVELDB OFF)
  endif()
endif()

# ---[ Rocksdb
if(USE_ROCKSDB)
  find_package(RocksDB)
  if (ROCKSDB_FOUND)
    include_directories(SYSTEM ${RocksDB_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${RocksDB_LIBRARIES})
  else()
    message(WARNING "Not compiling with RocksDB. Suppress this warning with -DUSE_ROCKSDB=OFF")
    set(USE_ROCKSDB OFF)
  endif()
endif()

# ---[ ZMQ
if(USE_ZMQ)
  find_package(ZMQ)
  if (ZMQ_FOUND)
    include_directories(SYSTEM ${ZMQ_INCLUDE_DIR})
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
    include_directories(SYSTEM ${Hiredis_INCLUDE})
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
  if (OpenCV_FOUND)
    include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${OpenCV_LIBS})
    message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
  else()
    message(WARNING "Not compiling with OpenCV. Suppress this warning with -DUSE_OPENCV=OFF")
    set(USE_OPENCV OFF)
  endif()
endif()

# ---[ EIGEN
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/eigen)
add_definitions(-DEIGEN_MPL2_ONLY)

# ---[ Python + Numpy
if (BUILD_PYTHON)
  find_package(PythonInterp 2.7)
  find_package(PythonLibs 2.7)
  find_package(NumPy REQUIRED)
  if (PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND AND NUMPY_FOUND)
    include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
    list(APPEND Caffe2_PYTHON_DEPENDENCY_LIBS ${PYTHON_LIBRARIES})
  else()
    message(WARNING "Python dependencies not met. Not compiling with python. Suppress this warning with -DBUILD_PYTHON=OFF")
    set(BUILD_PYTHON OFF)
  endif()
endif()

# ---[ pybind11
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/pybind11/include)

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
    set(CMAKE_EXE_LINKER_FLAGS ${MPI_CXX_LINK_FLAGS})
    find_program(OMPI_INFO
      NAMES ompi_info
      HINTS ${MPI_CXX_LIBRARIES}/../bin)
    if (OMPI_INFO)
      execute_process(COMMAND ${OMPI_INFO}
                      OUTPUT_VARIABLE _output)
      if (_output MATCHES "smcuda")
        message(STATUS "Found OpenMPI with CUDA support built.")
      else()
        message(WARNING "OpenMPI found, but it is not built with CUDA support.")
        add_definitions(-DCAFFE2_FORCE_FALLBACK_CUDA_MPI)
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
  if(OpenMP_FOUND)
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
if (ANDROID)
  list(APPEND Caffe2_DEPENDENCY_LIBS log)
endif()

# ---[ CUDA
if (USE_CUDA)
  include(cmake/Cuda.cmake)
  # ---[ CUDNN
  if(HAVE_CUDA)
    find_package(CuDNN REQUIRED)
    if(CUDNN_FOUND)
      include_directories(SYSTEM ${CUDNN_INCLUDE_DIRS})
      list(APPEND Caffe2_DEPENDENCY_LIBS ${CUDNN_LIBRARIES})
    endif()
  else()
    message(WARNING "Not compiling with CUDA. Suppress this warning with -DUSE_CUDA=OFF")
    set(USE_CUDA OFF)
  endif()
endif()

# ---[ NCCL
if(USE_NCCL)
  if (NOT USE_CUDA)
    message(WARNING "If not using cuda, one should not use NCCL either.")
    set(USE_NCCL OFF)
  elseif (NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    message(WARNING "NCCL is currently only supported under Linux.")
    set(USE_NCCL OFF)
  else()
    include("cmake/External/nccl.cmake")
    include_directories(SYSTEM ${NCCL_INCLUDE_DIRS})
    message(STATUS "NCCL: ${NCCL_LIBRARIES}")
    list(APPEND Caffe2_DEPENDENCY_LIBS ${NCCL_LIBRARIES})
  endif()
endif()

# ---[ CUB
if(USE_CUDA)
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/cub)
endif()

# ---[ CNMEM
if(USE_CNMEM)
  if (NOT USE_CUDA)
    message(WARNING "If not using cuda, one should not use CNMEM either.")
    set(USE_CNMEM OFF)
  else()
    add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/cnmem)
    include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/cnmem/include)
    list(APPEND Caffe2_DEPENDENCY_LIBS "${CMAKE_CURRENT_BINARY_DIR}/third_party/cnmem/libcnmem.so")
    add_definitions(-DCAFFE2_USE_CNMEM)
  endif()
endif()

if(USE_GLOO)
  if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    message(WARNING "Gloo can only be used on Linux.")
    set(USE_GLOO OFF)
  else()
    set(GLOO_INSTALL OFF CACHE BOOL "" FORCE)
    set(GLOO_STATIC_OR_SHARED STATIC CACHE STRING "" FORCE)

    # Temporarily override variables to avoid building Gloo tests/benchmarks
    set(__BUILD_TEST ${BUILD_TEST})
    set(__BUILD_BENCHMARK ${BUILD_BENCHMARK})
    set(BUILD_TEST OFF)
    set(BUILD_BENCHMARK OFF)
    add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/gloo)
    include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/gloo)
    set(BUILD_TEST ${__BUILD_TEST})
    set(BUILD_BENCHMARK ${__BUILD_BENCHMARK})

    # Pick the right dependency depending on USE_CUDA
    if(NOT USE_CUDA)
      list(APPEND Caffe2_DEPENDENCY_LIBS gloo)
    else()
      list(APPEND Caffe2_DEPENDENCY_LIBS gloo_cuda)
    endif()
  endif()
endif()
