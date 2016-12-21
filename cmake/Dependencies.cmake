# This list is required for static linking and exported to Caffe2Config.cmake
set(Caffe2_LINKER_LIBS "")

# ---[ Custom Protobuf
include("cmake/External/Protobuf.cmake")

# ---[ Threads
find_package(Threads REQUIRED)
list(APPEND Caffe2_LINKER_LIBS ${CMAKE_THREAD_LIBS_INIT})

# ---[ BLAS
set(BLAS "Atlas" CACHE STRING "Selected BLAS library")
set_property(CACHE BLAS PROPERTY STRINGS "Atlas;OpenBLAS;MKL")
list(APPEND Caffe2_LINKER_LIBS cblas)

if(BLAS STREQUAL "Atlas")
  find_package(Atlas REQUIRED)
  include_directories(SYSTEM ${ATLAS_INCLUDE_DIRS})
  list(APPEND Caffe2_LINKER_LIBS ${ATLAS_LIBRARIES})
elseif(BLAS STREQUAL "OpenBLAS")
  find_package(OpenBLAS REQUIRED)
  include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIR})
  list(APPEND Caffe2_LINKER_LIBS ${OpenBLAS_LIB})
elseif(BLAS STREQUAL "MKL")
  find_package(MKL REQUIRED)
  include_directories(SYSTEM ${MKL_INCLUDE_DIR})
  list(APPEND Caffe2_LINKER_LIBS ${MKL_LIBRARIES})
endif()

# ---[ Google-glog
include("cmake/External/glog.cmake")
add_definitions(-DCAFFE2_USE_GOOGLE_GLOG)
include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
list(APPEND Caffe2_LINKER_LIBS ${GLOG_LIBRARIES})

# ---[ Google-gflags
include("cmake/External/gflags.cmake")
include_directories(SYSTEM ${GFLAGS_INCLUDE_DIRS})
list(APPEND Caffe2_LINKER_LIBS ${GFLAGS_LIBRARIES})

# ---[ Googletest
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/googletest)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/third_party/googletest/googletest/include)

# ---[ LMDB
if(USE_LMDB)
  find_package(LMDB REQUIRED)
  include_directories(SYSTEM ${LMDB_INCLUDE_DIR})
  list(APPEND Caffe2_LINKER_LIBS ${LMDB_LIBRARIES})
endif()

# ---[ LevelDB
if(USE_LEVELDB)
  find_package(LevelDB REQUIRED)
  include_directories(SYSTEM ${LevelDB_INCLUDE})
  list(APPEND Caffe2_LINKER_LIBS ${LevelDB_LIBRARIES})
endif()

# ---[ Snappy
if(USE_LEVELDB)
  find_package(Snappy REQUIRED)
  include_directories(SYSTEM ${Snappy_INCLUDE_DIR})
  list(APPEND Caffe2_LINKER_LIBS ${Snappy_LIBRARIES})
endif()

# ---[ OpenCV
if(USE_OPENCV)
  # OpenCV 3
  find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
  if(NOT OpenCV_FOUND)
    # OpenCV 2
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
  endif()
  include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
  list(APPEND Caffe2_LINKER_LIBS ${OpenCV_LIBS})
  message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
endif()

# ---[ EIGEN
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/third_party/eigen)

# ---[ Python + Numpy
find_package(PythonInterp 2.7)
find_package(PythonLibs 2.7)
find_package(NumPy REQUIRED)

include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})
list(APPEND Caffe2_LINKER_LIBS ${PYTHON_LIBRARIES})

# ---[ pybind11
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/third_party/pybind11/include)

# ---[ MPI
if(USE_MPI)
  find_package(MPI)
  if(MPI_CXX_FOUND)
    include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
    list(APPEND Caffe2_LINKER_LIBS ${MPI_CXX_LIBRARIES})
    set(CMAKE_EXE_LINKER_FLAGS ${MPI_CXX_LINK_FLAGS})
  endif()
endif()

# ---[ OpenMP
find_package(OpenMP)
if(OpenMP_FOUND)
  # set(CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS})
  list(APPEND Caffe2_LINKER_LIBS ${OpenMP_CXX_FLAGS})
endif()

# ---[ CUDA
include(cmake/Cuda.cmake)


# ---[ CUDNN
if(HAVE_CUDA)
  find_package(CuDNN REQUIRED)
  if(CUDNN_FOUND)
    include_directories(SYSTEM ${CUDNN_INCLUDE_DIRS})
    list(APPEND Caffe2_LINKER_LIBS ${CUDNN_LIBRARIES})
  endif()
endif()

# ---[ NCCL
if(HAVE_CUDA)
  include("cmake/External/nccl.cmake")
  include_directories(SYSTEM ${NCCL_INCLUDE_DIRS})
  message(STATUS "NCCL: ${NCCL_LIBRARIES}")
  list(APPEND Caffe2_LINKER_LIBS ${NCCL_LIBRARIES})
endif()

# ---[ CUB
if(HAVE_CUDA)
  include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/third_party/cub)
endif()

# ---[ CNMEM
if(HAVE_CUDA)
  add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/cnmem)
  include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/third_party/cnmem/include)
  # message(STATUS "cnmem: ${CMAKE_SOURCE_DIR}/third_party/cnmem/libcnmem.so")
  # message(STATUS "${CMAKE_CURRENT_BINARY_DIR}")
  list(APPEND Caffe2_LINKER_LIBS "${CMAKE_CURRENT_BINARY_DIR}/third_party/cnmem/libcnmem.so")
endif()
