#cmake_minimum_required(VERSION 2.8.12)
cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)

project(Caffe2 CXX)

# TODO(bwasti): versioning

# Useful functions.
function (exclude OUTPUT INPUT)
set(EXCLUDES ${ARGN})
foreach(EXCLUDE ${EXCLUDES})
        list(REMOVE_ITEM INPUT "${EXCLUDE}")
endforeach()
set(${OUTPUT} ${INPUT} PARENT_SCOPE)
endfunction(exclude)

function (prepend OUTPUT PREPEND)
set(OUT "")
foreach(ITEM ${ARGN})
        list(APPEND OUT "${PREPEND}${ITEM}")
endforeach()
set(${OUTPUT} ${OUT} PARENT_SCOPE)
endfunction(prepend)

# ---[ CMake scripts + modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# External projects
include(ExternalProject)

include(cmake/Utils.cmake)

# options
option(USE_NERVANA_GPU "Use Nervana GPU backend" OFF)
option(USE_LMDB "Use LMDB" ON)
option(USE_LEVELDB "Use LMDB" ON)
option(USE_OPENCV "Use openCV" ON)
option(USE_ZMQ "Use ZMQ" OFF)
option(USE_ROCKSDB "Use RocksDB" OFF)
option(USE_MPI "Use MPI" OFF)
option(BUILD_SHARED_LIBS "Build libcaffe2.so" ON)

# ---[ Dependencies
include(cmake/Dependencies.cmake)

# ---[ Set link flag
if(BUILD_SHARED_LIBS)
  set(Caffe2_LINK Caffe2_CPU Caffe2_GPU Caffe_PROTO Caffe2_PROTO)
else()
    set(Caffe2_LINK -Wl,--whole-archive Caffe2_CPU Caffe2_GPU Caffe_PROTO Caffe2_PROTO -Wl,--no-whole-archive)
endif()

# ---[ Third party builds.
include_directories(${CMAKE_SOURCE_DIR})

# ---[ Old caffe protobuf.
add_subdirectory(caffe/proto)

# ---[ Main build
add_subdirectory(caffe2)

# ---[ Copy all python files to build directory
# Copy the files
install(DIRECTORY ${CMAKE_SOURCE_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}
        FILES_MATCHING PATTERN "*.py")

