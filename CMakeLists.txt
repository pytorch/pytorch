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
option(USE_NERVANA_GPU
       "Use Nervana GPU backend" OFF)
option(USE_LMDB
       "Use LMDB" ON)
option(USE_LEVELDB
       "Use LMDB" ON)
option(USE_OPENCV
       "Use openCV" ON)
option(USE_ZMQ
       "Use ZMQ" OFF)
option(USE_ROCKSDB
       "Use RocksDB" OFF)
option(USE_MPI
       "Use MPI" OFF)

# ---[ Dependencies
include(cmake/Dependencies.cmake)

# Third party builds.
include_directories(${CMAKE_SOURCE_DIR})

# Old caffe protobuf.
add_subdirectory(caffe/proto)

add_subdirectory(caffe2)
target_link_libraries(Caffe2_CPU)
target_compile_features(Caffe2_CPU PRIVATE cxx_range_for)

# ---[ Copy all python files to build directory
# ---[ Assemble list of python files to copy
# Copy the files
# file(COPY ${Caffe2_PYTHON_SRC} DESTINATION ${CMAKE_BINARY_DIR})
install(DIRECTORY ${CMAKE_SOURCE_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX}
        FILES_MATCHING PATTERN "*.py")

