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

# Third party builds.
include_directories(${CMAKE_SOURCE_DIR})
file(GLOB Caffe2_THIRD_PARTY "${CMAKE_SOURCE_DIR}/third_party/*")
include_directories(${Caffe2_THIRD_PARTY})

# googletest.
add_subdirectory("${CMAKE_SOURCE_DIR}/third_party/googletest")
include_directories("${CMAKE_SOURCE_DIR}/third_party/googletest/googletest/include")

# Old caffe protobuf.
add_subdirectory(caffe/proto)

add_subdirectory(caffe2)
target_link_libraries(Caffe2_CPU)
target_compile_features(Caffe2_CPU PRIVATE cxx_range_for)
