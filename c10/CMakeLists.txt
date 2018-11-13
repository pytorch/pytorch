cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(c10 CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Main build file for the C10 library.
#
# Note that the C10 library should maintain minimal dependencies - especially,
# it should not depend on any library that is implementation specific or
# backend specific. It should in particular NOT be dependent on any generated
# protobuf header files, because protobuf header files will transitively force
# one to link against a specific protobuf version.

# ---[ Configure macro file.
set(C10_USE_GFLAGS ${USE_GFLAGS}) # used in cmake_macros.h.in
set(C10_USE_GLOG ${USE_GLOG}) # used in cmake_macros.h.in
set(C10_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}) # used in cmake_macros.h.in
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/macros/cmake_macros.h.in
    ${CMAKE_BINARY_DIR}/c10/macros/cmake_macros.h)

# Note: if you want to add ANY dependency to the c10 library, make sure you
# check with the core PyTorch developers as the dependendency will be
# transitively passed on to all libraries dependent on PyTorch.
file(GLOB C10_SRCS
        *.cpp
        detail/*.cpp
        macros/*.cpp
        util/*.cpp
        )
file(GLOB_RECURSE C10_ALL_TEST_FILES test/*.cpp)
file(GLOB_RECURSE C10_HEADERS *.h)
add_library(c10 ${C10_SRCS} ${C10_HEADERS})
# If building shared library, set dllimport/dllexport proper.
target_compile_options(c10 PRIVATE "-DC10_BUILD_MAIN_LIB")
# Enable hidden visibility if compiler supports it.
if (${COMPILER_SUPPORTS_HIDDEN_VISIBILITY})
  target_compile_options(c10 PRIVATE "-fvisibility=hidden")
endif()

# ---[ Dependency of c10
if (${USE_GFLAGS})
   target_link_libraries(c10 PUBLIC gflags)
endif()

if (${USE_GLOG})
    target_link_libraries(c10 PUBLIC glog::glog)
endif()

if (ANDROID)
    target_link_libraries(c10 PRIVATE log)
endif()

target_include_directories(
    c10 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>)

add_subdirectory(test)

# ---[ Installation
# Note: for now, we will put all export path into one single Caffe2Targets group
# to deal with the cmake deployment need. Inside the Caffe2Targets set, the
# individual libraries like libc10.so and libcaffe2.so are still self-contained.
install(TARGETS c10 EXPORT Caffe2Targets DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")
install(FILES ${CMAKE_BINARY_DIR}/c10/macros/cmake_macros.h
        DESTINATION include/c10/macros)
