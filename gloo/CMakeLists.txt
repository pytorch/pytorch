# GLOO_SRCS is the list of source files that we need to build libgloo.
set(GLOO_SRCS)

# GLOO_HDRS is the list of header files that we need to install.
set(GLOO_HDRS)

# GLOO_TEST_SRCS is the list of test sources.
set(GLOO_TEST_SRCS)

# GLOO_BENCHMARK_SRCS is the list of benchmark sources.
set(GLOO_TEST_SRCS)

# Compiled sources in root directory
list(APPEND GLOO_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/context.cc"
)

file(GLOB GLOO_ROOT_HDRS "*.h")
list(APPEND GLOO_HDRS ${GLOO_ROOT_HDRS})

add_subdirectory(common)
add_subdirectory(rendezvous)
add_subdirectory(transport)

# Enable the following to get a list of source files
if(FALSE)
  message(STATUS "Sources: ")
  foreach(tmp ${GLOO_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()
endif()

add_library(gloo ${GLOO_SRCS})
target_link_libraries(gloo ${gloo_DEPENDENCY_LIBS})
get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
include_directories(${PARENT_DIR})
install(TARGETS gloo DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
foreach(HEADER ${GLOO_HDRS})
  string(REGEX MATCH "(.*)[/\\]" DIR ${HEADER})
  string(REGEX REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "gloo" DIR ${DIR})
  install(FILES ${HEADER} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${DIR})
endforeach()

if(BUILD_TEST)
add_subdirectory(test)
add_executable(gloo_test ${GLOO_TEST_SRCS})
target_link_libraries(gloo_test gloo gtest)
endif()

if(BUILD_BENCHMARK)
add_subdirectory(benchmark)
add_executable(gloo_benchmark ${GLOO_BENCHMARK_SRCS})
target_link_libraries(gloo_benchmark gloo)
endif()
