# ---[ CPU files.
file(GLOB_RECURSE NOMNI_SRCS *.cc)
file(GLOB_RECURSE NOMNI_TEST_SRCS *test.cc)
exclude(NOMNI_SRCS "${NOMNI_SRCS}" "${NOMNI_TEST_SRCS}")

# TODO(orionr): The nomnigraph source should likely just be included
# in the Caffe2 source list, since this won't live separately
add_library(nomnigraph STATIC "${NOMNI_SRCS}")
target_compile_options(nomnigraph PRIVATE "-DCAFFE2_BUILD_MAIN_LIB")
add_dependencies(nomnigraph Caffe2_PROTO)

target_include_directories(nomnigraph PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<INSTALL_INTERFACE:include>)
list(APPEND Caffe2_DEPENDENCY_LIBS nomnigraph)
set(Caffe2_DEPENDENCY_LIBS ${Caffe2_DEPENDENCY_LIBS} PARENT_SCOPE)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")

if (BUILD_TEST)
  foreach(test_src ${NOMNI_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} "${test_src}")
    target_link_libraries(${test_name} nomnigraph gtest_main)
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    install(TARGETS ${test_name} DESTINATION test)
  endforeach()
endif()
