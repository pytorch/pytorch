# ---[ CPU files.
file(GLOB_RECURSE NOMNI_SRCS *.cc)
file(GLOB_RECURSE NOMNI_TEST_SRCS *test.cc)
exclude(NOMNI_SRCS "${NOMNI_SRCS}" "${NOMNI_TEST_SRCS}")

add_library(nomnigraph STATIC "${NOMNI_SRCS}")
add_dependencies(nomnigraph Caffe_PROTO Caffe2_PROTO)

target_include_directories(nomnigraph PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
list(APPEND Caffe2_DEPENDENCY_LIBS nomnigraph)
set(Caffe2_DEPENDENCY_LIBS ${Caffe2_DEPENDENCY_LIBS} PARENT_SCOPE)

if (BUILD_TEST)
  foreach(test_src ${NOMNI_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} "${test_src}")
    target_link_libraries(${test_name} nomnigraph gtest_main)
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    install(TARGETS ${test_name} DESTINATION test)
  endforeach()
endif()
