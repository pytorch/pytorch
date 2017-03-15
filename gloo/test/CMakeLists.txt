set(GLOO_TEST_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/allreduce_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/barrier_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/broadcast_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/main.cc"
  )

add_executable(gloo_test ${GLOO_TEST_SRCS})
target_link_libraries(gloo_test gloo gtest)
