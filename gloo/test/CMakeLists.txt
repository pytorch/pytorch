set(GLOO_TEST_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/allreduce_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/barrier_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/broadcast_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/linux_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/main.cc"
  )

add_executable(test ${GLOO_TEST_SRCS})
target_link_libraries(test gloo gtest)

if(USE_CUDA)
  set(GLOO_TEST_CUDA_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_allreduce_test.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_base_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_broadcast_test.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cc"
    )

  cuda_add_executable(test_cuda ${GLOO_TEST_CUDA_SRCS})
  target_link_libraries(test_cuda gloo_cuda gtest)
endif()
