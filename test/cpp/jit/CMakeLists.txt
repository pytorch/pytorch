set(JIT_TEST_ROOT ${TORCH_ROOT}/test/cpp/jit)

add_executable(test_jit
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${JIT_TEST_ROOT}/test.cpp)

target_link_libraries(test_jit PRIVATE caffe2 gtest)
target_include_directories(test_jit PRIVATE ${ATen_CPU_INCLUDE})
target_compile_definitions(test_jit PRIVATE USE_GTEST)

if (USE_CUDA)
  target_link_libraries(test_jit PRIVATE
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIB}
    ${CUDA_CUDA_LIB}
    ${TORCH_CUDA_LIBRARIES})

  target_link_libraries(test_jit PRIVATE caffe2_gpu)

  target_compile_definitions(test_jit PRIVATE USE_CUDA)
endif()
