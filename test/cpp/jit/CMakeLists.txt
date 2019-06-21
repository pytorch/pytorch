set(JIT_TEST_ROOT ${TORCH_ROOT}/test/cpp/jit)

add_executable(test_jit
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${JIT_TEST_ROOT}/test.cpp)

target_link_libraries(test_jit PRIVATE torch gtest)
target_include_directories(test_jit PRIVATE ${ATen_CPU_INCLUDE})
target_compile_definitions(test_jit PRIVATE USE_GTEST)

if (USE_CUDA)
  target_link_libraries(test_jit PRIVATE
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIB}
    ${CUDA_CUDA_LIB}
    ${TORCH_CUDA_LIBRARIES})

  target_compile_definitions(test_jit PRIVATE USE_CUDA)
elseif (USE_ROCM)
  target_link_libraries(test_jit PRIVATE
    ${ROCM_HIPRTC_LIB}
    ${PYTORCH_HIP_HCC_LIBRARIES}
    ${TORCH_CUDA_LIBRARIES})

  target_link_libraries(test_jit PRIVATE caffe2_gpu)

  target_compile_definitions(test_jit PRIVATE USE_ROCM)
endif()

if (INSTALL_TEST)
  install(TARGETS test_jit DESTINATION bin)
  # Install PDB files for MSVC builds
  if (MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_jit> DESTINATION bin OPTIONAL)
  endif()
endif()
