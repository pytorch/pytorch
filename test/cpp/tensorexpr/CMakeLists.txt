set(TENSOREXPR_TEST_ROOT ${TORCH_ROOT}/test/cpp/tensorexpr)

file(GLOB TENSOREXPR_TEST_SRCS ${TENSOREXPR_TEST_ROOT}/test_*.cpp)
set(TENSOREXPR_TEST_SRCS ${TENSOREXPR_TEST_SRCS} PARENT_SCOPE)

add_executable(test_tensorexpr
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${TENSOREXPR_TEST_ROOT}/gtest.cpp
  ${TENSOREXPR_TEST_ROOT}/padded_buffer.cpp
  ${TENSOREXPR_TEST_SRCS})

target_link_libraries(test_tensorexpr PRIVATE torch gtest)
target_include_directories(test_tensorexpr PRIVATE ${ATen_CPU_INCLUDE})

if (USE_CUDA)
  target_link_libraries(test_tensorexpr PRIVATE
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIB}
    ${CUDA_CUDA_LIB}
    ${TORCH_CUDA_LIBRARIES})

  target_compile_definitions(test_tensorexpr PRIVATE USE_CUDA)
elseif (USE_ROCM)
  target_link_libraries(test_tensorexpr PRIVATE
    ${ROCM_HIPRTC_LIB}
    ${PYTORCH_HIP_HCC_LIBRARIES}
    ${TORCH_CUDA_LIBRARIES})

  target_link_libraries(test_tensorexpr PRIVATE caffe2_gpu)

  target_compile_definitions(test_tensorexpr PRIVATE USE_ROCM)
endif()

if (INSTALL_TEST)
  install(TARGETS test_tensorexpr DESTINATION bin)
  # Install PDB files for MSVC builds
  if (MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_tensorexpr> DESTINATION bin OPTIONAL)
  endif()
endif()
