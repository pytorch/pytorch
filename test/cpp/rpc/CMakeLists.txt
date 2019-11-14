set(RPC_TEST_ROOT ${TORCH_ROOT}/test/cpp/rpc)

file(GLOB RPC_TEST_SRCS ${RPC_TEST_ROOT}/test_*.cpp)
set(RPC_TEST_SRCS ${RPC_TEST_SRCS} PARENT_SCOPE)

add_executable(test_cpp_rpc
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${RPC_TEST_ROOT}/gtest.cpp
  ${RPC_TEST_SRCS})

target_link_libraries(test_cpp_rpc PRIVATE torch gtest)
target_include_directories(test_cpp_rpc PRIVATE ${ATen_CPU_INCLUDE})

if (USE_CUDA)
  target_link_libraries(test_cpp_rpc PRIVATE
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIB}
    ${CUDA_CUDA_LIB}
    ${TORCH_CUDA_LIBRARIES})

  target_compile_definitions(test_cpp_rpc PRIVATE USE_CUDA)
elseif (USE_ROCM)
  target_link_libraries(test_cpp_rpc PRIVATE
    ${ROCM_HIPRTC_LIB}
    ${PYTORCH_HIP_HCC_LIBRARIES}
    ${TORCH_CUDA_LIBRARIES})

  target_link_libraries(test_cpp_rpc PRIVATE caffe2_gpu)

  target_compile_definitions(test_cpp_rpc PRIVATE USE_ROCM)
endif()

if (INSTALL_TEST)
  install(TARGETS test_cpp_rpc DESTINATION bin)
  # Install PDB files for MSVC builds
  if (MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_cpp_rpc> DESTINATION bin OPTIONAL)
  endif()
endif()
