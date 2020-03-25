set(JIT_TEST_ROOT ${TORCH_ROOT}/test/cpp/jit)

file(GLOB JIT_TEST_SRCS ${JIT_TEST_ROOT}/test_*.cpp)
set(JIT_TEST_SRCS ${JIT_TEST_SRCS} PARENT_SCOPE)

add_executable(test_jit
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${JIT_TEST_ROOT}/gtest.cpp
  ${JIT_TEST_SRCS})

set(JIT_TEST_DEPENDENCIES torch gtest)

if(MSVC)
  list(APPEND JIT_TEST_DEPENDENCIES onnx_library)
endif(MSVC)

target_link_libraries(test_jit PRIVATE ${JIT_TEST_DEPENDENCIES})
target_include_directories(test_jit PRIVATE ${ATen_CPU_INCLUDE})

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
