IF (MSVC)
  IF (MSVC_VERSION LESS 1911)
    return()
  ENDIF()
ENDIF(MSVC)

ADD_EXECUTABLE(scalar_test scalar_test.cpp)
target_link_libraries(scalar_test ATen_cpu)

ADD_EXECUTABLE(apply_utils_test apply_utils_test.cpp)
target_link_libraries(apply_utils_test ATen_cpu)

ADD_EXECUTABLE(basic basic.cpp)
target_link_libraries(basic ATen_cpu)

add_executable(atest atest.cpp)
target_link_libraries(atest ATen_cpu)

add_executable(broadcast_test broadcast_test.cpp)
target_link_libraries(broadcast_test ATen_cpu)

add_executable(wrapdim_test wrapdim_test.cpp)
target_link_libraries(wrapdim_test ATen_cpu)

add_executable(dlconvertor_test dlconvertor_test.cpp)
target_link_libraries(dlconvertor_test ATen_cpu)

add_executable(native_test native_test.cpp)
target_link_libraries(native_test ATen_cpu)

add_executable(scalar_tensor_test scalar_tensor_test.cpp)
target_link_libraries(scalar_tensor_test ATen_cpu)

add_executable(test_parallel test_parallel.cpp)
target_link_libraries(test_parallel ATen_cpu)

add_executable(undefined_tensor_test undefined_tensor_test.cpp)
target_link_libraries(undefined_tensor_test ATen_cpu)

add_executable(verify_api_visibility verify_api_visibility.cpp)
target_link_libraries(verify_api_visibility ATen_cpu)

add_executable(tbb_init_test tbb_init_test.cpp)
target_link_libraries(tbb_init_test ATen_cpu)

if(NOT NO_CUDA OR WITH_ROCM)
  torch_cuda_based_add_executable(integer_divider_test integer_divider_test.cu)
  target_link_libraries(integer_divider_test ATen_cpu ATen_cuda_library)
endif()

if(NOT NO_CUDA OR WITH_ROCM)
  torch_cuda_based_add_executable(cuda_rng_test cuda_rng_test.cpp)
  target_link_libraries(cuda_rng_test ATen_cpu ATen_cuda_library)
endif()

if(NOT NO_CUDA OR WITH_ROCM)
  torch_cuda_based_add_executable(apply_test apply_test.cpp)
  target_link_libraries(apply_test ATen_cpu ATen_cuda_library)
endif()

if (CUDNN_FOUND)
  add_executable(cudnn_test cudnn_test.cpp)
  target_link_libraries(cudnn_test ATen_cpu ATen_cuda_library)
endif()
