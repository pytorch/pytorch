IF (MSVC)
  IF (MSVC_VERSION LESS 1911)
    return()
  ENDIF()
ENDIF(MSVC)

ADD_EXECUTABLE(scalar_test scalar_test.cpp)
target_link_libraries(scalar_test ATen)

ADD_EXECUTABLE(basic basic.cpp)
target_link_libraries(basic ATen)

add_executable(atest atest.cpp)
target_link_libraries(atest ATen)

add_executable(broadcast_test broadcast_test.cpp)
target_link_libraries(broadcast_test ATen)

add_executable(wrapdim_test wrapdim_test.cpp)
target_link_libraries(wrapdim_test ATen)

add_executable(dlconvertor_test dlconvertor_test.cpp)
target_link_libraries(dlconvertor_test ATen)

add_executable(native_test native_test.cpp)
target_link_libraries(native_test ATen)

add_executable(scalar_tensor_test scalar_tensor_test.cpp)
target_link_libraries(scalar_tensor_test ATen)

add_executable(test_parallel test_parallel.cpp)
target_link_libraries(test_parallel ATen)

add_executable(undefined_tensor_test undefined_tensor_test.cpp)
target_link_libraries(undefined_tensor_test ATen)

add_executable(verify_api_visibility verify_api_visibility.cpp)
target_link_libraries(verify_api_visibility ATen)

add_executable(tbb_init_test tbb_init_test.cpp)
target_link_libraries(tbb_init_test ATen)

if(NOT NO_CUDA)
  cuda_add_executable(integer_divider_test integer_divider_test.cu)
  target_link_libraries(integer_divider_test ATen)
endif()

if(NOT NO_CUDA)
  cuda_add_executable(cuda_rng_test cuda_rng_test.cpp)
  target_link_libraries(cuda_rng_test ATen)
endif()

if (CUDNN_FOUND)
  add_executable(cudnn_test cudnn_test.cpp)
  target_link_libraries(cudnn_test ATen)
endif()
