ADD_EXECUTABLE(scalar_test scalar_test.cpp)
target_link_libraries(scalar_test PRIVATE ATen)

ADD_EXECUTABLE(basic basic.cpp)
target_link_libraries(basic PRIVATE ATen)

add_executable(atest atest.cpp)
target_link_libraries(atest PRIVATE ATen)

add_executable(broadcast_test broadcast_test.cpp)
target_link_libraries(broadcast_test PRIVATE ATen)

add_executable(wrapdim_test wrapdim_test.cpp)
target_link_libraries(wrapdim_test PRIVATE ATen)

add_executable(dlconvertor_test dlconvertor_test.cpp)
target_link_libraries(dlconvertor_test PRIVATE ATen)

add_executable(native_test native_test.cpp)
target_link_libraries(native_test PRIVATE ATen)

add_executable(scalar_tensor_test scalar_tensor_test.cpp)
target_link_libraries(scalar_tensor_test PRIVATE ATen)

add_executable(test_parallel test_parallel.cpp)
target_link_libraries(test_parallel PRIVATE ATen)

add_executable(undefined_tensor_test undefined_tensor_test.cpp)
target_link_libraries(undefined_tensor_test PRIVATE ATen)

add_executable(verify_api_visibility verify_api_visibility.cpp)
target_link_libraries(verify_api_visibility PRIVATE ATen)

add_executable(tbb_init_test tbb_init_test.cpp)
target_link_libraries(tbb_init_test PRIVATE ATen)

if(NOT NO_CUDA)
  cuda_add_executable(integer_divider_test integer_divider_test.cu)
  target_link_libraries(integer_divider_test PRIVATE ATen)
endif()

if (CUDNN_FOUND)
  add_executable(cudnn_test cudnn_test.cpp)
  target_link_libraries(cudnn_test PRIVATE ATen)
endif()
