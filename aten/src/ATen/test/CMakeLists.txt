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

add_executable(undefined_tensor_test undefined_tensor_test.cpp)
target_link_libraries(undefined_tensor_test ATen)

add_executable(verify_api_visibility verify_api_visibility.cpp)
target_link_libraries(verify_api_visibility ATen)
