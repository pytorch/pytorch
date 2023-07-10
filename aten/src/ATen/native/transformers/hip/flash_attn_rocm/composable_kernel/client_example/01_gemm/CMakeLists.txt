add_executable(client_gemm gemm.cpp)
target_link_libraries(client_gemm PRIVATE composable_kernel::device_operations)
