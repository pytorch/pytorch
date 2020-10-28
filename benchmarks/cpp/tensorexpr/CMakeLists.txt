add_executable(tensorexpr_bench bench_gemm.cpp bench_compile.cpp main.cpp)
target_link_libraries(tensorexpr_bench PRIVATE torch_library benchmark)
