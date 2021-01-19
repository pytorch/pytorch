add_executable(
  tensorexpr_bench
  bench_approx.cpp
  bench_compile.cpp
  bench_fuser_overhead.cpp
  bench_gemm.cpp
  main.cpp)

target_link_libraries(tensorexpr_bench PRIVATE torch_library benchmark)
