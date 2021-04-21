find_package(AVX)

add_executable(
  tensorexpr_bench
  bench_approx.cpp
  bench_batchnorm.cpp
  bench_concat.cpp
  bench_compile.cpp
  bench_fuser_overhead.cpp
  bench_gemm.cpp
  bench_parallel.cpp
  bench_reduce.cpp
  main.cpp)

if(C_AVX2_FOUND)
  message(STATUS "AVX2 compiler support found")
  target_compile_options(tensorexpr_bench PUBLIC -mavx2)
  target_compile_definitions(tensorexpr_bench PUBLIC USE_AVX2)
endif()

target_link_libraries(tensorexpr_bench PRIVATE torch_library benchmark)
