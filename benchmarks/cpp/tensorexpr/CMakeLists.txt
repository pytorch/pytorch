find_package(AVX)

if(C_AVX2_FOUND)
  message(STATUS "AVX2 compiler support found")
  add_compile_options(-DUSE_AVX2)
  add_compile_options(-mavx2)
endif()

add_executable(tensorexpr_bench bench_gemm.cpp bench_compile.cpp bench_reduce.cpp main.cpp)
target_link_libraries(tensorexpr_bench PRIVATE torch_library benchmark)
