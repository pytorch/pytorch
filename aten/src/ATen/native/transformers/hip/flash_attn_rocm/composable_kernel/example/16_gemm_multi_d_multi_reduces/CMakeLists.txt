add_custom_target(example_gemm_reduce_xdl)
add_custom_target(example_gemm_reduce_xdl_max)
add_custom_target(example_gemm_reduce_xdl_mean_meansquare)
add_custom_target(example_gemm_add_add_mean_meansquare_xdl)

add_example_executable(example_gemm_max_xdl_fp16 gemm_max_xdl_fp16.cpp)
add_example_executable(example_gemm_max_xdl_int8 gemm_max_xdl_int8.cpp)
add_example_executable(example_gemm_max_xdl_fp32 gemm_max_xdl_fp32.cpp)
add_example_executable(example_gemm_max_xdl_bf16 gemm_max_xdl_bf16.cpp)

add_example_executable(example_gemm_add_add_mean_meansquare_xdl_fp16 gemm_add_add_mean_meansquare_xdl_fp16.cpp)

add_example_executable(example_gemm_mean_meansquare_xdl_fp16 gemm_mean_meansquare_xdl_fp16.cpp)
add_example_executable(example_gemm_mean_meansquare_xdl_fp32 gemm_mean_meansquare_xdl_fp32.cpp)
add_example_executable(example_gemm_mean_meansquare_xdl_bf16 gemm_mean_meansquare_xdl_bf16.cpp)
add_example_executable(example_gemm_add_addsquare_xdl_int8 gemm_add_addsquare_xdl_int8.cpp)

add_dependencies(example_gemm_reduce_xdl_max
                 example_gemm_max_xdl_bf16
                 example_gemm_max_xdl_fp16
                 example_gemm_max_xdl_fp32
                 example_gemm_max_xdl_int8)

add_dependencies(example_gemm_reduce_xdl_mean_meansquare
                 example_gemm_mean_meansquare_xdl_fp16
                 example_gemm_mean_meansquare_xdl_fp32
                 example_gemm_mean_meansquare_xdl_bf16
                 example_gemm_add_addsquare_xdl_int8)

add_dependencies(example_gemm_add_add_mean_meansquare_xdl example_gemm_add_add_mean_meansquare_xdl_fp16)

add_dependencies(example_gemm_reduce_xdl
                 example_gemm_reduce_xdl_mean_meansquare
                 example_gemm_reduce_xdl_max
                 example_gemm_add_add_mean_meansquare_xdl)

if(USE_BITINT_EXTENSION_INT4)
  add_example_executable(example_gemm_max_xdl_int4 gemm_max_xdl_int4.cpp)
  add_dependencies(example_gemm_reduce_xdl_max example_gemm_max_xdl_int4)
endif()
