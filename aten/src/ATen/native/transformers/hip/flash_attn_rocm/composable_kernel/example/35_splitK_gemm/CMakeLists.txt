add_custom_target(example_splitK_gemm_xdl)

add_example_executable(example_splitK_gemm_xdl_fp32 splitK_gemm_xdl_fp32.cpp)
add_example_executable(example_splitK_gemm_xdl_fp16 splitK_gemm_xdl_fp16.cpp)
add_example_executable(example_splitK_gemm_xdl_bfp16 splitK_gemm_xdl_bfp16.cpp)
add_example_executable(example_splitK_gemm_xdl_int8 splitK_gemm_xdl_int8.cpp)

add_dependencies(example_splitK_gemm_xdl
                 example_splitK_gemm_xdl_fp32
                 example_splitK_gemm_xdl_fp16
                 example_splitK_gemm_xdl_bfp16
                 example_splitK_gemm_xdl_int8)

if(USE_BITINT_EXTENSION_INT4)
  add_example_executable(example_splitK_gemm_xdl_int4 splitK_gemm_xdl_int4.cpp)
  add_dependencies(example_splitK_gemm_xdl example_splitK_gemm_xdl_int4)
endif()
