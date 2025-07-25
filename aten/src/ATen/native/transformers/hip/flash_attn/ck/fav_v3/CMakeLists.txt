include(CMakePrintHelpers)

# Generate AITER/CK Asm code
execute_process(
  COMMAND
    python3
    ${CMAKE_SOURCE_DIR}/third_party/aiter/csrc/py_itfs_cu/fmha_v3_bwd_kernel_generate.py
    --receipt 1 --output_dir ${CMAKE_CURRENT_LIST_DIR}
  RESULT_VARIABLE ret)

if(ret AND NOT ret EQUAL 0)
  message(FATAL_ERROR "Failed to generate FAv3 CK Kernels")
endif()

execute_process(
  COMMAND
    python3
    ${CMAKE_SOURCE_DIR}/third_party/aiter/csrc/cpp_itfs/mha_bwd_generate.py
    --receipt 3 --output_dir ${CMAKE_CURRENT_LIST_DIR}
  RESULT_VARIABLE ret)

# Change file extensions to .hip
execute_process(
  COMMAND
    bash -c
    "for file in ${CMAKE_CURRENT_LIST_DIR}/*.cpp; do mv -- \"$file\" \"\${file%.cpp}.hip\"; done"
)
