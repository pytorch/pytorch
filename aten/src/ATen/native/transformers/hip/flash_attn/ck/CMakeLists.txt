# generate a list of kernels, but not actually emit files at config stage
execute_process(
  COMMAND
    python3
    ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py
    --api fwd --receipt 600 --list_blobs
    ${CMAKE_CURRENT_LIST_DIR}/fwd_blob_list.txt
  RESULT_VARIABLE ret)

if(ret AND NOT ret EQUAL 0)
  message(
    FATAL_ERROR
      "CK Tile FMHA FAILED to generate a list of FWD kernels via Python.")
endif()

execute_process(
  COMMAND
    python3
    ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py
    --api bwd --receipt 600 --list_blobs
    ${CMAKE_CURRENT_LIST_DIR}/bwd_blob_list.txt
  RESULT_VARIABLE ret)

if(ret AND NOT ret EQUAL 0)
  message(
    FATAL_ERROR
      "CK Tile FMHA FAILED to generate a list of BWD kernels via Python.")
endif()

# Generate the files for both fwd and bwd
execute_process(
  COMMAND
    python3
    ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py
    --api fwd --receipt 600 --output_dir ${CMAKE_CURRENT_LIST_DIR})

if(ret AND NOT ret EQUAL 0)
  message(FATAL_ERROR "CK Tile FMHA FAILED to generate FWD kernels.")
endif()

execute_process(
  COMMAND
    python3
    ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py
    --api bwd --receipt 600 --output_dir ${CMAKE_CURRENT_LIST_DIR}
  RESULT_VARIABLE ret)

if(ret AND NOT ret EQUAL 0)
  message(FATAL_ERROR "CK Tile FMHA FAILED to generate BWD kernels.")
endif()

# Change make_kernel to make_kernel_pt for fwd
execute_process(
  COMMAND
    bash -c
    "${CMAKE_CURRENT_LIST_DIR}/add_make_kernel_pt.sh ${CMAKE_CURRENT_LIST_DIR}/fwd_blob_list.txt"
  RESULT_VARIABLE ret)

if(ret AND NOT ret EQUAL 0)
  message(
    FATAL_ERROR
      "CK Tile FMHA FAILED to change make_kernel to make_kernel_pt for the fwd pass"
  )
endif()

# Change make_kernel to make_kernel_pt for bwd
execute_process(
  COMMAND
    bash -c
    "${CMAKE_CURRENT_LIST_DIR}/add_make_kernel_pt.sh ${CMAKE_CURRENT_LIST_DIR}/bwd_blob_list.txt"
  RESULT_VARIABLE ret)

if(ret AND NOT ret EQUAL 0)
  message(
    FATAL_ERROR
      "CK Tile FMHA FAILED to change make_kernel to make_kernel_pt for the bwd pass"
  )
endif()

# Change file extensions to .hip
execute_process(
  COMMAND
    bash -c
    "for file in ${CMAKE_CURRENT_LIST_DIR}/*.cpp; do mv -- \"$file\" \"\${file%.cpp}.hip\"; done"
  RESULT_VARIABLE ret)

if(ret AND NOT ret EQUAL 0)
  message(
    FATAL_ERROR
      "CK Tile FMHA FAILED to change the generated instances extensions from .cpp to .hpp"
  )
endif()
