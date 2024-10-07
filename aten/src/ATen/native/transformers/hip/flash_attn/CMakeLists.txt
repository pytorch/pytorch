include(CMakePrintHelpers)
# validate user-specified fmha_fwd API list
set(FMHA_FWD_KNOWN_APIS "fwd;fwd_splitkv;fwd_appendkv")
set(FMHA_FWD_ENABLE_APIS "fwd" CACHE STRING
    "semicolon-separated list of APIs to generate (${FMHA_FWD_KNOWN_APIS}) & link, or \"all\".")
if(FMHA_FWD_ENABLE_APIS STREQUAL "all")
  set(FMHA_FWD_ENABLE_APIS ${FMHA_FWD_KNOWN_APIS})
endif()

foreach(api ${FMHA_FWD_ENABLE_APIS})
  if(NOT "${api}" IN_LIST FMHA_FWD_KNOWN_APIS)
    message(FATAL_ERROR "${api} isn't a known api: ${FMHA_FWD_KNOWN_APIS}.")
  endif()
endforeach()

# "fwd" is a must-have api for the fmha_fwd example, add it if not specified
if(NOT "fwd" IN_LIST FMHA_FWD_ENABLE_APIS)
  list(APPEND FMHA_FWD_ENABLE_APIS "fwd")
endif()

string(REPLACE ";" "," FMHA_FWD_APIS "${FMHA_FWD_ENABLE_APIS}")
cmake_print_variables(FMHA_FWD_APIS)
# generate a list of kernels, but not actually emit files at config sta
execute_process(
  COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/../../../../../../../third_party/composable_kernel/example/ck_tile/01_fmha/generate.py
  --api fwd,fwd_splitkv,fwd_appendkv --list_blobs ${CMAKE_CURRENT_LIST_DIR}/fwd_blob_list.txt
  RESULT_VARIABLE ret
)
if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to generate a list of FWD kernels via Python.")
endif()

execute_process(
  COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/../../../../../../../third_party/composable_kernel/example/ck_tile/01_fmha/generate.py
  --api bwd --list_blobs ${CMAKE_CURRENT_LIST_DIR}/bwd_blob_list.txt --receipt 3
  RESULT_VARIABLE ret
)
if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to genrate a list of BWD kernels via Python.")
endif()

# NOTE: for cmake, the FMHA_FWD_GEN_BLOBS/FMHA_BWD_GEN_BLOBS files must be in the same directory
#       as current cmake list, otherwise will not figure out the dependency properly
file(STRINGS ${CMAKE_CURRENT_LIST_DIR}/fwd_blob_list.txt FMHA_FWD_GEN_BLOBS)
file(STRINGS ${CMAKE_CURRENT_LIST_DIR}/bwd_blob_list.txt FMHA_BWD_GEN_BLOBS)

execute_process(COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/../../../../../../../third_party/composable_kernel/example/ck_tile/01_fmha/generate.py --api fwd,fwd_splitkv,fwd_appendkv --output_dir ${CMAKE_CURRENT_LIST_DIR}
)

execute_process(COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/../../../../../../../third_party/composable_kernel/example/ck_tile/01_fmha/generate.py --api bwd --output_dir ${CMAKE_CURRENT_LIST_DIR} --receipt 3
)

# Change file extensions to .hip
execute_process(COMMAND bash -c "for file in ${CMAKE_CURRENT_LIST_DIR}/*.cpp; do mv -- \"$file\" \"\${file%.cpp}.hip\"; done")
