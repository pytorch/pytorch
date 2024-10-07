include(CMakePrintHelpers)
message("ALPHA: MADE IT TO MY CMAKELISTS")
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
message("HINATA")
cmake_print_variables(FMHA_FWD_APIS)
# generate a list of kernels, but not actually emit files at config sta
execute_process(
  COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/../../../../../../../third_party/composable_kernel/example/ck_tile/01_fmha/generate.py
  --api ${FMHA_FWD_APIS} --list_blobs ${CMAKE_CURRENT_LIST_DIR}/fwd_blob_list.txt
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

#add_custom_command(
#  OUTPUT ${FMHA_FWD_GEN_BLOBS}
#  COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/../../../../../../../third_party/composable_kernel/example/ck_tile/01_fmha/generate.py
#  --api ${FMHA_FWD_APIS} --output_dir ${CMAKE_CURRENT_LIST_DIR}
#)

execute_process(COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/../../../../../../../third_party/composable_kernel/example/ck_tile/01_fmha/generate.py --api ${FMHA_FWD_APIS} --output_dir ${CMAKE_CURRENT_LIST_DIR}
)

#add_custom_command(
#  OUTPUT ${FMHA_BWD_GEN_BLOBS}
#  COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/../../../../../../../third_party/composable_kernel/example/ck_tile/01_fmha/generate.py
#  --api bwd --output_dir ${CMAKE_CURRENT_LIST_DIR} --receipt 3
#)

execute_process(COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/../../../../../../../third_party/composable_kernel/example/ck_tile/01_fmha/generate.py --api bwd --output_dir ${CMAKE_CURRENT_LIST_DIR} --receipt 3
)
message("CHANGING FILE EXTENSIONS TO HIP!")
# Need to create script to run bash for-loop over generated files since it's not possible directly
# in `execute_process`


#file(WRITE change_cpp_to_hip_ext.sh "
#!/bin/bash
#for file in *.cpp; do mv -- \"$file\" \"${file%.cpp}.hip\"; done
#")

#execute_process(COMMAND chmod +x change_cpp_to_hip_ext.sh)



execute_process(COMMAND bash -c "for file in ${CMAKE_CURRENT_LIST_DIR}/*.cpp; do mv -- \"$file\" \"\${file%.cpp}.hip\"; done")

message("FINISHED CHANGING EXTENSIONS TO HIP!")
message("ALPHA - END MY CMAKE BLURB")

#[==[
set(EXAMPLE_FMHA_FWD "tile_example_fmha_fwd")
# not using add_example_executable() to add this target, since we don't want this to have
# to be included in "make all/install/check"
message("adding example ${EXAMPLE_FMHA_FWD}")
add_executable(${EXAMPLE_FMHA_FWD} EXCLUDE_FROM_ALL ${CMAKE_CURRENT_LIST_DIR}/../../../../../../../third_party/composable_kernel/example/ck_tile/01_fmha/fmha_fwd.cpp)
target_include_directories(${EXAMPLE_FMHA_FWD} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../../../../../../third_party/composable_kernel/example/ck_tile/01_fmha/)
target_sources(${EXAMPLE_FMHA_FWD} PUBLIC ${FMHA_FWD_GEN_BLOBS})

set(EXAMPLE_FMHA_BWD "tile_example_fmha_bwd")
# not using add_example_executable() to add this target, since we don't want this to have
# to be included in "make all/install/check"
message("adding example ${EXAMPLE_FMHA_BWD}")
add_executable(${EXAMPLE_FMHA_BWD} EXCLUDE_FROM_ALL ${CMAKE_CURRENT_LIST_DIR}/../../../../../../../third_party/composable_kernel/example/ck_tile/01_fmha/fmha_bwd.cpp)
target_include_directories(${EXAMPLE_FMHA_BWD} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../../../../../../third_party/composable_kernel/example/ck_tile/01_fmha/)
target_sources(${EXAMPLE_FMHA_BWD} PUBLIC ${FMHA_BWD_GEN_BLOBS})

# NOTE: this is dangerous since will change the whole kernel to flush denormals
#       WIP with compiler team for an exp2 intrinsic..., then remove this
if(NOT DEFINED FMHA_FWD_FAST_EXP2)
    set(FMHA_FWD_FAST_EXP2 true)
endif()

set(EXAMPLE_FMHA_FWD_COMPILE_OPTIONS)
set(EXAMPLE_FMHA_BWD_COMPILE_OPTIONS)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
#       ... because they are auto-generated
if(FMHA_FWD_FAST_EXP2)
	list(APPEND EXAMPLE_FMHA_FWD_COMPILE_OPTIONS -Wno-undefined-func-template -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -fgpu-flush-denormals-to-zero)
else()
	list(APPEND EXAMPLE_FMHA_FWD_COMPILE_OPTIONS -Wno-undefined-func-template -DCK_TILE_FMHA_FWD_FAST_EXP2=0)
endif()
list(APPEND EXAMPLE_FMHA_BWD_COMPILE_OPTIONS -Wno-undefined-func-template -fgpu-flush-denormals-to-zero)

# conditionally enable call to the fwd_splitkv API in fmha_fwd example
if("fwd_splitkv" IN_LIST FMHA_FWD_ENABLE_APIS)
  list(APPEND EXAMPLE_FMHA_FWD_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_SPLITKV_API=1)
else()
  list(APPEND EXAMPLE_FMHA_FWD_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_SPLITKV_API=0)
endif()

# conditionally enable call to the fwd_appendkv API in fmha_fwd example
if("fwd_appendkv" IN_LIST FMHA_FWD_ENABLE_APIS)
  list(APPEND EXAMPLE_FMHA_FWD_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_APPENDKV_API=1)
else()
  list(APPEND EXAMPLE_FMHA_FWD_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_APPENDKV_API=0)
endif()

# Allow comparing floating points directly in order to check sentinel values
list(APPEND EXAMPLE_FMHA_FWD_COMPILE_OPTIONS -Wno-float-equal)
list(APPEND EXAMPLE_FMHA_BWD_COMPILE_OPTIONS -Wno-float-equal)

target_compile_options(${EXAMPLE_FMHA_FWD} PUBLIC ${EXAMPLE_FMHA_FWD_COMPILE_OPTIONS})
target_compile_options(${EXAMPLE_FMHA_BWD} PUBLIC ${EXAMPLE_FMHA_BWD_COMPILE_OPTIONS})

# TODO: we have to turn off this global prop, otherwise the progress bar generated
# by cmake will print too many files, execvp: /bin/sh: Argument list too long
# however, this property may affect global
# TODO: consider codegen a makefile by us
set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
message("GAMMA - SHOULD NOT SEE THIS")
]==]
