# generate a list of kernels, but not actually emit files at config stage
execute_process(
  COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/../../../../../../../../third_party/composable_kernel/example/ck_tile/01_fmha/generate.py
  --api fwd --receipt 2 --list_blobs ${CMAKE_CURRENT_LIST_DIR}/fwd_blob_list.txt
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to generate a list of FWD kernels via Python.")
endif()

execute_process(
  COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/../../../../../../../../third_party/composable_kernel/example/ck_tile/01_fmha/generate.py
  --api bwd --receipt 2 --list_blobs ${CMAKE_CURRENT_LIST_DIR}/bwd_blob_list.txt
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to generate a list of BWD kernels via Python.")
endif()

# Generate the files for both fwd and bwd
execute_process(COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/../../../../../../../../third_party/composable_kernel/example/ck_tile/01_fmha/generate.py --api fwd --receipt 2 --output_dir ${CMAKE_CURRENT_LIST_DIR}
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to generate FWD kernels.")
endif()

execute_process(COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/../../../../../../../../third_party/composable_kernel/example/ck_tile/01_fmha/generate.py --api bwd --receipt 2 --output_dir ${CMAKE_CURRENT_LIST_DIR}
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to generate BWD kernels.")
endif()


# Finally, parse supported CK supported archs string and add ifdefs to generated files
set(GFX_LIST ${CK_SUPPORTED_GFX_ARCHS})

if(NOT "${GFX_LIST}" STREQUAL "")
  string(REPLACE "," ";" GFX_LIST ${GFX_LIST})

  # Start the final string
  set(HEADER "#if (")

  foreach(SUBSTRING IN LISTS GFX_LIST)
    # Append the condition
    set(HEADER "${HEADER}defined(__${SUBSTRING}__) || ")
  endforeach()

  string(REGEX REPLACE " \\|\\| $"
                    ")" HEADER "${HEADER}")

  set(FOOTER "#endif")
  set(FWD_LIST "fwd_blob_list.txt")
  set(BWD_LIST "bwd_blob_list.txt")

  file(READ ${FWD_LIST} FWD_FILES)
  file(READ ${BWD_LIST} BWD_FILES)

  set(FULL_LIST "${FWD_FILES}\n${BWD_FILES}")

  string(REPLACE "\n" ";" FULL_LIST "${FULL_LIST}")

  # Add ifdef header and footer
  foreach(INSTANCE IN LISTS FULL_LIST)
    if(EXISTS ${INSTANCE})
      file(READ ${INSTANCE} CURR_FILE)

      file(WRITE ${INSTANCE} "${HEADER}\n")

      file(APPEND ${INSTANCE} "${CURR_FILE}")

      file(APPEND ${INSTANCE} "\n${FOOTER}")

    else()
      message(WARNING "File ${INSTANCE} does not exist. Skipping.")
    endif()
  endforeach()
endif()
# add ifdefs to prevent calling stubbed out instances
#set(MHA_FILES "${CMAKE_CURRENT_SOURCE_DIR}/mha_fwd_ck.hip;${CMAKE_CURRENT_SOURCE_DIR}/mha_bwd_ck.hip;${CMAKE_CURRENT_SOURCE_DIR}/mha_varlen_fwd_ck.hip;${CMAKE_CURRENT_SOURCE_DIR}/mha_varlen_bwd_ck.hip")
#execute_process(COMMAND ls)
#message(${CMAKE_CURRENT_SOURCE_DIR})
#message(${CMAKE_CURRENT_LIST_DIR})
#foreach(MHA IN LISTS MHA_FILES)
# Ensure the file exists
#    if (EXISTS ${MHA})
#        # Use sed to modify the file in place
#        execute_process(
#            COMMAND sed -i -e "/float t = fmha/i ${HEADER}" ${MHA}
#            RESULT_VARIABLE SED_RESULT
#        )
#
#        if (NOT SED_RESULT EQUAL 0)
#            message(WARNING "Failed to modify file: ${MHA}")
#        endif()
#        execute_process(
#            COMMAND sed -i -e "/TORCH_CHECK(t >= 0/a ${FOOTER}" ${MHA}
#            RESULT_VARIABLE SED_RESULT
#        )
#        if (NOT SED_RESULT EQUAL 0)
#            message(WARNING "Failed to modify file: ${MHA}")
#        endif()
#    else()
#        message(WARNING "File ${MHA} does not exist. Skipping.")
#    endif()
#endforeach()

# Change file extensions to .hip
execute_process(COMMAND bash -c "for file in ${CMAKE_CURRENT_LIST_DIR}/*.cpp; do mv -- \"$file\" \"\${file%.cpp}.hip\"; done"
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to change the generated instances extensions from .cpp to .hpp")
endif()
