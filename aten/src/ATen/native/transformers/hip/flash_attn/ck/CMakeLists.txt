# composable_kernel's ck_tile/01_fmha/generate.py registers its codegen ops via
# importlib's Loader.load_module(), removed in Python 3.15. Route it through a
# small compatibility shim so ROCm wheels build on 3.15 without bumping the CK
# submodule. See https://github.com/pytorch/pytorch/issues/184900.
set(CK_FMHA_GENERATE python3 ${CMAKE_CURRENT_LIST_DIR}/generate_compat.py
    ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py)

# Ask the CK codegen for the architecture families behind AT_ROCM_CK_SDPA_ARCHS.
# Without --targets the generator falls back to its own default ("gfx9,gfx950"),
# so no instances are emitted for RDNA4 even when the build is configured for it.
set(_ck_fmha_targets "")
foreach(ARCH IN LISTS AT_ROCM_CK_SDPA_ARCHS)
  if(ARCH STREQUAL "gfx942")
    list(APPEND _ck_fmha_targets gfx9)
  elseif(ARCH STREQUAL "gfx950")
    list(APPEND _ck_fmha_targets gfx950)
  elseif(ARCH MATCHES "^gfx12")
    list(APPEND _ck_fmha_targets gfx12)
  else()
    message(FATAL_ERROR "CK Tile FMHA: no codegen target family known for ${ARCH}")
  endif()
endforeach()
list(REMOVE_DUPLICATES _ck_fmha_targets)
if(NOT _ck_fmha_targets)
  # Guard against an empty resolved list: aten/src/ATen/CMakeLists.txt disables
  # USE_ROCM_CK_SDPA before reaching here.
  message(FATAL_ERROR "CK Tile FMHA: AT_ROCM_CK_SDPA_ARCHS is empty")
endif()
list(JOIN _ck_fmha_targets "," _ck_fmha_targets_arg)
message(STATUS "CK Tile FMHA codegen targets: ${_ck_fmha_targets_arg}")
set(CK_FMHA_TARGETS --targets ${_ck_fmha_targets_arg})

# generate a list of kernels, but not actually emit files at config stage
execute_process(
  COMMAND ${CK_FMHA_GENERATE} ${CK_FMHA_TARGETS} --optdim=32,64,128,256
  --api fwd --receipt 4 --filter "*_lse*ntrload*nsink*" --list_blobs ${CMAKE_CURRENT_LIST_DIR}/fwd_blob_list.txt
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to generate a list of FWD kernels via Python.")
endif()

execute_process(
  COMMAND ${CK_FMHA_GENERATE} ${CK_FMHA_TARGETS} --optdim=32,64,128,256
  --api fwd_splitkv --receipt 4 --filter "*psdv*_lse*_nsquant*" --list_blobs ${CMAKE_CURRENT_LIST_DIR}/fwd_splitkv_blob_list.txt
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
    message( FATAL_ERROR "CK Tile FMHA FAILED to generate a list of FWD_SPLITKV kernels via Python.")
endif()

execute_process(
  COMMAND ${CK_FMHA_GENERATE} ${CK_FMHA_TARGETS} --optdim=32,64,128,256
  --api fwd_appendkv --receipt 4 --filter "*psskddv_*" --list_blobs ${CMAKE_CURRENT_LIST_DIR}/fwd_appendkv_blob_list.txt
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
    message( FATAL_ERROR "CK Tile FMHA FAILED to generate a list of FWD_APPENDKV kernels via Python.")
endif()

execute_process(
  COMMAND ${CK_FMHA_GENERATE} ${CK_FMHA_TARGETS}
  --api bwd --optdim=32,64,128,256 --receipt 4 --filter "*psdv*@*psd*@*_pd1dv1*_ntrload*" --list_blobs ${CMAKE_CURRENT_LIST_DIR}/bwd_blob_list.txt
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to generate a list of BWD kernels via Python.")
endif()

# Generate the files for both fwd, fwd_splitkv, fwd_appendkv, and bwd
execute_process(COMMAND ${CK_FMHA_GENERATE} ${CK_FMHA_TARGETS} --api fwd  --optdim=32,64,128,256 --receipt 4 --filter "*_lse*ntrload*nsink*" --output_dir ${CMAKE_CURRENT_LIST_DIR}
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to generate FWD kernels.")
endif()

execute_process(COMMAND ${CK_FMHA_GENERATE} ${CK_FMHA_TARGETS} --api fwd_splitkv --optdim=32,64,128,256 --receipt 4 --filter "*psdv*_lse*_nsquant*" --output_dir ${CMAKE_CURRENT_LIST_DIR}
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
    message( FATAL_ERROR "CK Tile FMHA FAILED to generate FWD_SPLITKV kernels.")
endif()

execute_process(COMMAND ${CK_FMHA_GENERATE} ${CK_FMHA_TARGETS} --api fwd_appendkv --optdim=32,64,128,256 --receipt 4 --filter "*psskddv_*" --output_dir ${CMAKE_CURRENT_LIST_DIR}
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
    message( FATAL_ERROR "CK Tile FMHA FAILED to generate FWD_APPENDKV kernels.")
endif()

execute_process(COMMAND ${CK_FMHA_GENERATE} ${CK_FMHA_TARGETS} --api bwd --optdim=32,64,128,256 --receipt 4 --filter "*psdv*@*psd*@*_pd1dv1*_ntrload*" --output_dir ${CMAKE_CURRENT_LIST_DIR}
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to generate BWD kernels.")
endif()

# Change make_kernel to make_kernel_pt for fwd
execute_process(
  COMMAND bash -c "${CMAKE_CURRENT_LIST_DIR}/add_make_kernel_pt.sh ${CMAKE_CURRENT_LIST_DIR}/fwd_blob_list.txt"
  RESULT_VARIABLE ret)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to change make_kernel to make_kernel_pt for the fwd pass")
endif()

execute_process(
  COMMAND bash -c "${CMAKE_CURRENT_LIST_DIR}/add_make_kernel_pt.sh ${CMAKE_CURRENT_LIST_DIR}/fwd_splitkv_blob_list.txt"
  RESULT_VARIABLE ret)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to change make_kernel to make_kernel_pt for the fwd_splitkv pass")
endif()

execute_process(
  COMMAND bash -c "${CMAKE_CURRENT_LIST_DIR}/add_make_kernel_pt.sh ${CMAKE_CURRENT_LIST_DIR}/fwd_appendkv_blob_list.txt"
  RESULT_VARIABLE ret)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to change make_kernel to make_kernel_pt for the fwd appendkv pass")
endif()

# Change make_kernel to make_kernel_pt for bwd
execute_process(
  COMMAND bash -c "${CMAKE_CURRENT_LIST_DIR}/add_make_kernel_pt.sh ${CMAKE_CURRENT_LIST_DIR}/bwd_blob_list.txt"
  RESULT_VARIABLE ret)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to change make_kernel to make_kernel_pt for the bwd pass")
endif()

# Change file extensions to .hip
execute_process(COMMAND bash -c "for file in ${CMAKE_CURRENT_LIST_DIR}/*.cpp; do mv -- \"$file\" \"\${file%.cpp}.hip\"; done"
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to change the generated instances extensions from .cpp to .hpp")
endif()

# The instances this configuration asks for. They are generated next to the
# sources, so the directory can also hold instances from an earlier configure
# with a different target list; name them from the blob lists written above
# rather than from all matching files present in the directory.
set(_ck_fmha_generated "")
foreach(_ck_fmha_api fwd fwd_splitkv fwd_appendkv bwd)
  file(STRINGS ${CMAKE_CURRENT_LIST_DIR}/${_ck_fmha_api}_blob_list.txt _ck_fmha_blobs)
  foreach(_ck_fmha_blob IN LISTS _ck_fmha_blobs)
    get_filename_component(_ck_fmha_name "${_ck_fmha_blob}" NAME)
    string(REGEX REPLACE "\\.cpp$" ".hip" _ck_fmha_name "${_ck_fmha_name}")
    # Validate the naming convention the source selection in
    # aten/src/ATen/CMakeLists.txt relies on.
    if(NOT _ck_fmha_name MATCHES "^fmha_.*\\.hip$")
      message(FATAL_ERROR "CK Tile FMHA: unexpected generated instance name ${_ck_fmha_name}")
    endif()
    if(NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/${_ck_fmha_name})
      message(FATAL_ERROR "CK Tile FMHA: ${_ck_fmha_name} is listed in ${_ck_fmha_api}_blob_list.txt but was not generated")
    endif()
    list(APPEND _ck_fmha_generated ${CMAKE_CURRENT_LIST_DIR}/${_ck_fmha_name})
  endforeach()
endforeach()
list(REMOVE_DUPLICATES _ck_fmha_generated)
set(CK_FMHA_GENERATED_SOURCES ${_ck_fmha_generated} PARENT_SCOPE)
