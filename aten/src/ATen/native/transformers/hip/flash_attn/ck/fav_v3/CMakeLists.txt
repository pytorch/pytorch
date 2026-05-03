include(CMakePrintHelpers)

# Create an isolated venv for codegen.py which requires pandas>=2.1.0 (for
# the "future.no_silent_downcasting" option).  This avoids polluting the CI /
# host Python environment while still satisfying the dependency.
set(CODEGEN_VENV_DIR "${CMAKE_CURRENT_BINARY_DIR}/_codegen_venv")

if(NOT EXISTS "${CODEGEN_VENV_DIR}/bin/python3")
    message(STATUS "Creating isolated codegen venv at ${CODEGEN_VENV_DIR}")
    execute_process(
        COMMAND python3 -m venv "${CODEGEN_VENV_DIR}"
        RESULT_VARIABLE ret
    )
    if(ret AND NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to create codegen venv")
    endif()

    execute_process(
        COMMAND "${CODEGEN_VENV_DIR}/bin/pip" install --quiet "pandas>=2.1.0,<=2.3.3" numpy
        RESULT_VARIABLE ret
    )
    if(ret AND NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to install pandas/numpy into codegen venv")
    endif()
endif()

# Generate AITER/CK Asm code (using the venv python so pandas is new enough)
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env "AITER_GPU_ARCHS=gfx942;gfx950"
            "${CODEGEN_VENV_DIR}/bin/python3" ${CMAKE_SOURCE_DIR}/third_party/aiter/hsa/codegen.py -m fmha_v3_bwd --output_dir ${CMAKE_CURRENT_LIST_DIR}
    RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
    message(FATAL_ERROR "Failed to generate FAv3 CK Kernels")
endif()

execute_process(COMMAND bash -c "cp ${CMAKE_SOURCE_DIR}/third_party/aiter/csrc/cpp_itfs/mha_bwd.cu ${CMAKE_CURRENT_LIST_DIR}/mha_bwd.hip")

# ============================================================================
# Generate aiter_embedded_hsa.h with embedded binary .co files
# ============================================================================
set(AITER_HSA_DIR "${CMAKE_SOURCE_DIR}/third_party/aiter/hsa")
set(AITER_EMBEDDED_HSA_HEADER_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(AITER_EMBEDDED_HSA_HEADER "${AITER_EMBEDDED_HSA_HEADER_DIR}/aiter_embedded_hsa.h")
set(GENERATE_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/generate_aiter_embedded_hsa.py")

# Generate the embedded HSA header using Python script (much faster than CMake loops)
execute_process(
    COMMAND python3 ${GENERATE_SCRIPT}
            --hsa-dir ${AITER_HSA_DIR}
            --output ${AITER_EMBEDDED_HSA_HEADER}
            --subdirs gfx942/fmha_v3_bwd gfx950/fmha_v3_bwd
    RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
    message(FATAL_ERROR "Failed to generate aiter_embedded_hsa.h")
endif()

message(STATUS "AITER embedded HSA header: ${AITER_EMBEDDED_HSA_HEADER}")

# Export variables to parent scope for use by ck_sdpa target
set(AITER_EMBEDDED_HSA_HEADER_DIR ${AITER_EMBEDDED_HSA_HEADER_DIR} PARENT_SCOPE)
