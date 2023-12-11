cmake_minimum_required(VERSION 3.1)

set(TORCH_ROOT ${CMAKE_CURRENT_LIST_DIR}/../..)
set(TEST_ROOT ${TORCH_ROOT}/test/edge)
set(OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out)
file(GLOB_RECURSE all_python "${TORCH_ROOT}/torchgen/*.py")
include(${TORCH_ROOT}/cmake/public/utils.cmake)
append_cxx_flag_if_supported("-Wno-unused-private-field" CMAKE_CXX_FLAGS)

# Generate unboxing kernels
set(GEN_COMMAND
        "${PYTHON_EXECUTABLE}" -m torchgen.gen_executorch
        --source-path=${TEST_ROOT}
        --install-dir=${OUTPUT_DIRECTORY}
        --tags-path=${TORCH_ROOT}/aten/src/ATen/native/tags.yaml
        --aten-yaml-path=${TORCH_ROOT}/aten/src/ATen/native/native_functions.yaml
        --use-aten-lib
        --op-selection-yaml-path=${TEST_ROOT}/selected_operators.yaml
        --custom-ops-yaml-path=${TEST_ROOT}/custom_ops.yaml
        )
set(GEN_COMMAND_sources
        ${OUTPUT_DIRECTORY}/RegisterCodegenUnboxedKernelsEverything.cpp
        ${OUTPUT_DIRECTORY}/RegisterCPUCustomOps.cpp
        ${OUTPUT_DIRECTORY}/Functions.h
        ${OUTPUT_DIRECTORY}/NativeFunctions.h
        ${OUTPUT_DIRECTORY}/CustomOpsNativeFunctions.h
        )
message(STATUS "Generating sources for unboxing kernels ${GEN_COMMAND}")
add_custom_command(
        COMMENT "Generating sources"
        OUTPUT ${GEN_COMMAND_sources}
        COMMAND ${GEN_COMMAND}
        DEPENDS
        ${all_python}
        ${TORCH_ROOT}/aten/src/ATen/native/native_functions.yaml
        ${TORCH_ROOT}/aten/src/ATen/native/tags.yaml
        ${TEST_ROOT}/templates/Functions.h
        ${TEST_ROOT}/templates/NativeFunctions.h
        ${TEST_ROOT}/templates/RegisterCodegenUnboxedKernels.cpp
        ${TEST_ROOT}/templates/RegisterDispatchKeyCustomOps.cpp
        WORKING_DIRECTORY ${TORCH_ROOT}
)
add_custom_target(unbox_target DEPENDS ${GEN_COMMAND_sources})

add_library(unbox_lib STATIC
        ${GEN_COMMAND_sources}
        ${TEST_ROOT}/operator_registry.cpp
        ${TEST_ROOT}/custom_ops.cpp
        )
target_include_directories(unbox_lib PUBLIC ${TEST_ROOT} ${ATen_CPU_INCLUDE})
target_link_libraries(unbox_lib PUBLIC torch_cpu)
target_compile_definitions(unbox_lib PUBLIC USE_ATEN_LIB)

add_executable(test_edge_op_registration
        ${TEST_ROOT}/test_operator_registration.cpp
        ${TEST_ROOT}/test_main.cpp
        )

target_compile_definitions(test_edge_op_registration PRIVATE USE_GTEST)

set(TEST_DEPENDENCIES gtest unbox_lib)

target_link_libraries(test_edge_op_registration PRIVATE
  ${TEST_DEPENDENCIES}
)
if((CMAKE_CXX_COMPILER_ID MATCHES "AppleClang") OR (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  target_link_options(test_edge_op_registration PRIVATE
          "-Wl,-force_load,$<TARGET_FILE:unbox_lib>"
          )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_link_options(test_edge_op_registration PRIVATE
          "-Wl,--whole-archive,$<TARGET_FILE:unbox_lib>,--no-whole-archive"
          )
endif()
if(INSTALL_TEST)
  install(TARGETS test_edge_op_registration DESTINATION bin)
endif()
