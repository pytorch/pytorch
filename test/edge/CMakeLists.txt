cmake_minimum_required(VERSION 3.1)

set(TORCH_ROOT ${CMAKE_CURRENT_LIST_DIR}/../..)
set(TEST_ROOT ${TORCH_ROOT}/test/edge)


# Generate unboxing kernels
set(GEN_COMMAND
        "${PYTHON_EXECUTABLE}" -m torchgen.gen_executorch
        --source-path=${TEST_ROOT}/templates
        --install_dir=${CMAKE_BINARY_DIR}/edge
        --tags-path=${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/tags.yaml
        --aten_yaml_path=${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/native_functions.yaml
        --use_aten_lib
        --op_selection_yaml_path=${TEST_ROOT}/selected_operators.yaml
        )
set(GEN_COMMAND_sources
        ${CMAKE_BINARY_DIR}/edge/RegisterCodegenUnboxedKernelsEverything.cpp
        ${CMAKE_BINARY_DIR}/edge/Functions.h
        ${CMAKE_BINARY_DIR}/edge/NativeFunctions.h
        )
message(STATUS "Generating sources for unboxing kernels ${GEN_COMMAND}")
add_custom_command(
        COMMENT "Generating sources"
        OUTPUT ${GEN_COMMAND_sources}
        COMMAND ${GEN_COMMAND}
        DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/native_functions.yaml
        ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/tags.yaml
        ${TEST_ROOT}/templates/Functions.h
        ${TEST_ROOT}/templates/NativeFunctions.h
        ${TEST_ROOT}/templates/RegisterCodegenUnboxedKernels.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
)
add_custom_target(UNBOX_TARGET DEPENDS ${GEN_COMMAND_sources})
add_executable(test_edge_op_registration
        ${TEST_ROOT}/test_operator_registration.cpp
        ${TEST_ROOT}/test_main.cpp
        ${UNBOX_TARGET}
        )

target_include_directories(test_edge_op_registration PRIVATE ${ATen_CPU_INCLUDE})

target_compile_definitions(test_edge_op_registration PRIVATE USE_GTEST)

set(TEST_DEPENDENCIES torch gtest)

target_link_libraries(test_edge_op_registration PRIVATE
  ${TEST_DEPENDENCIES}
)

if(INSTALL_TEST)
  install(TARGETS test_edge_op_registration DESTINATION bin)
endif()
