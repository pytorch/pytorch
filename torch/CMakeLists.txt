if (CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  if (NOT BUILD_TORCH)
    return()
  endif()
else()
  cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
  project(torch CXX C)
  find_package(Caffe2 REQUIRED)
  option(USE_CUDA "Use CUDA" ON)
  option(TORCH_BUILD_TEST "Build torch test binaries" ON)
endif()

if(NOT TORCH_INSTALL_BIN_DIR)
  set(TORCH_INSTALL_BIN_DIR bin)
endif()

if(NOT TORCH_INSTALL_INCLUDE_DIR)
  set(TORCH_INSTALL_INCLUDE_DIR include/libtorch)
endif()

if(NOT TORCH_INSTALL_LIB_DIR)
  set(TORCH_INSTALL_LIB_DIR lib)
endif()

# Generate files
set(TOOLS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../tools")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/ATen/common_with_cwrap.py"
               "${TOOLS_PATH}/shared/cwrap_common.py"
               COPYONLY)

configure_file("${CAFFE2_PATH}/torch/_utils_internal.py"
               "${TOOLS_PATH}/shared/_utils_internal.py"
               COPYONLY)

add_custom_command(
  OUTPUT
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/nn/THNN.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/nn/THCUNN.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/generated/VariableType.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/generated/VariableType.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/generated/Functions.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/generated/Functions.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/generated/python_functions.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/generated/python_functions.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/generated/python_variable_methods.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/generated/python_variable_methods_dispatch.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/generated/python_torch_functions.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/generated/python_torch_functions_dispatch.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/generated/python_nn_functions.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/generated/python_nn_functions.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/generated/python_nn_functions_dispatch.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/generated/aten_dispatch.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/generated/aten_schema.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/generated/aten_interned_strings.h"
  COMMAND
  python tools/setup_helpers/generate_code.py
    --declarations-path "${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml"
    --nn-path "aten/src/"
  DEPENDS
  "${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml"
  "${CMAKE_CURRENT_LIST_DIR}/../aten/src/THNN/generic/THNN.h"
  "${TOOLS_PATH}/autograd/templates/VariableType.h"
  "${TOOLS_PATH}/autograd/templates/VariableType.cpp"
  "${TOOLS_PATH}/autograd/templates/Functions.h"
  "${TOOLS_PATH}/autograd/templates/Functions.cpp"
  "${TOOLS_PATH}/autograd/templates/python_functions.h"
  "${TOOLS_PATH}/autograd/templates/python_functions.cpp"
  "${TOOLS_PATH}/autograd/templates/python_variable_methods.cpp"
  "${TOOLS_PATH}/autograd/templates/python_variable_methods_dispatch.h"
  "${TOOLS_PATH}/autograd/templates/python_torch_functions.cpp"
  "${TOOLS_PATH}/autograd/templates/python_torch_functions_dispatch.h"
  "${TOOLS_PATH}/autograd/templates/python_nn_functions.cpp"
  "${TOOLS_PATH}/autograd/templates/python_nn_functions.h"
  "${TOOLS_PATH}/autograd/templates/python_nn_functions_dispatch.h"
  "${TOOLS_PATH}/autograd/gen_autograd.py"
  "${TOOLS_PATH}/autograd/gen_autograd_functions.py"
  "${TOOLS_PATH}/autograd/gen_variable_type.py"
  "${TOOLS_PATH}/jit/templates/aten_dispatch.cpp"
  "${TOOLS_PATH}/jit/templates/aten_interned_strings.h"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")

set(TORCH_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/aten_variable_hooks.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/generated/VariableType.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/generated/Functions.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/profiler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/saved_variable.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/grad_mode.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/anomaly_mode.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/function.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/input_buffer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/functions/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/functions/special.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/functions/basic_ops.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/functions/accumulate_grad.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/functions/tensor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/variable.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/autograd/engine.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/assertions.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/utils/variadic.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/generated/aten_dispatch.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/generated/aten_schema.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/variable_flags.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/interpreter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/ir.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/graph_executor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/fusion_compiler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/passes/graph_fuser.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/passes/common_subexpression_elimination.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/passes/shape_analysis.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/passes/canonicalize.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/passes/dead_code_elimination.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/passes/lower_tuples.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/passes/peephole.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/passes/inplace_check.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/passes/batch_mm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/passes/create_autodiff_subgraphs.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/passes/remove_expands.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/passes/decompose_addmm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/passes/loop_unrolling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/interned_strings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/script/compiler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/script/lexer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/script/module.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/tracer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/tracer_state.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/autodiff.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/type.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/export.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/import.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/onnx/onnx.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/onnx/onnx.pb.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/torch.cpp)

if (NOT NO_API)
  list(APPEND TORCH_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/api/src/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/api/src/cuda.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/api/src/nn/cursor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/api/src/nn/module.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/api/src/nn/modules/batchnorm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/api/src/nn/modules/conv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/api/src/nn/modules/dropout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/api/src/nn/modules/embedding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/api/src/nn/modules/functional.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/api/src/nn/modules/linear.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/api/src/nn/modules/rnn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/api/src/optimizers.cpp
  )
endif()

add_library(torch SHARED ${TORCH_SRCS})

# https://gcc.gnu.org/onlinedocs/gcc-4.0.3/gcc/Warning-Options.html
target_compile_options(torch
  PRIVATE
  -Wall
  -Wextra
  -pedantic
  -Wcast-align
  -Wcast-qual
  -Wctor-dtor-privacy
  -Wdisabled-optimization
  -Winit-self
  -Wmissing-include-dirs
  -Woverloaded-virtual
  -Wsign-promo
  -Wstrict-overflow=5
  -Wundef
  -fdiagnostics-show-option
  -Wno-unused-parameter
  -Wno-missing-braces # This warning is buggy
  -Wno-unknown-pragmas)

if ($ENV{WERROR})
  target_compile_options(torch PRIVATE -Werror)
endif()

target_link_libraries(torch
  caffe2_library)
if(USE_CUDA)
  target_link_libraries(torch caffe2_gpu_library)
endif()
if(USE_ROCM)
  target_link_libraries(torch caffe2_hip_library)
endif()

set(TH_CPU_INCLUDE
  # dense
  ${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/TH
  ${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/THC
  ${CMAKE_CURRENT_BINARY_DIR}/../aten/src/TH
  ${CMAKE_CURRENT_BINARY_DIR}/../aten/src/THC
  # sparse
  ${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/THS
  ${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/THCS
  ${CMAKE_CURRENT_BINARY_DIR}/../aten/src/THS
  ${CMAKE_CURRENT_BINARY_DIR}/../aten/src/THCS

  ${CMAKE_CURRENT_SOURCE_DIR}/../aten/src
  ${CMAKE_CURRENT_BINARY_DIR}/../aten/src
  ${CMAKE_BINARY_DIR}/aten/src)
target_include_directories(torch PRIVATE ${TH_CPU_INCLUDE})

set(ATen_CPU_INCLUDE
  ${CMAKE_CURRENT_SOURCE_DIR}/../aten/src
  ${CMAKE_CURRENT_BINARY_DIR}/../aten/src/ATen)
target_include_directories(torch PRIVATE ${ATen_CPU_INCLUDE})

target_include_directories(torch PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc)

if (NOT NO_API)
  target_include_directories(torch PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/api
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/api/include)
endif()

# SYSTEM headers are included with -isystem and thus do not trigger warnings.
target_include_directories(torch SYSTEM PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cereal/include" # For cereal/
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/nanopb")

set_target_properties(torch PROPERTIES VERSION 1 SOVERSION 1)

if(NOT ${CMAKE_VERSION} VERSION_LESS "3.1")
  set_property(TARGET torch PROPERTY CXX_STANDARD 11)
endif()

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
        DESTINATION ${TORCH_INSTALL_INCLUDE_DIR}/torch
        FILES_MATCHING PATTERN "*.h")

install(TARGETS torch
  RUNTIME DESTINATION "${TORCH_INSTALL_BIN_DIR}"
  LIBRARY DESTINATION "${TORCH_INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${TORCH_INSTALL_LIB_DIR}")

if (TORCH_BUILD_TEST)
  # JIT Tests. TODO: Put into test/cpp/jit folder

  add_executable(test_jit ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/test_jit.cpp)

  target_link_libraries(test_jit torch)

  target_include_directories(test_jit PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/catch/single_include")

  # API Tests

  if (NOT NO_API)
    set(TORCH_API_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../test/cpp/api")

    add_executable(test_api
      ${TORCH_API_TEST_DIR}/any.cpp
      ${TORCH_API_TEST_DIR}/modules.cpp
      ${TORCH_API_TEST_DIR}/cursor.cpp
      ${TORCH_API_TEST_DIR}/integration.cpp
      ${TORCH_API_TEST_DIR}/main.cpp
      ${TORCH_API_TEST_DIR}/misc.cpp
      ${TORCH_API_TEST_DIR}/module.cpp
      ${TORCH_API_TEST_DIR}/optim.cpp
      ${TORCH_API_TEST_DIR}/sequential.cpp
      ${TORCH_API_TEST_DIR}/rnn.cpp
      ${TORCH_API_TEST_DIR}/serialization.cpp
      ${TORCH_API_TEST_DIR}/static.cpp
      ${TORCH_API_TEST_DIR}/tensor.cpp
      ${TORCH_API_TEST_DIR}/tensor_cuda.cpp
      # Temporary until ATen tests are built with Caffe2
      ${TORCH_API_TEST_DIR}/tensor_options.cpp
      ${TORCH_API_TEST_DIR}/tensor_options_cuda.cpp
    )

      target_include_directories(test_api
        PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/catch/single_include")

    target_link_libraries(test_api torch)
  endif()
endif()
