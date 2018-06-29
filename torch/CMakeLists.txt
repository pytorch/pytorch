if (NOT CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
  include(CMakeDependentOption)
  option(USE_CUDA "Use CUDA" ON)
  option(TORCH_BUILD_TEST "Build torch test binaries" ON)

  # Flag for shared dependencies
  set(BUILD_TORCH ON)
endif()

cmake_policy(VERSION 3.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (VERBOSE)
  message(STATUS "CAFFE2_PATH is ${CAFFE2_PATH}")
  message(STATUS "CAFFE2_BUILD_PATH is ${CAFFE2_BUILD_PATH}")
  message(STATUS "INSTALL_PREFIX is ${INSTALL_PREFIX}")
endif()

set(CAFFE2_INCLUDE_DIR "${CAFFE2_PATH}")
set(CAFFE2_BUILD_LIB_DIR "${CAFFE2_BUILD_PATH}/lib")
set(CAFFE2_INSTALL_INCLUDE_DIR "${INSTALL_PREFIX}/include")
set(CAFFE2_INSTALL_SHARE_DIR "${INSTALL_PREFIX}/share")
set(CAFFE2_INSTALL_LIB_DIR "${INSTALL_PREFIX}/lib")
set(TORCH_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

find_library(CAFFE2_LIBRARY caffe2
  NAMES libcaffe2.so libcaffe2.dylib caffe2.lib
  PATHS ${CAFFE2_INSTALL_LIB_DIR} NO_DEFAULT_PATH)
find_library(CAFFE2_GPU_LIBRARY caffe2_gpu
  NAMES libcaffe2_gpu.so libcaffe2_gpu.dylib caffe2_gpu.lib
  PATHS ${CAFFE2_INSTALL_LIB_DIR} NO_DEFAULT_PATH)
find_library(PROTOBUF_LIBRARY protobuf
  NAMES libprotobuf.a libprotobufd.a libprotobuf.lib libprotobufd.lib
  PATHS ${CAFFE2_BUILD_LIB_DIR} NO_DEFAULT_PATH)

add_subdirectory(../third_party/nanopb protobuf-nanopb)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(USE_CUDA)
  set(CMAKE_MODULE_PATH
    ${INSTALL_PREFIX}/share/cmake
    ${TORCH_SRC_DIR}/../cmake/Modules
    ${TORCH_SRC_DIR}/../cmake/public
    ${TORCH_SRC_DIR}/../cmake/Modules_CUDA_fix
    /usr/lib/x86_64-linux-gnu/
    ${CMAKE_MODULE_PATH})
  set(CMAKE_LIBRARY_PATH /usr/lib/x86_64-linux-gnu/ ${CMAKE_LIBRARY_PATH})

  if(NOT CUDA_FOUND)
    find_package(CUDA 7.0)
  endif()

  find_package(MAGMA)
  if(CUDA_FOUND AND MAGMA_FOUND)
    include_directories("${MAGMA_INCLUDE_DIR}")
    set(CMAKE_REQUIRED_INCLUDES "${MAGMA_INCLUDE_DIR};${CUDA_INCLUDE_DIRS}")
    include(CheckPrototypeDefinition)
    check_prototype_definition(magma_get_sgeqrf_nb
     "magma_int_t magma_get_sgeqrf_nb( magma_int_t m, magma_int_t n );"
     "0"
     "magma.h"
      MAGMA_V2)
    IF (MAGMA_V2)
      add_definitions(-DMAGMA_V2)
    endif (MAGMA_V2)

    set(USE_MAGMA 1)
    if(VERBOSE)
      message(STATUS "Compiling with MAGMA support")
      message(STATUS "MAGMA INCLUDE DIRECTORIES: ${MAGMA_INCLUDE_DIR}")
      message(STATUS "MAGMA LIBRARIES: ${MAGMA_LIBRARIES}")
      message(STATUS "MAGMA V2 check: ${MAGMA_V2}")
    endif()
  else()
    message(STATUS "MAGMA not found. Compiling without MAGMA support")
  endif()
endif()

add_definitions(-DUSE_CATCH -D_FORCE_INLINES -DONNX_NAMESPACE=${ONNX_NAMESPACE})

if(NOT TORCH_INSTALL_BIN_DIR)
  set(TORCH_INSTALL_BIN_DIR bin)
endif()

if(NOT TORCH_INSTALL_INCLUDE_DIR)
  set(TORCH_INSTALL_INCLUDE_DIR include/libtorch)
endif()

if(NOT TORCH_INSTALL_LIB_DIR)
  set(TORCH_INSTALL_LIB_DIR lib)
endif()

if(USE_CUDA)
  add_definitions(-DUSE_CUDA)

  set(TORCH_CUDA_LIBRARIES
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libcuda.so
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvrtc.so
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvToolsExt.so
    ${CUDA_LIBRARIES})

  list(APPEND CUDA_INCLUDE_DIRS
    ${CAFFE2_INSTALL_INCLUDE_DIR}/THC)
endif()

# RPATH stuff
# see https://cmake.org/Wiki/CMake_RPATH_handling
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
endif()
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()


# Generate files
set(TOOLS_PATH "${TORCH_SRC_DIR}/../tools")

configure_file("${CAFFE2_PATH}/aten/src/ATen/common_with_cwrap.py"
               "${TOOLS_PATH}/shared/cwrap_common.py"
               COPYONLY)

configure_file("${CAFFE2_PATH}/torch/_utils_internal.py"
               "${TOOLS_PATH}/shared/_utils_internal.py"
               COPYONLY)

add_custom_command(
  OUTPUT
  "${TORCH_SRC_DIR}/csrc/nn/THNN.cpp"
  "${TORCH_SRC_DIR}/csrc/nn/THCUNN.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType.h"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/Functions.h"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/Functions.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_functions.h"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_functions.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_variable_methods.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_variable_methods_dispatch.h"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_torch_functions.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_torch_functions_dispatch.h"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_nn_functions.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_nn_functions.h"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_nn_functions_dispatch.h"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/variable_factories.h"
  "${TORCH_SRC_DIR}/csrc/jit/generated/aten_dispatch.cpp"
  "${TORCH_SRC_DIR}/csrc/jit/generated/aten_schema.cpp"
  "${TORCH_SRC_DIR}/csrc/jit/generated/aten_interned_strings.h"
  COMMAND
  python tools/setup_helpers/generate_code.py
    --declarations-path "${CAFFE2_INSTALL_SHARE_DIR}/ATen/Declarations.yaml"
    --nn-path "aten/src/"
  DEPENDS
  "${CAFFE2_INSTALL_SHARE_DIR}/ATen/Declarations.yaml"
  "${CAFFE2_INSTALL_INCLUDE_DIR}/THNN/generic/THNN.h"
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
  "${TOOLS_PATH}/autograd/templates/variable_factories.h"
  "${TOOLS_PATH}/autograd/gen_autograd.py"
  "${TOOLS_PATH}/autograd/gen_autograd_functions.py"
  "${TOOLS_PATH}/autograd/gen_variable_type.py"
  "${TOOLS_PATH}/jit/templates/aten_dispatch.cpp"
  "${TOOLS_PATH}/jit/templates/aten_interned_strings.h"
  WORKING_DIRECTORY "${TORCH_SRC_DIR}/..")

set(TORCH_SRCS
  ${TORCH_SRC_DIR}/csrc/autograd/aten_variable_hooks.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/generated/Functions.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/profiler.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/saved_variable.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/grad_mode.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/anomaly_mode.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/function.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/input_buffer.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/functions/utils.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/functions/special.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/functions/basic_ops.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/functions/accumulate_grad.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/functions/tensor.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/variable.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/engine.cpp
  ${TORCH_SRC_DIR}/csrc/assertions.cpp
  ${TORCH_SRC_DIR}/csrc/utils/variadic.cpp
  ${TORCH_SRC_DIR}/csrc/jit/generated/aten_dispatch.cpp
  ${TORCH_SRC_DIR}/csrc/jit/generated/aten_schema.cpp
  ${TORCH_SRC_DIR}/csrc/jit/variable_flags.cpp
  ${TORCH_SRC_DIR}/csrc/jit/interpreter.cpp
  ${TORCH_SRC_DIR}/csrc/jit/ir.cpp
  ${TORCH_SRC_DIR}/csrc/jit/graph_executor.cpp
  ${TORCH_SRC_DIR}/csrc/jit/fusion_compiler.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/graph_fuser.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/common_subexpression_elimination.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/shape_analysis.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/canonicalize.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/dead_code_elimination.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/erase_number_types.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/lower_tuples.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/lower_grad_of.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/peephole.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/inplace_check.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/batch_mm.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/create_autodiff_subgraphs.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/remove_expands.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/decompose_addmm.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/specialize_undef.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/loop_unrolling.cpp
  ${TORCH_SRC_DIR}/csrc/jit/interned_strings.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/compiler.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/lexer.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/module.cpp
  ${TORCH_SRC_DIR}/csrc/jit/tracer.cpp
  ${TORCH_SRC_DIR}/csrc/jit/tracer_state.cpp
  ${TORCH_SRC_DIR}/csrc/jit/autodiff.cpp
  ${TORCH_SRC_DIR}/csrc/jit/type.cpp
  ${TORCH_SRC_DIR}/csrc/jit/export.cpp
  ${TORCH_SRC_DIR}/csrc/jit/import.cpp
  ${TORCH_SRC_DIR}/csrc/onnx/onnx.cpp
  ${TORCH_SRC_DIR}/csrc/onnx/onnx.npb.cpp
  ${TORCH_SRC_DIR}/csrc/torch.cpp)

if (NOT NO_API)
  list(APPEND TORCH_SRCS
    ${TORCH_SRC_DIR}/csrc/api/src/utils.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/cuda.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/cursor.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/module.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/batchnorm.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/conv.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/dropout.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/embedding.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/functional.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/linear.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/rnn.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/optimizer.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/adam.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/adagrad.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/lbfgs.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/rmsprop.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/sgd.cpp
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
  ${TORCH_CUDA_LIBRARIES}
  ${CAFFE2_LIBRARY}
  ${PROTOBUF_LIBRARY}
  protobuf-nanopb
)
if(USE_CUDA)
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    target_link_libraries(torch -Wl,--no-as-needed ${CAFFE2_GPU_LIBRARY} -Wl,--as-needed)
  else()
    target_link_libraries(torch ${CAFFE2_GPU_LIBRARY})
  endif()
endif()

target_include_directories(torch
  PUBLIC
  "${CAFFE2_INCLUDE_DIR}"
  "${CAFFE2_INSTALL_INCLUDE_DIR}"
  "${CAFFE2_INSTALL_INCLUDE_DIR}/TH"
  "${TORCH_SRC_DIR}/.."
  "${TORCH_SRC_DIR}")

if (NOT NO_API)
  target_include_directories(torch PUBLIC
    "${TORCH_SRC_DIR}/csrc/api/"
    "${TORCH_SRC_DIR}/csrc/api/include")
endif()

# SYSTEM headers are included with -isystem and thus do not trigger warnings.
target_include_directories(torch SYSTEM PUBLIC
  "${TORCH_SRC_DIR}/../third_party/cereal/include" # For cereal/
  "${TORCH_SRC_DIR}/../third_party/nanopb")

if(USE_CUDA)
  target_include_directories(torch SYSTEM PUBLIC "${CUDA_INCLUDE_DIRS}")
endif()

set_target_properties(torch PROPERTIES VERSION 1 SOVERSION 1)

if(NOT ${CMAKE_VERSION} VERSION_LESS "3.1")
  set_property(TARGET torch PROPERTY CXX_STANDARD 11)
endif()

install(DIRECTORY "${TORCH_SRC_DIR}/csrc"
        DESTINATION ${TORCH_INSTALL_INCLUDE_DIR}/torch
        FILES_MATCHING PATTERN "*.h")

install(TARGETS torch
  RUNTIME DESTINATION "${TORCH_INSTALL_BIN_DIR}"
  LIBRARY DESTINATION "${TORCH_INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${TORCH_INSTALL_LIB_DIR}")

if (TORCH_BUILD_TEST)
  # JIT Tests. TODO: Put into test/cpp/jit folder

  add_executable(test_jit ${TORCH_SRC_DIR}/csrc/jit/test_jit.cpp)

  target_link_libraries(test_jit torch)

  target_include_directories(test_jit PUBLIC
    "${TORCH_SRC_DIR}/../third_party/catch/single_include")

  # API Tests

  if (NOT NO_API)
    set(TORCH_API_TEST_DIR "${TORCH_SRC_DIR}/../test/cpp/api")

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
        "${TORCH_SRC_DIR}/../third_party/catch/single_include")

    target_link_libraries(test_api torch)
  endif()
endif()
