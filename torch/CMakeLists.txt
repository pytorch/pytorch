if (CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  if (NOT BUILD_TORCH)
    return()
  endif()
else()
  cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
  project(torch CXX C)
  find_package(Caffe2 REQUIRED)
  option(USE_CUDA "Use CUDA" ON)
endif()

option(BUILD_TORCH_TEST "Build torch test binaries" ON)

# TODO: Unify with version from setup.py
set(TORCH_VERSION_MAJOR 0)
set(TORCH_VERSION_MINOR 4)
set(TORCH_VERSION_PATCH 1)
set(TORCH_VERSION "${TORCH_VERSION_MAJOR}.${TORCH_VERSION_MINOR}.${TORCH_VERSION_PATCH}")

set(TORCH_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TORCH_ROOT "${TORCH_SRC_DIR}/..")

add_subdirectory(../third_party/nanopb protobuf-nanopb)

if(NOT TORCH_INSTALL_BIN_DIR)
  set(TORCH_INSTALL_BIN_DIR bin)
endif()

if(NOT TORCH_INSTALL_INCLUDE_DIR)
  set(TORCH_INSTALL_INCLUDE_DIR include/libtorch)
endif()

if(NOT TORCH_INSTALL_LIB_DIR)
  set(TORCH_INSTALL_LIB_DIR lib)
endif()

# RPATH stuff
# see https://cmake.org/Wiki/CMake_RPATH_handling
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
endif()
# Use separate rpaths during build and install phases
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
# Don't use the install-rpath during the build phase
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
# Automatically add all linked folders that are NOT in the build directory to
# the rpath (per library?)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# Always ensure that CMAKE_INSTALL_PREFIX/lib is in the rpath
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
  list(APPEND CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()

# Get the correct Python executable
if (DEFINED ENV{PYTORCH_PYTHON})
  message(STATUS "Using python found in $ENV{PYTORCH_PYTHON}")
  set(PYCMD "$ENV{PYTORCH_PYTHON}")
else()
  SET(PYCMD "python")
endif()

# Generate files
set(TOOLS_PATH "${TORCH_ROOT}/tools")

configure_file("${TORCH_ROOT}/aten/src/ATen/common_with_cwrap.py"
               "${TOOLS_PATH}/shared/cwrap_common.py"
               COPYONLY)

configure_file("${TORCH_SRC_DIR}/_utils_internal.py"
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
  "${TORCH_SRC_DIR}/csrc/jit/generated/register_aten_ops.cpp"
  "${TORCH_SRC_DIR}/csrc/jit/generated/aten_interned_strings.h"
  COMMAND
  ${PYCMD} tools/setup_helpers/generate_code.py
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
  "${TOOLS_PATH}/autograd/templates/variable_factories.h"
  "${TOOLS_PATH}/autograd/gen_autograd.py"
  "${TOOLS_PATH}/autograd/gen_autograd_functions.py"
  "${TOOLS_PATH}/autograd/gen_variable_type.py"
  "${TOOLS_PATH}/autograd/derivatives.yaml"
  "${TOOLS_PATH}/jit/gen_jit_dispatch.py"
  "${TOOLS_PATH}/jit/templates/register_aten_ops.cpp"
  "${TOOLS_PATH}/jit/templates/aten_interned_strings.h"
  WORKING_DIRECTORY "${TORCH_ROOT}")

set(TORCH_SRCS
  ${TORCH_SRC_DIR}/csrc/autograd/anomaly_mode.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/engine.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/function.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/functions/accumulate_grad.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/functions/basic_ops.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/functions/comm.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/functions/tensor.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/functions/utils.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/generated/Functions.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/grad_mode.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/input_buffer.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/profiler.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/saved_variable.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/variable.cpp
  ${TORCH_SRC_DIR}/csrc/cuda/comm.cpp
  ${TORCH_SRC_DIR}/csrc/jit/autodiff.cpp
  ${TORCH_SRC_DIR}/csrc/jit/export.cpp
  ${TORCH_SRC_DIR}/csrc/jit/fusion_compiler.cpp
  ${TORCH_SRC_DIR}/csrc/jit/generated/register_aten_ops.cpp
  ${TORCH_SRC_DIR}/csrc/jit/graph_executor.cpp
  ${TORCH_SRC_DIR}/csrc/jit/import.cpp
  ${TORCH_SRC_DIR}/csrc/jit/interned_strings.cpp
  ${TORCH_SRC_DIR}/csrc/jit/interpreter.cpp
  ${TORCH_SRC_DIR}/csrc/jit/constants.cpp
  ${TORCH_SRC_DIR}/csrc/jit/ir.cpp
  ${TORCH_SRC_DIR}/csrc/jit/ivalue.cpp
  ${TORCH_SRC_DIR}/csrc/jit/operator.cpp
  ${TORCH_SRC_DIR}/csrc/jit/operator.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/batch_mm.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/canonicalize.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/constant_propagation.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/common_subexpression_elimination.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/create_autodiff_subgraphs.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/dead_code_elimination.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/decompose_addmm.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/erase_number_types.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/graph_fuser.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/inplace_check.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/loop_unrolling.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/lower_grad_of.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/lower_tuples.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/peephole.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/remove_expands.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/shape_analysis.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/specialize_undef.cpp
  ${TORCH_SRC_DIR}/csrc/jit/register_prim_ops.cpp
  ${TORCH_SRC_DIR}/csrc/jit/register_prim_ops.cpp
  ${TORCH_SRC_DIR}/csrc/jit/register_symbols.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/compiler.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/lexer.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/module.cpp
  ${TORCH_SRC_DIR}/csrc/jit/test_jit.cpp
  ${TORCH_SRC_DIR}/csrc/jit/tracer.cpp
  ${TORCH_SRC_DIR}/csrc/jit/type.cpp
  ${TORCH_SRC_DIR}/csrc/torch.cpp
  ${TORCH_SRC_DIR}/csrc/utils/tensor_flatten.cpp
  ${TORCH_SRC_DIR}/csrc/utils/variadic.cpp
  )

if (NOT NO_API AND NOT USE_ROCM)
  list(APPEND TORCH_SRCS
    ${TORCH_SRC_DIR}/csrc/api/src/utils.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/cuda.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/cursor.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/init.cpp
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

add_library(torch ${TORCH_SRCS})

target_compile_definitions(torch PRIVATE _THP_CORE)

# until they can be unified, keep these lists synced with setup.py
if(MSVC)
  if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set (MSVC_RUNTIME_LIBRARY_FLAG "/MDd")
  else()
    set (MSVC_RUNTIME_LIBRARY_FLAG "/MD")
  endif()

  target_compile_options(torch PRIVATE
    ${MSVC_RUNTIME_LIBRARY_OPTION}
    /Z7
    /EHa
    /DNOMINMAX
    /wd4267
    /wd4251
    /wd4522
    /wd4522
    /wd4838
    /wd4305
    /wd4244
    /wd4190
    /wd4101
    /wd4996
    /wd4275
    /bigobj
    )
else()
  target_compile_options(torch PRIVATE
    -std=c++11
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-missing-field-initializers
    -Wno-write-strings
    -Wno-unknown-pragmas
    # Clang has an unfixed bug leading to spurious missing braces
    # warnings, see https://bugs.llvm.org/show_bug.cgi?id=21629
    -Wno-missing-braces
    )

  if(NOT APPLE)
    target_compile_options(torch PRIVATE
      # Considered to be flaky.  See the discussion at
      # https://github.com/pytorch/pytorch/pull/9608
      -Wno-maybe-uninitialized)
  endif()

endif()

# see the source file for explanation
set_source_files_properties(
  ${TORCH_SRC_DIR}/csrc/jit/register_symbols.cpp
  PROPERTIES COMPILE_FLAGS -O0
  )

if (MSVC)
elseif ($ENV{WERROR})
  target_compile_options(torch PRIVATE -Werror -Wno-strict-overflow)
endif()

if (MSVC)
  target_link_libraries(torch onnx onnx_library)
endif()

target_link_libraries(torch
  caffe2_library
  protobuf-nanopb)

find_package(OpenMP)
if(OPENMP_FOUND)
  if (VERBOSE)
    message(STATUS "Compiling with OpenMP")
  endif()
  target_compile_options(torch INTERFACE -fopenmp)
  target_link_libraries(torch -fopenmp)
endif()

if (NOT NO_API AND NOT USE_ROCM)
  target_include_directories(torch PUBLIC
    ${TORCH_SRC_DIR}/csrc/api
    ${TORCH_SRC_DIR}/csrc/api/include)
endif()

if(USE_CUDA)
  if(MSVC)
    set(TORCH_CUDA_LIBRARIES
      ${NVTOOLEXT_HOME}/lib/x64/nvToolsExt64_1.lib
      ${CUDA_LIBRARIES})
    target_include_directories(torch PRIVATE "${NVTOOLEXT_HOME}/include")
  elseif(APPLE)
    set(TORCH_CUDA_LIBRARIES
      ${CUDA_TOOLKIT_ROOT_DIR}/lib/libcudart.dylib
      ${CUDA_TOOLKIT_ROOT_DIR}/lib/libnvrtc.dylib
      ${CUDA_TOOLKIT_ROOT_DIR}/lib/libnvToolsExt.dylib
      ${CUDA_LIBRARIES})
    set_target_properties(torch PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  else()
    set(TORCH_CUDA_LIBRARIES
      ${CUDA_CUDA_LIB}
      ${CUDA_NVRTC_LIB}
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvToolsExt.so
      ${CUDA_LIBRARIES})
  endif()

  if(MSVC OR APPLE)
    target_link_libraries(torch caffe2_gpu_library ${TORCH_CUDA_LIBRARIES})
  else()
    # TODO: using the full TORCH_CUDA_LIBRARIES here causes some
    # builds to fail in CI, as libcuda.so can no longer be found. It's
    # not clear why this is the case, and the situation should be
    # investigated/cleaned up.  Note that the test+jit/test_api
    # targets below do require linking against the full
    # TORCH_CUDA_LIBRARIES, even on Linux
    target_link_libraries(torch caffe2_gpu_library ${CUDA_LIBRARIES})
  endif()
  target_compile_definitions(torch PRIVATE USE_CUDA)
endif()

if(USE_ROCM)
  target_link_libraries(torch caffe2_hip_library)
  target_compile_definitions(torch PRIVATE
    USE_ROCM
    __HIP_PLATFORM_HCC__
    )
  target_include_directories(torch PRIVATE
    /opt/rocm/include
    /opt/rocm/hcc/include
    /opt/rocm/rocblas/include
    /opt/rocm/hcsparse/include
    )
endif()


set(TH_CPU_INCLUDE
  # dense
  ${TORCH_ROOT}/aten/src/TH
  ${CMAKE_CURRENT_BINARY_DIR}/../aten/src/TH
  ${TORCH_ROOT}/aten/src
  ${CMAKE_CURRENT_BINARY_DIR}/../aten/src
  ${CMAKE_BINARY_DIR}/aten/src)
target_include_directories(torch PRIVATE ${TH_CPU_INCLUDE})

if(USE_CUDA OR USE_ROCM)
  set(TH_CUDA_INCLUDE
    # dense
    ${TORCH_ROOT}/aten/src/THC
    ${CMAKE_CURRENT_BINARY_DIR}/../aten/src/THC)
  target_include_directories(torch PRIVATE ${TH_CUDA_INCLUDE})
endif()

set(ATen_CPU_INCLUDE
  ${TORCH_ROOT}/aten/src
  ${CMAKE_CURRENT_BINARY_DIR}/../aten/src
  ${CMAKE_CURRENT_BINARY_DIR}/../aten/src/ATen
  ${CMAKE_BINARY_DIR}/aten/src)
target_include_directories(torch PRIVATE ${ATen_CPU_INCLUDE})

target_include_directories(torch PUBLIC
  ${TORCH_SRC_DIR}/csrc)

# SYSTEM headers are included with -isystem and thus do not trigger warnings.
target_include_directories(torch SYSTEM PUBLIC
  "${TORCH_ROOT}/third_party/cereal/include" # For cereal/
  "${TORCH_ROOT}/third_party/nanopb")

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

# JIT Tests. TODO: Put into test/cpp/jit folder
if (BUILD_TORCH_TEST AND NOT MSVC AND NOT APPLE AND NOT USE_ROCM)
  add_executable(test_jit ${TORCH_SRC_DIR}/csrc/jit/test_jit.cpp)
  target_link_libraries(test_jit torch ${TORCH_CUDA_LIBRARIES})
  target_compile_definitions(test_jit PUBLIC USE_CATCH _FORCE_INLINES)
  target_include_directories(test_jit PUBLIC
    "${TORCH_ROOT}/third_party/catch/single_include"
    ${ATen_CPU_INCLUDE})

  if (USE_CUDA)
    target_link_libraries(test_jit ${CUDA_LIBRARIES})
  endif()
endif()

if (BUILD_TORCH_TEST AND NOT NO_API AND NOT USE_ROCM)
  set(TORCH_API_TEST_DIR "${TORCH_ROOT}/test/cpp/api")

  add_executable(test_api
    ${TORCH_API_TEST_DIR}/any.cpp
    ${TORCH_API_TEST_DIR}/cursor.cpp
    ${TORCH_API_TEST_DIR}/integration.cpp
    ${TORCH_API_TEST_DIR}/main.cpp
    ${TORCH_API_TEST_DIR}/misc.cpp
    ${TORCH_API_TEST_DIR}/module.cpp
    ${TORCH_API_TEST_DIR}/modules.cpp
    ${TORCH_API_TEST_DIR}/optim.cpp
    ${TORCH_API_TEST_DIR}/parallel.cpp
    ${TORCH_API_TEST_DIR}/rnn.cpp
    ${TORCH_API_TEST_DIR}/sequential.cpp
    ${TORCH_API_TEST_DIR}/serialization.cpp
    ${TORCH_API_TEST_DIR}/static.cpp
    ${TORCH_API_TEST_DIR}/tensor_cuda.cpp
    ${TORCH_API_TEST_DIR}/tensor.cpp
    # Temporary until ATen tests are built with Caffe2
    ${TORCH_API_TEST_DIR}/tensor_options.cpp
    ${TORCH_API_TEST_DIR}/tensor_options_cuda.cpp
    )

  target_include_directories(test_api
    PUBLIC
    "${TORCH_ROOT}/third_party/catch/single_include"
    ${ATen_CPU_INCLUDE})

  target_link_libraries(test_api torch ${TORCH_CUDA_LIBRARIES})

  if (NOT MSVC)
    if (APPLE)
      target_compile_options(test_api PRIVATE
        # Clang has an unfixed bug leading to spurious missing braces
        # warnings, see https://bugs.llvm.org/show_bug.cgi?id=21629
        -Wno-missing-braces)
    else()
      target_compile_options(test_api PRIVATE
        # Considered to be flaky.  See the discussion at
        # https://github.com/pytorch/pytorch/pull/9608
        -Wno-maybe-uninitialized
        # gcc gives nonsensical warnings about variadic.h
        -Wno-unused-but-set-parameter)
    endif()
  endif()
endif()

# CMake config for external projects.
configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/TorchConfigVersion.cmake.in
    ${PROJECT_BINARY_DIR}/TorchConfigVersion.cmake
    @ONLY)
configure_file(
    ${TORCH_ROOT}/cmake/TorchConfig.cmake.in
    ${PROJECT_BINARY_DIR}/TorchConfig.cmake
    @ONLY)
