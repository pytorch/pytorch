if (CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  if (NOT BUILD_TORCH)
    return()
  endif()
else()
  cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
  project(torch CXX C)
  find_package(Caffe2 REQUIRED)
  option(USE_CUDA "Use CUDA" ON)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

option(TORCH_STATIC "Build libtorch.a rather than libtorch.so" OFF)

set(TORCH_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TORCH_ROOT "${TORCH_SRC_DIR}/..")

if(NOT TORCH_INSTALL_BIN_DIR)
  set(TORCH_INSTALL_BIN_DIR bin)
endif()

if(NOT TORCH_INSTALL_INCLUDE_DIR)
  set(TORCH_INSTALL_INCLUDE_DIR include)
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
  "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_0.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_1.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_2.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_3.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_4.cpp"
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
  "${TORCH_SRC_DIR}/csrc/jit/generated/register_aten_ops_0.cpp"
  "${TORCH_SRC_DIR}/csrc/jit/generated/register_aten_ops_1.cpp"
  "${TORCH_SRC_DIR}/csrc/jit/generated/register_aten_ops_2.cpp"
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
  "${TOOLS_PATH}/autograd/deprecated.yaml"
  "${TOOLS_PATH}/autograd/derivatives.yaml"
  "${TOOLS_PATH}/autograd/gen_autograd_functions.py"
  "${TOOLS_PATH}/autograd/gen_autograd.py"
  "${TOOLS_PATH}/autograd/gen_python_functions.py"
  "${TOOLS_PATH}/autograd/gen_variable_factories.py"
  "${TOOLS_PATH}/autograd/gen_variable_type.py"
  "${TOOLS_PATH}/autograd/load_derivatives.py"
  "${TOOLS_PATH}/autograd/nested_dict.py"
  "${TOOLS_PATH}/autograd/utils.py"
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
  ${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_0.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_1.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_2.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_3.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_4.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/grad_mode.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/input_buffer.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/profiler.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/saved_variable.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/variable.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/VariableTypeManual.cpp
  ${TORCH_SRC_DIR}/csrc/cuda/comm.cpp
  ${TORCH_SRC_DIR}/csrc/jit/autodiff.cpp
  ${TORCH_SRC_DIR}/csrc/jit/export.cpp
  ${TORCH_SRC_DIR}/csrc/jit/generated/register_aten_ops_0.cpp
  ${TORCH_SRC_DIR}/csrc/jit/generated/register_aten_ops_1.cpp
  ${TORCH_SRC_DIR}/csrc/jit/generated/register_aten_ops_2.cpp
  ${TORCH_SRC_DIR}/csrc/jit/graph_executor.cpp
  ${TORCH_SRC_DIR}/csrc/jit/import.cpp
  ${TORCH_SRC_DIR}/csrc/jit/interpreter.cpp
  ${TORCH_SRC_DIR}/csrc/jit/constants.cpp
  ${TORCH_SRC_DIR}/csrc/jit/node_hashing.cpp
  ${TORCH_SRC_DIR}/csrc/jit/ir.cpp
  ${TORCH_SRC_DIR}/csrc/jit/operator.cpp
  ${TORCH_SRC_DIR}/csrc/jit/operator.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/batch_mm.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/canonicalize.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/constant_propagation.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/constant_pooling.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/common_subexpression_elimination.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/create_autodiff_subgraphs.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/inline_autodiff_subgraphs.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/dead_code_elimination.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/canonicalize_ops.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/erase_number_types.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/graph_fuser.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/inplace_check.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/loop_unrolling.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/lower_grad_of.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/lower_tuples.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/peephole.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/remove_expands.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/remove_inplace_ops.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/shape_analysis.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/requires_grad_analysis.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/specialize_undef.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/python_print.cpp
  ${TORCH_SRC_DIR}/csrc/jit/fuser/interface.cpp
  ${TORCH_SRC_DIR}/csrc/jit/register_prim_ops.cpp
  ${TORCH_SRC_DIR}/csrc/jit/register_special_ops.cpp
  ${TORCH_SRC_DIR}/csrc/jit/scope.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/compiler.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/builtin_functions.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/lexer.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/module.cpp
  ${TORCH_SRC_DIR}/csrc/jit/tracer.cpp
  ${TORCH_SRC_DIR}/csrc/torch.cpp
  ${TORCH_SRC_DIR}/csrc/utils/tensor_flatten.cpp
  ${TORCH_SRC_DIR}/csrc/utils/variadic.cpp
  ${TORCH_ROOT}/test/cpp/jit/no-gtest.cpp
  )

SET(USE_CPU_FUSER 0)
if (NOT WIN32)
  SET(USE_CPU_FUSER 1)

  list(APPEND TORCH_SRCS
    ${TORCH_SRC_DIR}/csrc/jit/fuser/kernel_cache.cpp
    ${TORCH_SRC_DIR}/csrc/jit/fuser/compiler.cpp
    ${TORCH_SRC_DIR}/csrc/jit/fuser/executor.cpp
    ${TORCH_SRC_DIR}/csrc/jit/fuser/codegen.cpp
    ${TORCH_SRC_DIR}/csrc/jit/fuser/fallback.cpp
    ${TORCH_SRC_DIR}/csrc/jit/fuser/cpu/fused_kernel.cpp
  )
endif()

SET(USE_CUDA_FUSER 0)
if (USE_CUDA AND NOT USE_ROCM AND NOT WIN32)
  SET(USE_CUDA_FUSER 1)

  list(APPEND TORCH_SRCS
    ${TORCH_SRC_DIR}/csrc/jit/fuser/cuda/fused_kernel.cpp
  )

endif()

CONFIGURE_FILE(
    ${TORCH_SRC_DIR}/csrc/jit/fuser/config.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/jit/fuser/config.h)

if (NOT NO_API AND NOT USE_ROCM)
  list(APPEND TORCH_SRCS
    ${TORCH_SRC_DIR}/csrc/api/src/cuda.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/data/datasets/mnist.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/data/samplers/random.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/data/samplers/sequential.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/data/samplers/stream.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/jit.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/init.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/module.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/batchnorm.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/conv.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/dropout.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/embedding.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/functional.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/linear.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/rnn.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/adagrad.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/adam.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/lbfgs.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/optimizer.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/rmsprop.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/serialize.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/sgd.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/serialize/input-archive.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/serialize/output-archive.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/utils.cpp
  )
endif()


if (TORCH_STATIC)
  add_library(torch STATIC ${TORCH_SRCS})
else()
  add_library(torch SHARED ${TORCH_SRCS})
endif()

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

if (MSVC)
elseif ($ENV{WERROR})
  target_compile_options(torch PRIVATE -Werror -Wno-strict-overflow)
endif()

if (MSVC)
  target_link_libraries(torch onnx onnx_library)
endif()

target_link_libraries(torch caffe2_library)

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
    if (NOT NVTOOLEXT_HOME)
      set(NVTOOLEXT_HOME "C:/Program Files/NVIDIA Corporation/NvToolsExt")
    endif()
    if ($ENV{NVTOOLEXT_HOME})
      set(NVTOOLEXT_HOME $ENV{NVTOOLEXT_HOME})
    endif()
    set(TORCH_CUDA_LIBRARIES
      ${NVTOOLEXT_HOME}/lib/x64/nvToolsExt64_1.lib
      ${CUDA_LIBRARIES})
    target_include_directories(torch PUBLIC "${NVTOOLEXT_HOME}/include")
  elseif(APPLE)
    set(TORCH_CUDA_LIBRARIES
      ${CUDA_TOOLKIT_ROOT_DIR}/lib/libcudart.dylib
      ${CUDA_TOOLKIT_ROOT_DIR}/lib/libnvrtc.dylib
      ${CUDA_TOOLKIT_ROOT_DIR}/lib/libnvToolsExt.dylib
      ${CUDA_LIBRARIES})
    set_target_properties(torch PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  else()
    set(TORCH_CUDA_LIBRARIES
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvToolsExt.so
      ${CUDA_LIBRARIES})
  endif()

  target_link_libraries(torch caffe2_gpu_library ${TORCH_CUDA_LIBRARIES})
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
    /opt/rocm/hipsparse/include
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
target_include_directories(torch PUBLIC ${ATen_CPU_INCLUDE})

target_include_directories(torch PUBLIC
  ${TORCH_SRC_DIR}/csrc)

set_target_properties(torch PROPERTIES VERSION 1 SOVERSION 1)

if(NOT ${CMAKE_VERSION} VERSION_LESS "3.1")
  set_property(TARGET torch PROPERTY CXX_STANDARD 11)
endif()

install(DIRECTORY "${TORCH_SRC_DIR}/csrc"
        DESTINATION ${TORCH_INSTALL_INCLUDE_DIR}/torch
        FILES_MATCHING PATTERN "*.h")
install(FILES "${TORCH_SRC_DIR}/script.h" "${TORCH_SRC_DIR}/extension.h"
        DESTINATION ${TORCH_INSTALL_INCLUDE_DIR}/torch)

install(TARGETS torch
  RUNTIME DESTINATION "${TORCH_INSTALL_BIN_DIR}"
  LIBRARY DESTINATION "${TORCH_INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${TORCH_INSTALL_LIB_DIR}")

if (BUILD_TEST AND NOT MSVC AND NOT USE_ROCM)
  add_subdirectory(${TORCH_ROOT}/test/cpp/jit ${CMAKE_BINARY_DIR}/test_jit)
endif()

if (BUILD_TEST AND NOT NO_API AND NOT USE_ROCM)
  add_subdirectory(${TORCH_ROOT}/test/cpp/api ${CMAKE_BINARY_DIR}/test_api)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  message(STATUS "${CMAKE_CXX_COMPILER} ${CMAKE_CURRENT_LIST_DIR}/abi-check.cpp -o ${CMAKE_BINARY_DIR}/abi-check")
  execute_process(
    COMMAND
      "${CMAKE_CXX_COMPILER}"
      "${CMAKE_CURRENT_LIST_DIR}/abi-check.cpp"
      "-o"
      "${CMAKE_BINARY_DIR}/abi-check"
    RESULT_VARIABLE ABI_CHECK_COMPILE_RESULT)
  if (ABI_CHECK_COMPILE_RESULT)
    message(FATAL_ERROR "Could not compile ABI Check: ${ABI_CHECK_COMPILE_RESULT}")
  endif()
  execute_process(
    COMMAND "${CMAKE_BINARY_DIR}/abi-check"
    RESULT_VARIABLE ABI_CHECK_RESULT
    OUTPUT_VARIABLE GLIBCXX_USE_CXX11_ABI)
  if (ABI_CHECK_RESULT)
    message(WARNING "Could not run ABI Check: ${ABI_CHECK_RESULT}")
  endif()
  message(STATUS "Determined _GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")
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
install(FILES
    ${PROJECT_BINARY_DIR}/TorchConfigVersion.cmake
    ${PROJECT_BINARY_DIR}/TorchConfig.cmake
    DESTINATION share/cmake/Torch)

if (USE_DISTRIBUTED)
  add_subdirectory(${TORCH_SRC_DIR}/lib/THD)
  if (NOT MSVC AND NOT APPLE)
    add_subdirectory(${TORCH_SRC_DIR}/lib/c10d)
  endif()
endif()

if (MSVC)
  add_subdirectory(${TORCH_SRC_DIR}/lib/libshm_windows)
else()
  add_subdirectory(${TORCH_SRC_DIR}/lib/libshm)
endif()
