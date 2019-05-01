if (CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  if (NOT BUILD_TORCH)
    return()
  endif()
else()
  cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
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

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

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
  COMMAND
  "${PYTHON_EXECUTABLE}" tools/setup_helpers/generate_code.py
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
  WORKING_DIRECTORY "${TORCH_ROOT}")

set(TORCH_SRCS
  ${TORCH_SRC_DIR}/csrc/autograd/anomaly_mode.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/engine.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/function.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/function_hook.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/functions/accumulate_grad.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/functions/basic_ops.cpp
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
  ${TORCH_SRC_DIR}/csrc/autograd/record_function.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/saved_variable.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/variable.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/VariableTypeManual.cpp
  ${TORCH_SRC_DIR}/csrc/jit/autodiff.cpp
  ${TORCH_SRC_DIR}/csrc/jit/attributes.cpp
  ${TORCH_SRC_DIR}/csrc/jit/argument_spec.cpp
  ${TORCH_SRC_DIR}/csrc/jit/export.cpp
  ${TORCH_SRC_DIR}/csrc/jit/pass_manager.cpp
  ${TORCH_SRC_DIR}/csrc/jit/pickler.cpp
  ${TORCH_SRC_DIR}/csrc/jit/generated/register_aten_ops_0.cpp
  ${TORCH_SRC_DIR}/csrc/jit/generated/register_aten_ops_1.cpp
  ${TORCH_SRC_DIR}/csrc/jit/generated/register_aten_ops_2.cpp
  ${TORCH_SRC_DIR}/csrc/jit/graph_executor.cpp
  ${TORCH_SRC_DIR}/csrc/jit/import_source.cpp
  ${TORCH_SRC_DIR}/csrc/jit/import.cpp
  ${TORCH_SRC_DIR}/csrc/jit/import_export_helpers.cpp
  ${TORCH_SRC_DIR}/csrc/jit/interpreter.cpp
  ${TORCH_SRC_DIR}/csrc/jit/constants.cpp
  ${TORCH_SRC_DIR}/csrc/jit/node_hashing.cpp
  ${TORCH_SRC_DIR}/csrc/jit/ir.cpp
  ${TORCH_SRC_DIR}/csrc/jit/irparser.cpp
  ${TORCH_SRC_DIR}/csrc/jit/netdef_converter.cpp
  ${TORCH_SRC_DIR}/csrc/jit/operator.cpp
  ${TORCH_SRC_DIR}/csrc/jit/register_c10_ops.cpp
  ${TORCH_SRC_DIR}/csrc/jit/subgraph_matcher.cpp
  ${TORCH_SRC_DIR}/csrc/jit/symbolic_script.cpp
  ${TORCH_SRC_DIR}/csrc/jit/profiling_record.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/alias_analysis.cpp
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
  ${TORCH_SRC_DIR}/csrc/jit/passes/inline_fork_wait.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/graph_fuser.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/inplace_check.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/loop_unrolling.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/lower_grad_of.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/lower_tuples.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/pattern_fusion.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/peephole.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/remove_expands.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/remove_inplace_ops.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/shape_analysis.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/requires_grad_analysis.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/specialize_autogradzero.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/python_print.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/utils/subgraph_utils.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/utils/check_alias_annotation.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/utils/memory_dag.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/quantization.cpp
  ${TORCH_SRC_DIR}/csrc/jit/fuser/interface.cpp
  ${TORCH_SRC_DIR}/csrc/jit/register_prim_ops.cpp
  ${TORCH_SRC_DIR}/csrc/jit/register_special_ops.cpp
  ${TORCH_SRC_DIR}/csrc/jit/register_quantized_ops.cpp
  ${TORCH_SRC_DIR}/csrc/jit/scope.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/compiler.cpp
  ${TORCH_SRC_DIR}/csrc/api/src/jit.cpp
  ${TORCH_SRC_DIR}/csrc/jit/testing/file_check.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/final_returns.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/schema_matching.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/script_type_parser.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/sugared_value.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/class_type.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/parser.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/builtin_functions.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/edit_distance.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/logging.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/module.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/jit_exception.cpp
  ${TORCH_SRC_DIR}/csrc/jit/tracer.cpp
  ${TORCH_SRC_DIR}/csrc/jit/hooks_for_testing.cpp
  ${TORCH_SRC_DIR}/csrc/utils/tensor_flatten.cpp
  ${TORCH_SRC_DIR}/csrc/utils/variadic.cpp
  ${TORCH_SRC_DIR}/csrc/jit/fuser/kernel_cache.cpp
  ${TORCH_SRC_DIR}/csrc/jit/fuser/compiler.cpp
  ${TORCH_SRC_DIR}/csrc/jit/fuser/executor.cpp
  ${TORCH_SRC_DIR}/csrc/jit/fuser/codegen.cpp
  ${TORCH_SRC_DIR}/csrc/jit/fuser/fallback.cpp
  ${TORCH_ROOT}/test/cpp/jit/test.cpp
  )

if (WIN32)
  list(APPEND TORCH_SRCS
    ${TORCH_SRC_DIR}/csrc/jit/fuser/cpu/dynamic_library_win.cpp
  )
else ()
  list(APPEND TORCH_SRCS
    ${TORCH_SRC_DIR}/csrc/jit/fuser/cpu/dynamic_library_unix.cpp
    ${TORCH_SRC_DIR}/csrc/jit/fuser/cpu/fused_kernel.cpp
  )
  if (USE_CUDA AND NOT USE_ROCM)
    list(APPEND TORCH_SRCS
      ${TORCH_SRC_DIR}/csrc/jit/fuser/cuda/fused_kernel.cpp
    )
    add_library(thnvrtc SHARED ${TORCH_SRC_DIR}/csrc/jit/fuser/cuda/thnvrtc.cpp)
    target_link_libraries(thnvrtc ${CUDA_NVRTC} ${CUDA_CUDA_LIB} ${CUDA_NVRTC_LIB})
    target_include_directories(thnvrtc PRIVATE ${CUDA_INCLUDE_DIRS})
    install(TARGETS thnvrtc DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  endif()
endif ()

if (USE_CUDA)
  list(APPEND TORCH_SRCS
    ${TORCH_SRC_DIR}/csrc/autograd/profiler_cuda.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/functions/comm.cpp
    ${TORCH_SRC_DIR}/csrc/cuda/comm.cpp
  )
endif()


if (USE_ROCM)
  list(APPEND TORCH_SRCS
    ${TORCH_SRC_DIR}/csrc/cuda/comm.cpp
  )
endif()


if (NOT NO_API)
  list(APPEND TORCH_SRCS
    ${TORCH_SRC_DIR}/csrc/api/src/cuda.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/data/datasets/mnist.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/data/samplers/distributed.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/data/samplers/random.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/data/samplers/sequential.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/data/samplers/stream.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/init.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/module.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/batchnorm.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/conv.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/dropout.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/embedding.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/functional.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/linear.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/named_any.cpp
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
  )
endif()


if (TORCH_STATIC)
  add_library(torch STATIC ${TORCH_SRCS})
  target_compile_definitions(torch PUBLIC TORCH_BUILD_STATIC_LIBS)
else()
  add_library(torch SHARED ${TORCH_SRCS})
endif()

target_compile_definitions(torch PUBLIC _THP_CORE)

# until they can be unified, keep these lists synced with setup.py
if(MSVC)
  target_compile_options(torch PUBLIC
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
  target_compile_options(torch PUBLIC
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
elseif (WERROR)
  target_compile_options(torch PRIVATE -Werror -Wno-strict-overflow)
endif()

if (MSVC)
  target_link_libraries(torch onnx onnx_library)
endif()

target_link_libraries(torch caffe2_library)

find_package(OpenMP QUIET)
if(OPENMP_FOUND)
  message(STATUS "pytorch is compiling with OpenMP. \n"
    "OpenMP CXX_FLAGS: ${OpenMP_CXX_FLAGS}. \n"
    "OpenMP libraries: ${OpenMP_CXX_LIBRARIES}.")
  target_compile_options(torch INTERFACE ${OpenMP_CXX_FLAGS})
  target_link_libraries(torch ${OpenMP_CXX_LIBRARIES})
endif()

if (NOT NO_API)
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
    find_library(LIBNVTOOLSEXT libnvToolsExt.so PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)
    set(TORCH_CUDA_LIBRARIES
      ${LIBNVTOOLSEXT}
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

set(ATen_CPU_INCLUDE
  ${TORCH_ROOT}/aten/src
  ${CMAKE_CURRENT_BINARY_DIR}/../aten/src
  ${CMAKE_CURRENT_BINARY_DIR}/../aten/src/ATen
  ${CMAKE_BINARY_DIR}/aten/src)
target_include_directories(torch PUBLIC ${ATen_CPU_INCLUDE})

target_include_directories(torch PUBLIC
  ${TORCH_SRC_DIR}/csrc)

target_include_directories(torch PUBLIC
  ${TORCH_ROOT}/third_party/miniz-2.0.8)

set_target_properties(torch PROPERTIES VERSION 1 SOVERSION 1)

if(NOT ${CMAKE_VERSION} VERSION_LESS "3.1")
  set_property(TARGET torch PROPERTY CXX_STANDARD 11)
endif()

# Prevent the unused functions being optimized away
# Otherwise torch.dll will be linked without caffe2_gpu.dll
if (MSVC)
  set_target_properties(torch PROPERTIES LINK_FLAGS "/OPT:NOREF")
endif(MSVC)

install(DIRECTORY "${TORCH_SRC_DIR}/csrc"
        DESTINATION ${TORCH_INSTALL_INCLUDE_DIR}/torch
        FILES_MATCHING PATTERN "*.h")
install(FILES "${TORCH_SRC_DIR}/script.h" "${TORCH_SRC_DIR}/extension.h"
        DESTINATION ${TORCH_INSTALL_INCLUDE_DIR}/torch)

install(TARGETS torch DESTINATION "${TORCH_INSTALL_LIB_DIR}")

if (MSVC AND NOT TORCH_STATIC)
  install(FILES $<TARGET_PDB_FILE:torch> DESTINATION "${TORCH_INSTALL_LIB_DIR}" OPTIONAL)
endif()

if (BUILD_TEST AND NOT MSVC AND NOT USE_ROCM)
  add_subdirectory(${TORCH_ROOT}/test/cpp/jit ${CMAKE_BINARY_DIR}/test_jit)
endif()

if (BUILD_TEST AND NOT NO_API)
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

if (BUILD_PYTHON)
  if (MSVC)
    add_subdirectory(${TORCH_SRC_DIR}/lib/libshm_windows)
  else()
    add_subdirectory(${TORCH_SRC_DIR}/lib/libshm)
  endif()

  set(TORCH_PYTHON_SRCS
    ${TORCH_SRC_DIR}/csrc/CudaIPCTypes.cpp
    ${TORCH_SRC_DIR}/csrc/DataLoader.cpp
    ${TORCH_SRC_DIR}/csrc/Device.cpp
    ${TORCH_SRC_DIR}/csrc/Dtype.cpp
    ${TORCH_SRC_DIR}/csrc/DynamicTypes.cpp
    ${TORCH_SRC_DIR}/csrc/Exceptions.cpp
    ${TORCH_SRC_DIR}/csrc/TypeInfo.cpp
    ${TORCH_SRC_DIR}/csrc/Generator.cpp
    ${TORCH_SRC_DIR}/csrc/Layout.cpp
    ${TORCH_SRC_DIR}/csrc/Module.cpp
    ${TORCH_SRC_DIR}/csrc/PtrWrapper.cpp
    ${TORCH_SRC_DIR}/csrc/Size.cpp
    ${TORCH_SRC_DIR}/csrc/Storage.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/python/init.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/functions/init.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/generated/python_functions.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/generated/python_nn_functions.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/generated/python_torch_functions.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/generated/python_variable_methods.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/init.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/python_anomaly_mode.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/python_cpp_function.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/python_engine.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/python_function.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/python_hook.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/python_legacy_variable.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/python_variable.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/python_variable_indexing.cpp
    ${TORCH_SRC_DIR}/csrc/byte_order.cpp
    ${TORCH_SRC_DIR}/csrc/jit/init.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/onnx.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/onnx/fixup_onnx_loop.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/onnx/prepare_division_for_onnx.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/onnx/peephole.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/onnx/constant_fold.cpp
    ${TORCH_SRC_DIR}/csrc/jit/python_arg_flatten.cpp
    ${TORCH_SRC_DIR}/csrc/jit/python_interpreter.cpp
    ${TORCH_SRC_DIR}/csrc/jit/python_ir.cpp
    ${TORCH_SRC_DIR}/csrc/jit/python_tracer.cpp
    ${TORCH_SRC_DIR}/csrc/jit/script/init.cpp
    ${TORCH_SRC_DIR}/csrc/jit/script/python_sugared_value.cpp
    ${TORCH_SRC_DIR}/csrc/jit/script/python_tree_views.cpp
    ${TORCH_SRC_DIR}/csrc/multiprocessing/init.cpp
    ${TORCH_SRC_DIR}/csrc/nn/THNN.cpp
    ${TORCH_SRC_DIR}/csrc/onnx/init.cpp
    ${TORCH_SRC_DIR}/csrc/serialization.cpp
    ${TORCH_SRC_DIR}/csrc/tensor/python_tensor.cpp
    ${TORCH_SRC_DIR}/csrc/utils.cpp
    ${TORCH_SRC_DIR}/csrc/utils/cuda_lazy_init.cpp
    ${TORCH_SRC_DIR}/csrc/utils/invalid_arguments.cpp
    ${TORCH_SRC_DIR}/csrc/utils/object_ptr.cpp
    ${TORCH_SRC_DIR}/csrc/utils/python_arg_parser.cpp
    ${TORCH_SRC_DIR}/csrc/utils/structseq.cpp
    ${TORCH_SRC_DIR}/csrc/utils/tensor_apply.cpp
    ${TORCH_SRC_DIR}/csrc/utils/tensor_dtypes.cpp
    ${TORCH_SRC_DIR}/csrc/utils/tensor_layouts.cpp
    ${TORCH_SRC_DIR}/csrc/utils/tensor_list.cpp
    ${TORCH_SRC_DIR}/csrc/utils/tensor_new.cpp
    ${TORCH_SRC_DIR}/csrc/utils/tensor_numpy.cpp
    ${TORCH_SRC_DIR}/csrc/utils/tensor_types.cpp
    ${TORCH_SRC_DIR}/csrc/utils/tuple_parser.cpp
    )

  set(TORCH_PYTHON_INCLUDE_DIRECTORIES
    ${PYTHON_INCLUDE_DIR}

    ${TORCH_ROOT}
    ${TORCH_ROOT}/aten/src
    ${TORCH_ROOT}/aten/src/TH

    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/aten/src
    ${CMAKE_BINARY_DIR}/caffe2/aten/src
    ${CMAKE_BINARY_DIR}/third_party
    ${CMAKE_BINARY_DIR}/third_party/onnx

    ${TORCH_ROOT}/third_party/build/nccl/include
    ${TORCH_ROOT}/third_party/gloo
    ${TORCH_ROOT}/third_party/onnx
    ${TORCH_ROOT}/third_party/pybind11/include

    ${TORCH_SRC_DIR}/csrc
    ${TORCH_SRC_DIR}/csrc/api/include
    ${TORCH_SRC_DIR}/lib
    )

  if (MSVC)
    list(APPEND TORCH_PYTHON_INCLUDE_DIRECTORIES
      ${TORCH_SRC_DIR}/lib/libshm_windows)
  else()
    list(APPEND TORCH_PYTHON_INCLUDE_DIRECTORIES
      ${TORCH_SRC_DIR}/lib/libshm)
  endif()

  set(TORCH_PYTHON_LINK_LIBRARIES
    torch
    shm)

  set(TORCH_PYTHON_COMPILE_DEFINITIONS)

  set(TORCH_PYTHON_COMPILE_OPTIONS)

  set(TORCH_PYTHON_LINK_FLAGS "")

  if (MSVC)
    string(APPEND TORCH_PYTHON_LINK_FLAGS " /NODEFAULTLIB:LIBCMT.LIB")
    list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${PYTHON_LIBRARIES})
    if (NOT ${CMAKE_BUILD_TYPE} MATCHES "Release")
      string(APPEND TORCH_PYTHON_LINK_FLAGS " /DEBUG:FULL")
    endif()
  elseif (APPLE)
    string(APPEND TORCH_PYTHON_LINK_FLAGS " -undefined dynamic_lookup")
  else()
    list(APPEND TORCH_PYTHON_COMPILE_OPTIONS
      -fno-strict-aliasing
      -Wno-write-strings
      -Wno-strict-aliasing)
  endif()

  if (USE_CUDA)
    list(APPEND TORCH_PYTHON_SRCS
      ${TORCH_SRC_DIR}/csrc/cuda/Module.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/Storage.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/Stream.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/Event.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/utils.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/python_comm.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/serialization.cpp
      ${TORCH_SRC_DIR}/csrc/nn/THCUNN.cpp
      )
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_CUDA)

    if(MSVC)
      list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${NVTOOLEXT_HOME}/lib/x64/nvToolsExt64_1.lib)
      list(APPEND TORCH_PYTHON_INCLUDE_DIRECTORIES "${NVTOOLEXT_HOME}/include")
    elseif(APPLE)
      list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib/libnvToolsExt.dylib)
    else()
      find_library(LIBNVTOOLSEXT libnvToolsExt.so PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)
      list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${LIBNVTOOLSEXT})
    endif()
  endif()

  if (USE_CUDNN)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_CUDNN)

    # NOTE: these are at the front, in case there's another cuDNN in
    # CUDA path.
    # Basically, this is the case where $CUDA_HOME/lib64 has an old or
    # incompatible libcudnn.so, which we can inadvertently link to if
    # we're not careful.
    list(INSERT 0 TORCH_PYTHON_LINK_LIBRARIES ${CUDNN_LIBRARY})
    list(INSERT 0 TORCH_PYTHON_INCLUDE_DIRECTORIES ${CUDNN_INCLUDE_DIR})
  endif()

  if (USE_MIOPEN)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_MIOPEN)
    list(APPEND TORCH_PYTHON_INCLUDE_DIRECTORIES ${MIOPEN_INCLUDE_DIR})
    list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${MIOPEN_LIBRARY})
  endif()

  if (USE_NUMPY)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_NUMPY)
  endif()

  if (USE_ROCM)
    list(APPEND TORCH_PYTHON_SRCS
      ${TORCH_SRC_DIR}/csrc/cuda/Module.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/Storage.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/Stream.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/Event.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/utils.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/python_comm.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/serialization.cpp
      ${TORCH_SRC_DIR}/csrc/nn/THCUNN.cpp
      )
    list(APPEND TORCH_PYTHON_LINK_LIBRARIES caffe2_hip_library)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS
      USE_ROCM
      __HIP_PLATFORM_HCC__
      )
  endif()

  if (USE_DISTRIBUTED)
    list(APPEND TORCH_PYTHON_SRCS ${TORCH_SRC_DIR}/csrc/distributed/Module.cpp)
    list(APPEND TORCH_PYTHON_INCLUDE_DIRECTORIES ${TORCH_SRC_DIR}/lib/THD)
    list(APPEND TORCH_PYTHON_LINK_LIBRARIES THD)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_DISTRIBUTED)
    if (NOT MSVC AND NOT APPLE)
      list(APPEND TORCH_PYTHON_SRCS ${TORCH_SRC_DIR}/csrc/distributed/c10d/init.cpp)
      list(APPEND TORCH_PYTHON_SRCS ${TORCH_SRC_DIR}/csrc/distributed/c10d/reducer.cpp)
      list(APPEND TORCH_PYTHON_LINK_LIBRARIES c10d)
      list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_C10D)
      if (USE_CUDA)
        list(APPEND TORCH_PYTHON_SRCS ${TORCH_SRC_DIR}/csrc/distributed/c10d/ddp.cpp)
      endif()
    endif()
  endif()

  if (USE_NCCL)
    list(APPEND TORCH_PYTHON_SRCS
      ${TORCH_SRC_DIR}/csrc/cuda/nccl.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/python_nccl.cpp)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_NCCL)
    list(APPEND TORCH_PYTHON_LINK_LIBRARIES __caffe2_nccl)
    if (USE_SYSTEM_NCCL)
    endif()
  endif()

  add_custom_target(torch_python_stubs DEPENDS "${TORCH_SRC_DIR}/__init__.pyi")
  # For Declarations.yaml dependency
  add_dependencies(torch_python_stubs ATEN_CPU_FILES_GEN_TARGET)
  add_custom_command(
    OUTPUT
    "${TORCH_SRC_DIR}/__init__.pyi"
    COMMAND
    "${PYTHON_EXECUTABLE}" -mtools.pyi.gen_pyi
      --declarations-path "${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml"
    DEPENDS
    "${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml"
    "${TORCH_SRC_DIR}/__init__.pyi.in"
    WORKING_DIRECTORY
    "${TORCH_ROOT}"
    )

  add_library(torch_python SHARED ${TORCH_PYTHON_SRCS})

  add_dependencies(torch_python torch_python_stubs)

  target_link_libraries(torch_python ${TORCH_PYTHON_LINK_LIBRARIES})

  target_compile_definitions(torch_python PRIVATE ${TORCH_PYTHON_COMPILE_DEFINITIONS})

  target_compile_options(torch_python PRIVATE ${TORCH_PYTHON_COMPILE_OPTIONS})

  target_include_directories(torch_python PUBLIC ${TORCH_PYTHON_INCLUDE_DIRECTORIES})

  if (NOT TORCH_PYTHON_LINK_FLAGS STREQUAL "")
    set_target_properties(torch_python PROPERTIES LINK_FLAGS ${TORCH_PYTHON_LINK_FLAGS})
  endif()

  install(TARGETS torch_python DESTINATION "${TORCH_INSTALL_LIB_DIR}")
endif()
