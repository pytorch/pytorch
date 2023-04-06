if(NOT BUILD_NVFUSER)
  return()
endif()

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(nvfuser)

if(NOT USE_ROCM)
  set(TORCHLIB_FLAVOR torch_cuda)
else()
  set(TORCHLIB_FLAVOR torch_hip)
endif()

# --- project

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/nvfuser")

set(NVFUSER_ROOT ${PROJECT_SOURCE_DIR})
set(NVFUSER_SRCS_DIR "${NVFUSER_ROOT}/csrc")
set(TORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(TORCH_INSTALL_LIB_DIR ${TORCH_ROOT}/torch/lib)

# --- build nvfuser_codegen library

set(NVFUSER_SRCS)
set(NVFUSER_CODEGEN ${PROJECT_NAME}_codegen)
list(APPEND NVFUSER_SRCS
    ${NVFUSER_SRCS_DIR}/arith.cpp
    ${NVFUSER_SRCS_DIR}/compute_at.cpp
    ${NVFUSER_SRCS_DIR}/inlining.cpp
    ${NVFUSER_SRCS_DIR}/compute_at_map.cpp
    ${NVFUSER_SRCS_DIR}/codegen.cpp
    ${NVFUSER_SRCS_DIR}/contiguity.cpp
    ${NVFUSER_SRCS_DIR}/dispatch.cpp
    ${NVFUSER_SRCS_DIR}/expr_evaluator.cpp
    ${NVFUSER_SRCS_DIR}/kernel_expr_evaluator.cpp
    ${NVFUSER_SRCS_DIR}/executor.cpp
    ${NVFUSER_SRCS_DIR}/executor_kernel_arg.cpp
    ${NVFUSER_SRCS_DIR}/executor_launch_params.cpp
    ${NVFUSER_SRCS_DIR}/evaluator_common.cpp
    ${NVFUSER_SRCS_DIR}/executor_utils.cpp
    ${NVFUSER_SRCS_DIR}/fusion.cpp
    ${NVFUSER_SRCS_DIR}/graph_fuser.cpp
    ${NVFUSER_SRCS_DIR}/grouped_reduction.cpp
    ${NVFUSER_SRCS_DIR}/index_compute.cpp
    ${NVFUSER_SRCS_DIR}/lower_index_compute.cpp
    ${NVFUSER_SRCS_DIR}/instrumentation.cpp
    ${NVFUSER_SRCS_DIR}/ir_base_nodes.cpp
    ${NVFUSER_SRCS_DIR}/ir_builder.cpp
    ${NVFUSER_SRCS_DIR}/ir_cloner.cpp
    ${NVFUSER_SRCS_DIR}/ir_container.cpp
    ${NVFUSER_SRCS_DIR}/ir_graphviz.cpp
    ${NVFUSER_SRCS_DIR}/ir_nodes.cpp
    ${NVFUSER_SRCS_DIR}/ir_iostream.cpp
    ${NVFUSER_SRCS_DIR}/ir_utils.cpp
    ${NVFUSER_SRCS_DIR}/iter_visitor.cpp
    ${NVFUSER_SRCS_DIR}/kernel.cpp
    ${NVFUSER_SRCS_DIR}/kernel_cache.cpp
    ${NVFUSER_SRCS_DIR}/kernel_ir.cpp
    ${NVFUSER_SRCS_DIR}/kernel_ir_dispatch.cpp
    ${NVFUSER_SRCS_DIR}/lower_alias_memory.cpp
    ${NVFUSER_SRCS_DIR}/lower_allocation.cpp
    ${NVFUSER_SRCS_DIR}/lower_double_buffer.cpp
    ${NVFUSER_SRCS_DIR}/lower_divisible_split.cpp
    ${NVFUSER_SRCS_DIR}/lower_expr_sort.cpp
    ${NVFUSER_SRCS_DIR}/lower_fused_reduction.cpp
    ${NVFUSER_SRCS_DIR}/lower_fusion_simplifier.cpp
    ${NVFUSER_SRCS_DIR}/lower_index.cpp
    ${NVFUSER_SRCS_DIR}/lower_index_hoist.cpp
    ${NVFUSER_SRCS_DIR}/lower_insert_syncs.cpp
    ${NVFUSER_SRCS_DIR}/lower_instrument.cpp
    ${NVFUSER_SRCS_DIR}/lower_loops.cpp
    ${NVFUSER_SRCS_DIR}/lower_magic_zero.cpp
    ${NVFUSER_SRCS_DIR}/lower_misaligned_vectorization.cpp
    ${NVFUSER_SRCS_DIR}/lower_predicate.cpp
    ${NVFUSER_SRCS_DIR}/lower_predicate_elimination.cpp
    ${NVFUSER_SRCS_DIR}/lower_replace_size.cpp
    ${NVFUSER_SRCS_DIR}/lower_shift.cpp
    ${NVFUSER_SRCS_DIR}/lower_sync_information.cpp
    ${NVFUSER_SRCS_DIR}/lower_thread_predicate.cpp
    ${NVFUSER_SRCS_DIR}/lower_trivial_broadcast.cpp
    ${NVFUSER_SRCS_DIR}/lower_trivial_reductions.cpp
    ${NVFUSER_SRCS_DIR}/lower_unroll.cpp
    ${NVFUSER_SRCS_DIR}/lower_utils.cpp
    ${NVFUSER_SRCS_DIR}/lower_validation.cpp
    ${NVFUSER_SRCS_DIR}/lower_warp_reduce.cpp
    ${NVFUSER_SRCS_DIR}/lower2device.cpp
    ${NVFUSER_SRCS_DIR}/lower_bank_conflict.cpp
    ${NVFUSER_SRCS_DIR}/manager.cpp
    ${NVFUSER_SRCS_DIR}/maxinfo_propagator.cpp
    ${NVFUSER_SRCS_DIR}/mutator.cpp
    ${NVFUSER_SRCS_DIR}/non_divisible_split.cpp
    ${NVFUSER_SRCS_DIR}/ops/alias.cpp
    ${NVFUSER_SRCS_DIR}/ops/composite.cpp
    ${NVFUSER_SRCS_DIR}/ops/normalization.cpp
    ${NVFUSER_SRCS_DIR}/parallel_dimension_map.cpp
    ${NVFUSER_SRCS_DIR}/parallel_type_bitmap.cpp
    ${NVFUSER_SRCS_DIR}/parser.cpp
    ${NVFUSER_SRCS_DIR}/partial_split_map.cpp
    ${NVFUSER_SRCS_DIR}/partition.cpp
    ${NVFUSER_SRCS_DIR}/predicate_compute.cpp
    ${NVFUSER_SRCS_DIR}/python_frontend/fusion_cache.cpp
    ${NVFUSER_SRCS_DIR}/python_frontend/fusion_definition.cpp
    ${NVFUSER_SRCS_DIR}/python_frontend/fusion_interface.cpp
    ${NVFUSER_SRCS_DIR}/register_interface.cpp
    ${NVFUSER_SRCS_DIR}/root_domain_map.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/pointwise.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/pointwise_utils.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/transpose.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/normalization.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/reduction.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/matmul.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/reduction_utils.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/registry.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/utils.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/vectorize_helper.cpp
    ${NVFUSER_SRCS_DIR}/type_inference.cpp
    ${NVFUSER_SRCS_DIR}/type_promotion.cpp
    ${NVFUSER_SRCS_DIR}/fusion_segmenter.cpp
    ${NVFUSER_SRCS_DIR}/tensor_view.cpp
    ${NVFUSER_SRCS_DIR}/transform_iter.cpp
    ${NVFUSER_SRCS_DIR}/transform_replay.cpp
    ${NVFUSER_SRCS_DIR}/transform_rfactor.cpp
    ${NVFUSER_SRCS_DIR}/transform_view.cpp
    ${NVFUSER_SRCS_DIR}/type.cpp
    ${NVFUSER_SRCS_DIR}/utils.cpp
    ${NVFUSER_SRCS_DIR}/mma_type.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/mma_utils.cpp
)

add_library(${NVFUSER_CODEGEN} SHARED ${NVFUSER_SRCS})

if(NOT USE_ROCM)
  target_compile_options(${NVFUSER_CODEGEN} PRIVATE "-DTORCH_CUDA_BUILD_MAIN_LIB")
  # NB: This must be target_compile_definitions, not target_compile_options,
  # as the latter is not respected by nvcc
  target_compile_definitions(${NVFUSER_CODEGEN} PRIVATE "-DTORCH_CUDA_BUILD_MAIN_LIB")
else()
  target_compile_options(${NVFUSER_CODEGEN} PRIVATE "-DTORCH_HIP_BUILD_MAIN_LIB")
  target_compile_definitions(${NVFUSER_CODEGEN} PRIVATE "-DTORCH_HIP_BUILD_MAIN_LIB")
  target_compile_definitions(${NVFUSER_CODEGEN} PRIVATE
    USE_ROCM
    __HIP_PLATFORM_HCC__
    )
endif()

target_link_libraries(${NVFUSER_CODEGEN} PRIVATE torch ${TORCHLIB_FLAVOR})
if(NOT USE_ROCM)
  target_link_libraries(${NVFUSER_CODEGEN} PRIVATE ${CUDA_NVRTC_LIB} torch::nvtoolsext)
  target_include_directories(${NVFUSER_CODEGEN} PRIVATE ${CUDA_INCLUDE_DIRS})
else()
  target_link_libraries(${NVFUSER_CODEGEN} PRIVATE ${ROCM_HIPRTC_LIB})
  target_include_directories(${NVFUSER_CODEGEN} PRIVATE ${Caffe2_HIP_INCLUDE})
endif()
if(NOT MSVC)
  target_compile_options(${NVFUSER_CODEGEN} PRIVATE -Wno-unused-variable)
endif()
target_include_directories(${NVFUSER_CODEGEN}
                           PUBLIC $<BUILD_INTERFACE:${NVFUSER_SRCS_DIR}>)
set_property(TARGET ${NVFUSER_CODEGEN} PROPERTY CXX_STANDARD 17)
install(TARGETS ${NVFUSER_CODEGEN} EXPORT NvfuserTargets DESTINATION "${TORCH_INSTALL_LIB_DIR}")
# installing nvfuser python tests
install(DIRECTORY "${NVFUSER_ROOT}/python_tests/"
        DESTINATION "${TORCH_ROOT}/test/_nvfuser"
        FILES_MATCHING PATTERN "*.py" )

file(WRITE "${TORCH_ROOT}/test/_nvfuser/.gitignore" "*")
# --- build nvfuser_python library

if(BUILD_PYTHON)
  set(NVFUSER "${PROJECT_NAME}")
  #find_package(pybind11 REQUIRED)

  set(NVFUSER_PYTHON_SRCS)
  list(APPEND NVFUSER_PYTHON_SRCS
      ${NVFUSER_SRCS_DIR}/python_frontend/python_bindings.cpp
      ${NVFUSER_SRCS_DIR}/python_frontend/python_bindings_extension.cpp
  )

  add_library(${NVFUSER} MODULE ${NVFUSER_PYTHON_SRCS})
  if(NOT USE_ROCM)
    target_compile_options(${NVFUSER} PRIVATE "-DTORCH_CUDA_BUILD_MAIN_LIB")
    # NB: This must be target_compile_definitions, not target_compile_options,
    # as the latter is not respected by nvcc
    target_compile_definitions(${NVFUSER} PRIVATE "-DTORCH_CUDA_BUILD_MAIN_LIB")
    target_link_libraries(${NVFUSER} PRIVATE torch::nvtoolsext)
  else()
    target_compile_options(${NVFUSER} PRIVATE "-DTORCH_HIP_BUILD_MAIN_LIB")
    target_compile_definitions(${NVFUSER} PRIVATE "-DTORCH_HIP_BUILD_MAIN_LIB")
    target_compile_definitions(${NVFUSER} PRIVATE
      USE_ROCM
      __HIP_PLATFORM_HCC__
      )
    target_include_directories(${NVFUSER_CODEGEN} PRIVATE ${Caffe2_HIP_INCLUDE})
  endif()

  target_link_libraries(${NVFUSER} PRIVATE ${NVFUSER_CODEGEN})
  target_link_libraries(${NVFUSER} PRIVATE torch torch_python ${TORCHLIB_FLAVOR})
  target_link_libraries(${NVFUSER} PRIVATE pybind::pybind11)
  target_include_directories(${NVFUSER} PRIVATE ${TORCH_ROOT})
  target_compile_definitions(${NVFUSER} PRIVATE EXTENSION_NAME=_C)
  target_compile_options(${NVFUSER} PRIVATE ${TORCH_PYTHON_COMPILE_OPTIONS})

  # avoid using Python3_add_library, copied from functorch
  set_target_properties(${NVFUSER} PROPERTIES PREFIX "" DEBUG_POSTFIX "")
  if(NOT MSVC)
    target_compile_options(${NVFUSER} PRIVATE -Wno-unused-variable)
    set_target_properties(${NVFUSER} PROPERTIES SUFFIX ".so")
  else()
    set_target_properties(${NVFUSER} PROPERTIES SUFFIX ".pyd")
  endif()

  set_target_properties(${NVFUSER} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_BINARY_DIR}/nvfuser)
  set_target_properties(${NVFUSER} PROPERTIES INSTALL_RPATH "${_rpath_portable_origin}/../torch/lib")

  if(TORCH_PYTHON_LINK_FLAGS AND NOT TORCH_PYTHON_LINK_FLAGS STREQUAL "")
    message(STATUS "somehow this is happening")
    set_target_properties(${NVFUSER} PROPERTIES LINK_FLAGS ${TORCH_PYTHON_LINK_FLAGS})
  endif()
  install(TARGETS ${NVFUSER} EXPORT NvfuserTargets DESTINATION ${TORCH_ROOT}/nvfuser/)

  # install nvfuser python files
  install(DIRECTORY "${NVFUSER_ROOT}/python/"
          DESTINATION "${TORCH_ROOT}/nvfuser"
          FILES_MATCHING PATTERN "*.py" )
  
  file(WRITE "${TORCH_ROOT}/nvfuser/.gitignore" "*")
endif()

# --- generate runtime files

# The list of NVFUSER runtime files
list(APPEND NVFUSER_RUNTIME_FILES
  ${NVFUSER_ROOT}/runtime/array.cu
  ${NVFUSER_ROOT}/runtime/block_reduction.cu
  ${NVFUSER_ROOT}/runtime/block_sync_atomic.cu
  ${NVFUSER_ROOT}/runtime/block_sync_default.cu
  ${NVFUSER_ROOT}/runtime/broadcast.cu
  ${NVFUSER_ROOT}/runtime/fp16_support.cu
  ${NVFUSER_ROOT}/runtime/fused_reduction.cu
  ${NVFUSER_ROOT}/runtime/fused_welford_helper.cu
  ${NVFUSER_ROOT}/runtime/fused_welford_impl.cu
  ${NVFUSER_ROOT}/runtime/bf16_support.cu
  ${NVFUSER_ROOT}/runtime/grid_broadcast.cu
  ${NVFUSER_ROOT}/runtime/grid_reduction.cu
  ${NVFUSER_ROOT}/runtime/grid_sync.cu
  ${NVFUSER_ROOT}/runtime/helpers.cu
  ${NVFUSER_ROOT}/runtime/index_utils.cu
  ${NVFUSER_ROOT}/runtime/random_numbers.cu
  ${NVFUSER_ROOT}/runtime/swizzle.cu
  ${NVFUSER_ROOT}/runtime/tensor.cu
  ${NVFUSER_ROOT}/runtime/tuple.cu
  ${NVFUSER_ROOT}/runtime/type_traits.cu
  ${NVFUSER_ROOT}/runtime/welford.cu
  ${NVFUSER_ROOT}/runtime/warp.cu
  ${NVFUSER_ROOT}/runtime/tensorcore.cu
  ${NVFUSER_ROOT}/runtime/memory.cu
  ${TORCH_ROOT}/aten/src/ATen/cuda/detail/PhiloxCudaStateRaw.cuh
  ${TORCH_ROOT}/aten/src/ATen/cuda/detail/UnpackRaw.cuh
)

if(USE_ROCM)
list(APPEND NVFUSER_RUNTIME_FILES
  ${NVFUSER_ROOT}/runtime/array_rocm.cu
  ${NVFUSER_ROOT}/runtime/bf16_support_rocm.cu
  ${NVFUSER_ROOT}/runtime/block_sync_default_rocm.cu
  ${NVFUSER_ROOT}/runtime/warp_rocm.cu
)
endif()

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include/nvfuser_resources")

# "stringify" NVFUSER runtime sources
# (generate C++ header files embedding the original input as a string literal)
set(NVFUSER_STRINGIFY_TOOL "${NVFUSER_ROOT}/tools/stringify_file.py")
foreach(src ${NVFUSER_RUNTIME_FILES})
  get_filename_component(filename ${src} NAME_WE)
  set(dst "${CMAKE_BINARY_DIR}/include/nvfuser_resources/${filename}.h")
  add_custom_command(
    COMMENT "Stringify NVFUSER runtime source file"
    OUTPUT ${dst}
    DEPENDS ${src} "${NVFUSER_STRINGIFY_TOOL}"
    COMMAND ${PYTHON_EXECUTABLE} ${NVFUSER_STRINGIFY_TOOL} -i ${src} -o ${dst}
  )
  add_custom_target(nvfuser_rt_${filename} DEPENDS ${dst})
  add_dependencies(${NVFUSER_CODEGEN} nvfuser_rt_${filename})

  # also generate the resource headers during the configuration step
  # (so tools like clang-tidy can run w/o requiring a real build)
  execute_process(COMMAND
    ${PYTHON_EXECUTABLE} ${NVFUSER_STRINGIFY_TOOL} -i ${src} -o ${dst})
endforeach()

target_include_directories(${NVFUSER_CODEGEN} PRIVATE "${CMAKE_BINARY_DIR}/include")

# -- build tests

# note: ideally we don't need USE_CUDA here, but our cpp tests are not ROCM compatible.
if(BUILD_TEST AND USE_CUDA)
  set(NVFUSER_TESTS "${PROJECT_NAME}_tests")
  set(JIT_TEST_SRCS)
  list(APPEND JIT_TEST_SRCS ${NVFUSER_SRCS_DIR}/python_frontend/test/test_nvfuser_fusion_definition.cpp)
  list(APPEND JIT_TEST_SRCS ${NVFUSER_SRCS_DIR}/python_frontend/test/test_nvfuser_fusion_cache.cpp)
  list(APPEND JIT_TEST_SRCS ${NVFUSER_SRCS_DIR}/python_frontend/test/test_nvfuser_fusion_record.cpp)
  list(APPEND JIT_TEST_SRCS ${NVFUSER_ROOT}/test/test_gpu1.cpp)
  list(APPEND JIT_TEST_SRCS ${NVFUSER_ROOT}/test/test_gpu2.cpp)
  list(APPEND JIT_TEST_SRCS ${NVFUSER_ROOT}/test/test_gpu3.cpp)
  list(APPEND JIT_TEST_SRCS ${NVFUSER_ROOT}/test/test_gpu_tensor_factories.cpp)
  list(APPEND JIT_TEST_SRCS ${NVFUSER_ROOT}/test/test_gpu_fused_reduction.cpp)
  list(APPEND JIT_TEST_SRCS ${NVFUSER_ROOT}/test/test_gpu_shift.cpp)
  list(APPEND JIT_TEST_SRCS ${NVFUSER_ROOT}/test/test_gpu_tensorcore.cpp)
  list(APPEND JIT_TEST_SRCS ${NVFUSER_ROOT}/test/test_gpu_view.cpp)
  list(APPEND JIT_TEST_SRCS ${NVFUSER_ROOT}/test/test_gpu_transpose.cpp)
  list(APPEND JIT_TEST_SRCS ${NVFUSER_ROOT}/test/test_gpu_rng.cu)
  list(APPEND JIT_TEST_SRCS ${NVFUSER_ROOT}/test/test_gpu_utils.cpp)

  add_executable(${NVFUSER_TESTS}
             ${TORCH_ROOT}/test/cpp/common/main.cpp
             ${TORCH_ROOT}/test/cpp/jit/test_utils.cpp
             ${JIT_TEST_SRCS})

  target_compile_definitions(${NVFUSER_TESTS} PRIVATE USE_GTEST)
  if(NOT USE_ROCM)
    target_compile_definitions(${NVFUSER_TESTS} PRIVATE USE_CUDA)
  else()
    target_compile_definitions(${NVFUSER_TESTS} PRIVATE USE_ROCM)
  endif()
  target_include_directories(${NVFUSER_TESTS} PRIVATE "${NVFUSER_ROOT}" "${TORCH_ROOT}/torch/csrc/api/include/")
  target_link_libraries(${NVFUSER_TESTS} PRIVATE ${NVFUSER_CODEGEN} torch ${TORCHLIB_FLAVOR} gtest_main gmock_main)
  if(NOT MSVC)
    target_compile_options(${NVFUSER_TESTS} PRIVATE -Wno-unused-variable)
  endif()

  install(TARGETS ${NVFUSER_TESTS} DESTINATION bin)
endif()

if(BUILD_NVFUSER_BENCHMARK)
  set(NVFUSER_BENCHMARK "${PROJECT_NAME}_bench")
  set(BENCHMARK_SRCS)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/batch_norm_channels_first.cpp)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/batch_norm_channels_first_backward.cpp)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/batch_norm_channels_last.cpp)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/batch_norm_channels_last_backward.cpp)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/bert.cpp)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/broadcast.cpp)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/gelu_backward.cpp)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/heuristic_lookup.cpp)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/shape_inference.cpp)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/instance_norm.cpp)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/layer_norm.cpp)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/layer_norm_backward.cpp)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/rms_norm.cpp)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/rms_norm_backward.cpp)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/lstm_cell.cpp)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/reduction.cpp)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/softmax.cpp)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/softmax_backward.cpp)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/scale_bias_relu.cpp)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/transpose.cpp)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/matmul.cpp)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/timm.cpp)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/utils.cpp)
  list(APPEND BENCHMARK_SRCS ${NVFUSER_ROOT}/benchmark/main.cpp)

  add_executable(${NVFUSER_BENCHMARK} ${BENCHMARK_SRCS})
  target_link_libraries(${NVFUSER_BENCHMARK} PRIVATE torch_library benchmark ${NVFUSER_CODEGEN})
  target_include_directories(${NVFUSER_BENCHMARK} PRIVATE ${NVFUSER_ROOT})
  if(NOT MSVC)
    target_compile_options_if_supported(nvfuser_bench -Werror)
    target_compile_options_if_supported(nvfuser_bench -Wno-unused-variable)
    target_compile_options_if_supported(nvfuser_bench -Wno-deprecated-copy)
  endif()

endif()
