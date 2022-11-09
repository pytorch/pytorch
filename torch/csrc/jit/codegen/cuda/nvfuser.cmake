if(USE_CUDA)
  set(TORCHLIB_FLAVOR torch_cuda)
elseif(USE_ROCM)
  set(TORCHLIB_FLAVOR torch_hip)
endif()

# The list of NVFUSER runtime files
list(APPEND NVFUSER_RUNTIME_FILES
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/array.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/block_reduction.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/block_sync_atomic.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/block_sync_default.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/broadcast.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/fp16_support.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/fused_reduction.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/fused_welford_helper.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/fused_welford_impl.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/bf16_support.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/grid_broadcast.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/grid_reduction.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/grid_sync.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/helpers.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/index_utils.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/random_numbers.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/tensor.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/tuple.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/type_traits.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/welford.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/warp.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/tensorcore.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/memory.cu
  ${TORCH_ROOT}/aten/src/ATen/cuda/detail/PhiloxCudaStateRaw.cuh
  ${TORCH_ROOT}/aten/src/ATen/cuda/detail/UnpackRaw.cuh
)

if(USE_ROCM)
list(APPEND NVFUSER_RUNTIME_FILES
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/array_rocm.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/bf16_support_rocm.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/block_sync_default_rocm.cu
  ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/warp_rocm.cu
)
endif()

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include/nvfuser_resources")

# "stringify" NVFUSER runtime sources
# (generate C++ header files embedding the original input as a string literal)
set(NVFUSER_STRINGIFY_TOOL "${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/tools/stringify_file.py")
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
  add_dependencies(${TORCHLIB_FLAVOR} nvfuser_rt_${filename})

  # also generate the resource headers during the configuration step
  # (so tools like clang-tidy can run w/o requiring a real build)
  execute_process(COMMAND
    ${PYTHON_EXECUTABLE} ${NVFUSER_STRINGIFY_TOOL} -i ${src} -o ${dst})
endforeach()

target_include_directories(${TORCHLIB_FLAVOR} PRIVATE "${CMAKE_BINARY_DIR}/include")
