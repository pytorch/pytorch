function(filter_out_gencode_arch output_var arch_to_remove)
  separate_arguments(_cuda_flags UNIX_COMMAND "${ARGN}")
  set(_filtered_cuda_flags)
  list(LENGTH _cuda_flags _cuda_flags_length)
  set(_index 0)

  while(_index LESS _cuda_flags_length)
    list(GET _cuda_flags ${_index} _flag)
    math(EXPR _value_index "${_index} + 1")

    if(_flag STREQUAL "-gencode" AND _value_index LESS _cuda_flags_length)
      list(GET _cuda_flags ${_value_index} _value)
      if(_value STREQUAL "arch=compute_${arch_to_remove},code=sm_${arch_to_remove}" OR _value STREQUAL "arch=compute_${arch_to_remove},code=compute_${arch_to_remove}" OR _value STREQUAL "arch=compute_${arch_to_remove},code=[compute_${arch_to_remove},sm_${arch_to_remove}]" OR _value STREQUAL "arch=compute_${arch_to_remove},code=[sm_${arch_to_remove},compute_${arch_to_remove}]")
        math(EXPR _index "${_index} + 2")
        continue()
      endif()
    endif()

    list(APPEND _filtered_cuda_flags "${_flag}")
    math(EXPR _index "${_index} + 1")
  endwhile()

  list(JOIN _filtered_cuda_flags " " _filtered_cuda_flags_string)
  set(${output_var} "${_filtered_cuda_flags_string}" PARENT_SCOPE)
endfunction()

file(GLOB flash_attention_cuda_kernels_cu ${PROJECT_SOURCE_DIR}/third_party/flash-attention/csrc/flash_attn/src/*.cu)
file(GLOB flash_attention_cuda_cpp ${PROJECT_SOURCE_DIR}/third_party/flash-attention/csrc/flash_attn/src/*.cpp)

filter_out_gencode_arch(CMAKE_CUDA_FLAGS 75 "${CMAKE_CUDA_FLAGS}")

add_library(flash_attention OBJECT EXCLUDE_FROM_ALL ${flash_attention_cuda_kernels_cu} ${flash_attention_cuda_cpp})

target_include_directories(flash_attention SYSTEM PUBLIC
  ${PROJECT_SOURCE_DIR}/third_party/flash-attention/csrc
  ${PROJECT_SOURCE_DIR}/third_party/flash-attention/include
  ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
  ${PROJECT_SOURCE_DIR}/third_party/flash-attention/csrc/flash_attn/src
)

target_compile_definitions(flash_attention PRIVATE
  # Copied from https://github.com/pytorch/pytorch/blob/a10024d7dea47c52469059a47efe376eb20adca0/caffe2/CMakeLists.txt#L1431
  FLASH_NAMESPACE=pytorch_flash
  FLASHATTENTION_DISABLE_ALIBI
  FLASHATTENTION_DISABLE_SOFTCAP
  UNFUSE_FMA
)

set_target_properties(flash_attention PROPERTIES POSITION_INDEPENDENT_CODE ON)
