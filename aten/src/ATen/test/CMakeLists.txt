if(MSVC)
  if(MSVC_VERSION LESS 1911)
    return()
  endif()
endif(MSVC)

list(APPEND ATen_CPU_TEST_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/Dict_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Dimname_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MaybeOwned_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/NamedTensor_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/apply_utils_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/atest.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basic.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/broadcast_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu_allocator_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu_generator_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu_profiling_allocator_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu_rng_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dispatch_key_set_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dlconvertor_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/extension_backend_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/half_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ivalue_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lazy_tensor_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math_kernel_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/memory_format_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/memory_overlapping_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mobile_memory_cleanup.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/native_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operator_name_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operators_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/packedtensoraccessor_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/pow_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/quantized_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reduce_ops_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reportMemoryUsage_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scalar_tensor_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scalar_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/StorageUtils_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/stride_properties_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tensor_iterator_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_parallel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/thread_init_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/type_ptr_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/type_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/undefined_tensor_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/verify_api_visibility.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/legacy_vmap_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/weakref_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/wrapdim_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/xla_tensor_test.cpp
  # Fix this.
  # ${CMAKE_CURRENT_SOURCE_DIR}/xnnpack_test.cpp
  )

list(APPEND ATen_CUDA_TEST_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_allocator_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_apply_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_atomic_ops_test.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_caching_host_allocator_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_complex_math_test.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_complex_test.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_cub_test.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_device_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_distributions_test.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_dlconvertor_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_generator_test.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_half_test.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_integer_divider_test.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_optional_test.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_packedtensoraccessor_test.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_reportMemoryUsage_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_allocatorTraceTracker_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_stream_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_vectorized_test.cu)
if(CAFFE2_USE_CUDNN)
  list(APPEND ATen_CUDA_TEST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_cudnn_test.cpp)
endif()

list(APPEND ATen_HIP_TEST_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_apply_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_complex_math_test.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_complex_test.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_distributions_test.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_dlconvertor_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_generator_test.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_half_test.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_integer_divider_test.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_optional_test.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_packedtensoraccessor_test.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_vectorized_test.hip)
# TODO: fix and enable these
#  ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_tensor_interop_test.cpp
#  ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_stream_test.cpp

list(APPEND ATen_VULKAN_TEST_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/vulkan_api_test.cpp)

list(APPEND ATen_MOBILE_TEST_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu_caching_allocator_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu_profiling_allocator_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/quantized_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/vec_test_all_types.cpp)

list(APPEND ATen_VEC_TEST_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/vec_test_all_types.cpp
  )

list(APPEND ATen_MPS_TEST_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/mps_test_print.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mps_test_allocator.cpp)
if(APPLE AND USE_MPS)
  list(APPEND ATen_MPS_TEST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/mps_test_objc_interface.mm)
endif()

list(APPEND ATen_XPU_TEST_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/xpu_caching_host_allocator_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/xpu_device_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/xpu_event_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/xpu_generator_test.cpp
  )

# ---[ Send the lists to the parent scope.
set(ATen_CPU_TEST_SRCS ${ATen_CPU_TEST_SRCS} PARENT_SCOPE)
set(ATen_CUDA_TEST_SRCS ${ATen_CUDA_TEST_SRCS} PARENT_SCOPE)
set(ATen_HIP_TEST_SRCS ${ATen_HIP_TEST_SRCS} PARENT_SCOPE)
set(ATen_VULKAN_TEST_SRCS ${ATen_VULKAN_TEST_SRCS} PARENT_SCOPE)
set(ATen_MOBILE_TEST_SRCS ${ATen_MOBILE_TEST_SRCS} PARENT_SCOPE)
set(ATen_VEC_TEST_SRCS ${ATen_VEC_TEST_SRCS} PARENT_SCOPE)
set(ATen_MPS_TEST_SRCS ${ATen_MPS_TEST_SRCS} PARENT_SCOPE)
set(ATen_XPU_TEST_SRCS ${ATen_XPU_TEST_SRCS} PARENT_SCOPE)
