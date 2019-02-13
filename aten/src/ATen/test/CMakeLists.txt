IF (MSVC)
  IF (MSVC_VERSION LESS 1911)
    return()
  ENDIF()
ENDIF(MSVC)

list(APPEND ATen_CPU_TEST_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/scalar_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/apply_utils_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basic.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/atest.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/half_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/broadcast_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/wrapdim_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dlconvertor_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/native_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scalar_tensor_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tensor_interop_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_parallel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/undefined_tensor_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/verify_api_visibility.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tbb_init_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/weakref_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/extension_backend_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/xla_tensor_test.cpp)

list(APPEND ATen_CUDA_TEST_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_integer_divider_test.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rng_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_apply_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_stream_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_half_test.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_optional_test.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_packedtensoraccessor_test.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_tensor_interop_test.cpp)
if (CUDNN_FOUND)
  list(APPEND ATen_CUDA_TEST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_cudnn_test.cpp)
endif()

# ---[ Send the lists to the parent scope.
set(ATen_CPU_TEST_SRCS ${ATen_CPU_TEST_SRCS} PARENT_SCOPE)
set(ATen_CUDA_TEST_SRCS ${ATen_CUDA_TEST_SRCS} PARENT_SCOPE)
