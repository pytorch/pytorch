set(NATIVERT_TEST_ROOT ${TORCH_ROOT}/test/cpp/nativert)

# Build the cpp gtest binary containing the cpp-only tests.
set(NATIVERT_TEST_SRCS
  ${NATIVERT_TEST_ROOT}/test_tensor_meta.cpp
  ${TORCH_ROOT}/torch/nativert/graph/TensorMeta.cpp
)

add_executable(test_nativert
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${NATIVERT_TEST_SRCS}
)

# TODO temporary until we can delete the old gtest polyfills.
target_compile_definitions(test_nativert PRIVATE USE_GTEST)

set(NATIVERT_TEST_DEPENDENCIES torch gtest)

target_link_libraries(test_nativert PRIVATE ${NATIVERT_TEST_DEPENDENCIES})
target_include_directories(test_nativert PRIVATE ${ATen_CPU_INCLUDE})

if(USE_CUDA)
  target_compile_definitions(test_nativert PRIVATE USE_CUDA)
elseif(USE_ROCM)
  target_link_libraries(test_nativert PRIVATE
    hiprtc::hiprtc
    hip::amdhip64
    ${TORCH_CUDA_LIBRARIES})

  target_compile_definitions(test_nativert PRIVATE USE_ROCM)
endif()

if(INSTALL_TEST)
  set_target_properties(test_nativert PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
  install(TARGETS test_nativert DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_nativert> DESTINATION bin OPTIONAL)
  endif()
endif()
