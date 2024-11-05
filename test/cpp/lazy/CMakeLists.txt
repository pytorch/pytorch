set(LAZY_TEST_ROOT ${TORCH_ROOT}/test/cpp/lazy)

# Build the cpp gtest binary containing the cpp-only tests.
set(LAZY_TEST_SRCS
  ${LAZY_TEST_ROOT}/test_backend_device.cpp
  ${LAZY_TEST_ROOT}/test_cache.cpp
  ${LAZY_TEST_ROOT}/test_ir.cpp
  ${LAZY_TEST_ROOT}/test_ir_util.cpp
  ${LAZY_TEST_ROOT}/test_misc.cpp
  ${LAZY_TEST_ROOT}/test_permutation_util.cpp
  ${LAZY_TEST_ROOT}/test_shape.cpp
  ${LAZY_TEST_ROOT}/test_trie_cache.cpp
  ${LAZY_TEST_ROOT}/test_util.cpp
)
if(BUILD_LAZY_TS_BACKEND)
    list(APPEND LAZY_TEST_SRCS
      ${LAZY_TEST_ROOT}/test_lazy_ops.cpp
      ${LAZY_TEST_ROOT}/test_lazy_ops_util.cpp
    )
endif()

add_executable(test_lazy
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${LAZY_TEST_SRCS}
)

# TODO temporary until we can delete the old gtest polyfills.
target_compile_definitions(test_lazy PRIVATE USE_GTEST)

set(LAZY_TEST_DEPENDENCIES torch gtest)

target_link_libraries(test_lazy PRIVATE ${LAZY_TEST_DEPENDENCIES})
target_include_directories(test_lazy PRIVATE ${ATen_CPU_INCLUDE})

if(USE_CUDA)
  target_compile_definitions(test_lazy PRIVATE USE_CUDA)
elseif(USE_ROCM)
  target_link_libraries(test_lazy PRIVATE
    hiprtc::hiprtc
    hip::amdhip64
    ${TORCH_CUDA_LIBRARIES})

  target_compile_definitions(test_lazy PRIVATE USE_ROCM)
endif()

if(INSTALL_TEST)
  set_target_properties(test_lazy PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
  install(TARGETS test_lazy DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_lazy> DESTINATION bin OPTIONAL)
  endif()
endif()
