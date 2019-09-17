set(TORCH_API_TEST_DIR "${TORCH_ROOT}/test/cpp/api")
set(TORCH_API_TEST_SOURCES
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${TORCH_API_TEST_DIR}/autograd.cpp
  ${TORCH_API_TEST_DIR}/any.cpp
  ${TORCH_API_TEST_DIR}/dataloader.cpp
  ${TORCH_API_TEST_DIR}/expanding-array.cpp
  ${TORCH_API_TEST_DIR}/functional.cpp
  ${TORCH_API_TEST_DIR}/integration.cpp
  ${TORCH_API_TEST_DIR}/init.cpp
  ${TORCH_API_TEST_DIR}/jit.cpp
  ${TORCH_API_TEST_DIR}/memory.cpp
  ${TORCH_API_TEST_DIR}/misc.cpp
  ${TORCH_API_TEST_DIR}/module.cpp
  ${TORCH_API_TEST_DIR}/modulelist.cpp
  ${TORCH_API_TEST_DIR}/modules.cpp
  ${TORCH_API_TEST_DIR}/optim.cpp
  ${TORCH_API_TEST_DIR}/ordered_dict.cpp
  ${TORCH_API_TEST_DIR}/rnn.cpp
  ${TORCH_API_TEST_DIR}/sequential.cpp
  ${TORCH_API_TEST_DIR}/serialize.cpp
  ${TORCH_API_TEST_DIR}/static.cpp
  ${TORCH_API_TEST_DIR}/tensor_cuda.cpp
  ${TORCH_API_TEST_DIR}/tensor_options_cuda.cpp
  ${TORCH_API_TEST_DIR}/tensor_options.cpp
  ${TORCH_API_TEST_DIR}/tensor.cpp
  ${TORCH_API_TEST_DIR}/torch_include.cpp
)

if (USE_CUDA)
  list(APPEND TORCH_API_TEST_SOURCES ${TORCH_API_TEST_DIR}/parallel.cpp)
endif()

add_executable(test_api ${TORCH_API_TEST_SOURCES})
target_include_directories(test_api PRIVATE ${ATen_CPU_INCLUDE})
target_link_libraries(test_api PRIVATE torch gtest)

if (USE_CUDA)
  target_link_libraries(test_api PRIVATE
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIB}
    ${CUDA_CUDA_LIB}
    ${TORCH_CUDA_LIBRARIES})

  target_compile_definitions(test_api PRIVATE "USE_CUDA")
endif()

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

if (INSTALL_TEST)
  install(TARGETS test_api DESTINATION bin)
  # Install PDB files for MSVC builds
  if (MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_api> DESTINATION bin OPTIONAL)
  endif()
endif()
