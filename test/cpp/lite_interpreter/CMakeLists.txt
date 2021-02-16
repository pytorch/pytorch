set(LITE_INTERPRETER_TEST_DIR "${TORCH_ROOT}/test/cpp/lite_interpreter")
set(LITE_INTERPRETER_TEST_DIR
  ${TORCH_ROOT}/test/cpp/lite_interpreter/main.cpp
  ${TORCH_ROOT}/test/cpp/lite_interpreter/test_cores.cpp
)

add_executable(test_lite_interpreter ${LITE_INTERPRETER_TEST_DIR})
target_include_directories(test_lite_interpreter PRIVATE ${ATen_CPU_INCLUDE})
target_link_libraries(test_lite_interpreter PRIVATE torch gtest)

if(USE_CUDA)
  target_link_libraries(test_lite_interpreter PRIVATE
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIB}
    ${CUDA_CUDA_LIB}
    ${TORCH_CUDA_LIBRARIES})

  target_compile_definitions(test_lite_interpreter PRIVATE "USE_CUDA")
endif()

if(NOT MSVC)
  if(APPLE)
    target_compile_options(test_lite_interpreter PRIVATE
      # Clang has an unfixed bug leading to spurious missing braces
      # warnings, see https://bugs.llvm.org/show_bug.cgi?id=21629
      -Wno-missing-braces)
  else()
    target_compile_options(test_lite_interpreter PRIVATE
      # Considered to be flaky.  See the discussion at
      # https://github.com/pytorch/pytorch/pull/9608
      -Wno-maybe-uninitialized
      # gcc gives nonsensical warnings about variadic.h
      -Wno-unused-but-set-parameter)
  endif()
endif()

if(INSTALL_TEST)
  install(TARGETS test_lite_interpreter DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_lite_interpreter> DESTINATION bin OPTIONAL)
  endif()
endif()
