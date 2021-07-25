set(
  LITE_INTERPRETER_RUNTIME_TEST_DIR
  "${TORCH_ROOT}/test/cpp/lite_interpreter_runtime")
set(LITE_INTERPRETER_RUNTIME_TEST_DIR
  ${TORCH_ROOT}/test/cpp/lite_interpreter_runtime/main.cpp
  ${TORCH_ROOT}/test/cpp/lite_interpreter_runtime/test_lite_interpreter_runtime.cpp
)

add_library(backend_with_compiler_runtime SHARED
        ${TORCH_ROOT}/test/cpp/jit/test_backend_compiler_lib.cpp
        ${TORCH_ROOT}/torch/csrc/jit/backends/backend_interface.cpp
        )
target_link_libraries(backend_with_compiler_runtime PRIVATE torch)

add_executable(
  test_lite_interpreter_runtime
  ${LITE_INTERPRETER_RUNTIME_TEST_DIR})
target_include_directories(
  test_lite_interpreter_runtime PRIVATE
  ${ATen_CPU_INCLUDE}
)

target_link_libraries(test_lite_interpreter_runtime PRIVATE torch gtest backend_with_compiler_runtime)

if(INSTALL_TEST)
  install(TARGETS test_lite_interpreter_runtime DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(
      FILES $<TARGET_PDB_FILE:test_lite_interpreter_runtime>
      DESTINATION bin OPTIONAL)
  endif()
endif()
