set(MOBILE_NNC_TEST_ROOT ${TORCH_ROOT}/test/mobile/nnc)

set(MOBILE_NNC_TEST_SRCS
  ${MOBILE_NNC_TEST_ROOT}/test_context.cpp
  ${MOBILE_NNC_TEST_ROOT}/test_nnc_backend.cpp
  ${MOBILE_NNC_TEST_ROOT}/test_registry.cpp
)

add_executable(test_mobile_nnc
  ${TORCH_ROOT}/test/cpp/lite_interpreter_runtime/main.cpp
  ${MOBILE_NNC_TEST_SRCS})

add_library(nnc_backend_lib SHARED
  ${TORCH_ROOT}/torch/csrc/jit/backends/backend_interface.cpp
  ${TORCH_ROOT}/torch/csrc/jit/mobile/nnc/backend.cpp
  ${TORCH_ROOT}/torch/csrc/jit/mobile/nnc/context.cpp
  ${TORCH_ROOT}/torch/csrc/jit/mobile/nnc/registry.cpp
)
target_link_libraries(nnc_backend_lib PRIVATE torch)

target_link_libraries(test_mobile_nnc PRIVATE torch gtest nnc_backend_lib)
target_include_directories(test_mobile_nnc PRIVATE ${ATen_CPU_INCLUDE})
target_compile_definitions(test_mobile_nnc PRIVATE USE_GTEST)

if(INSTALL_TEST)
  install(TARGETS test_mobile_nnc DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_mobile_nnc> DESTINATION bin OPTIONAL)
  endif()
endif()
