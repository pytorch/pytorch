set(TORCH_RPC_TEST_DIR "${TORCH_ROOT}/test/cpp/rpc")
set(TORCH_RPC_TEST_SOURCES
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${TORCH_RPC_TEST_DIR}/e2e_test_base.cpp
  ${TORCH_RPC_TEST_DIR}/test_wire_serialization.cpp
)
set(TORCH_RPC_TEST_DEPENDENCY_LIBS
  torch gtest
)

if(USE_GLOO)
  list(APPEND TORCH_RPC_TEST_SOURCES
    ${TORCH_RPC_TEST_DIR}/test_e2e_tensorpipe.cpp
  )
endif()

if(USE_TENSORPIPE)
  list(APPEND TORCH_RPC_TEST_SOURCES
    ${TORCH_RPC_TEST_DIR}/test_tensorpipe_serialization.cpp
  )
  list(APPEND TORCH_RPC_TEST_DEPENDENCY_LIBS
    tensorpipe
  )
endif()

add_executable(test_cpp_rpc ${TORCH_RPC_TEST_SOURCES})
target_include_directories(
  test_cpp_rpc PRIVATE
  ${ATen_CPU_INCLUDE})
target_include_directories(
  test_cpp_rpc PRIVATE
  $<BUILD_INTERFACE:${TORCH_SRC_DIR}/csrc/distributed>)
target_link_libraries(test_cpp_rpc PRIVATE ${TORCH_RPC_TEST_DEPENDENCY_LIBS})

if(USE_CUDA)
  target_compile_definitions(test_cpp_rpc PRIVATE "USE_CUDA")
endif()

if(INSTALL_TEST)
  install(TARGETS test_cpp_rpc DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_cpp_rpc> DESTINATION bin OPTIONAL)
  endif()
endif()
