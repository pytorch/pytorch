set(NATIVERT_TEST_ROOT ${TORCH_ROOT}/test/cpp/nativert)

file(GLOB_RECURSE NATIVERT_ALL_TEST_FILES "${NATIVERT_TEST_ROOT}/test_*.cpp")

# Build the cpp gtest binary containing the cpp-only tests.
set(NATIVERT_TEST_SRCS
  ${NATIVERT_ALL_TEST_FILES}
  ${TORCH_ROOT}/torch/nativert/graph/TensorMeta.cpp
  ${TORCH_ROOT}/torch/nativert/graph/Graph.cpp
  ${TORCH_ROOT}/torch/nativert/graph/GraphSignature.cpp
  ${TORCH_ROOT}/torch/nativert/graph/Serialization.cpp
  ${TORCH_ROOT}/torch/nativert/executor/OpKernel.cpp
  ${TORCH_ROOT}/torch/nativert/executor/PlacementUtils.cpp
  ${TORCH_ROOT}/torch/nativert/executor/Weights.cpp
  ${TORCH_ROOT}/torch/nativert/common/FileUtil.cpp
  ${TORCH_ROOT}/torch/nativert/executor/memory/FunctionSchema.cpp
  ${TORCH_ROOT}/torch/nativert/executor/ExecutionPlanner.cpp
  ${TORCH_ROOT}/torch/nativert/detail/ITree.cpp
  ${TORCH_ROOT}/torch/nativert/executor/ExecutionFrame.cpp
  ${TORCH_ROOT}/torch/nativert/kernels/C10Kernel.cpp
  ${TORCH_ROOT}/torch/nativert/executor/memory/GreedyBySize.cpp
  ${TORCH_ROOT}/torch/nativert/executor/memory/Bump.cpp
  ${TORCH_ROOT}/torch/nativert/executor/memory/DisjointStorageGroups.cpp
)

add_executable(test_nativert
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${NATIVERT_TEST_SRCS}
)

# TODO temporary until we can delete the old gtest polyfills.
target_compile_definitions(test_nativert PRIVATE USE_GTEST)

set(NATIVERT_TEST_DEPENDENCIES torch gtest_main)

target_link_libraries(test_nativert PRIVATE ${NATIVERT_TEST_DEPENDENCIES})
target_link_libraries(test_nativert PRIVATE fmt::fmt-header-only)
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
